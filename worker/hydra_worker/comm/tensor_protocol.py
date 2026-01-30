"""Custom binary tensor serialization protocol.

Wire format:
    Header (40 bytes):
        Magic(4) | Version(1) | Flags(1) | MsgType(2) |
        SequenceID(8) | BatchID(8) | Timestamp(8) | TensorCount(2) | Reserved(6)

    Per-Tensor Descriptor (32 bytes):
        DType(1) | NDim(1) | Device(1) | Layout(1) |
        Shape[4](16) | Offset(8) | Size(8)

    Tensor Data (contiguous raw bytes)
"""

import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import msgpack


# Protocol constants
MAGIC = 0x48594452  # "HYDR"
VERSION = 1
HEADER_SIZE = 40
# Tensor descriptor: BBBB (4) + 4I (16) + QQ (16) = 36 bytes
TENSOR_DESC_SIZE = 36

# Message types
MSG_TYPE_HIDDEN_STATE = 0x0001
MSG_TYPE_GRADIENT = 0x0002
MSG_TYPE_CONTROL = 0x0003

# Data type mapping
DTYPE_MAP = {
    torch.float32: (0, np.float32, 4),
    torch.float16: (1, np.float16, 2),
    torch.bfloat16: (2, np.float16, 2),  # Serialize as float16
    torch.int8: (3, np.int8, 1),
    torch.int32: (4, np.int32, 4),
    torch.int64: (5, np.int64, 8),
}

DTYPE_REVERSE = {
    0: (torch.float32, np.float32),
    1: (torch.float16, np.float16),
    2: (torch.bfloat16, np.float16),
    3: (torch.int8, np.int8),
    4: (torch.int32, np.int32),
    5: (torch.int64, np.int64),
}


@dataclass
class TensorHeader:
    """Header for a single tensor."""
    dtype: int
    ndim: int
    device: int
    layout: int
    shape: Tuple[int, ...]
    offset: int
    size: int


@dataclass
class TensorMessage:
    """Message containing one or more tensors."""
    sequence_id: int = 0
    batch_id: int = 0
    timestamp: int = 0
    msg_type: int = MSG_TYPE_HIDDEN_STATE
    flags: int = 0
    tensors: List[Tuple[TensorHeader, memoryview]] = None

    def __post_init__(self):
        if self.tensors is None:
            self.tensors = []


class TensorSerializer:
    """Efficient tensor serialization with minimal copying."""

    @classmethod
    def serialize(
        cls,
        tensors: List[torch.Tensor] | torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        batch_id: int = 0,
        sequence_id: int = 0,
        msg_type: int = MSG_TYPE_HIDDEN_STATE,
    ) -> bytes:
        """Serialize tensor(s) to wire format.

        Args:
            tensors: Single tensor or list of tensors
            metadata: Optional metadata (will be msgpacked into reserved bytes)
            batch_id: Batch identifier
            sequence_id: Sequence identifier
            msg_type: Message type

        Returns:
            Serialized bytes
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]

        tensor_count = len(tensors)
        header_total = HEADER_SIZE + tensor_count * TENSOR_DESC_SIZE

        # Calculate data sizes
        tensor_data = []
        for tensor in tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            # Move to CPU for serialization
            cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor

            # Handle bfloat16 by converting to float16
            if tensor.dtype == torch.bfloat16:
                cpu_tensor = cpu_tensor.to(torch.float16)

            tensor_data.append(cpu_tensor.numpy().tobytes())

        data_size = sum(len(d) for d in tensor_data)
        total_size = header_total + data_size

        # Allocate buffer
        buffer = bytearray(total_size)

        # Pack main header (big-endian for network)
        timestamp = time.time_ns()
        struct.pack_into(
            ">IBBHQQQI6x",  # 40 bytes
            buffer,
            0,
            MAGIC,
            VERSION,
            0,  # flags
            msg_type,
            sequence_id,
            batch_id,
            timestamp,
            tensor_count,
        )

        # Pack tensor descriptors
        offset = HEADER_SIZE
        data_offset = 0

        for i, tensor in enumerate(tensors):
            dtype_info = DTYPE_MAP.get(tensor.dtype, DTYPE_MAP[torch.float32])
            dtype_code = dtype_info[0]

            shape = list(tensor.shape) + [0] * (4 - tensor.ndim)
            data_size = len(tensor_data[i])

            struct.pack_into(
                ">BBBB4IQQ",  # 32 bytes
                buffer,
                offset,
                dtype_code,
                tensor.ndim,
                0,  # device (0 = CPU after transfer)
                0,  # layout (0 = contiguous)
                shape[0],
                shape[1] if len(shape) > 1 else 0,
                shape[2] if len(shape) > 2 else 0,
                shape[3] if len(shape) > 3 else 0,
                data_offset,
                data_size,
            )

            offset += TENSOR_DESC_SIZE
            data_offset += data_size

        # Copy tensor data
        data_start = header_total
        for td in tensor_data:
            buffer[data_start : data_start + len(td)] = td
            data_start += len(td)

        # Optionally append metadata
        if metadata:
            meta_bytes = msgpack.packb(metadata)
            # Append to end of buffer
            buffer.extend(meta_bytes)

        return bytes(buffer)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Deserialize bytes back to tensor(s).

        Args:
            data: Serialized bytes
            device: Target device for tensors

        Returns:
            Tuple of (list of tensors, metadata dict)
        """
        if len(data) < HEADER_SIZE:
            raise ValueError("Data too short for header")

        # Parse main header
        (
            magic,
            version,
            flags,
            msg_type,
            sequence_id,
            batch_id,
            timestamp,
            tensor_count,
        ) = struct.unpack_from(">IBBHQQQI6x", data, 0)

        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        header_total = HEADER_SIZE + tensor_count * TENSOR_DESC_SIZE

        # Parse tensor descriptors
        tensors = []
        offset = HEADER_SIZE

        for _ in range(tensor_count):
            (
                dtype_code,
                ndim,
                device_code,
                layout,
                s0,
                s1,
                s2,
                s3,
                data_offset,
                data_size,
            ) = struct.unpack_from(">BBBB4IQQ", data, offset)

            shape = (s0, s1, s2, s3)[:ndim]
            offset += TENSOR_DESC_SIZE

            # Get tensor data
            tensor_start = header_total + data_offset
            tensor_bytes = data[tensor_start : tensor_start + data_size]

            # Reconstruct tensor
            torch_dtype, np_dtype = DTYPE_REVERSE[dtype_code]

            # Calculate expected size
            element_size = np.dtype(np_dtype).itemsize
            expected_elements = 1
            for s in shape:
                expected_elements *= s
            expected_size = expected_elements * element_size

            if len(tensor_bytes) != expected_size:
                raise ValueError(
                    f"Tensor data size mismatch: got {len(tensor_bytes)} bytes, "
                    f"expected {expected_size} bytes for shape {shape} dtype {np_dtype}"
                )

            arr = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(arr.copy())

            # Convert back to bfloat16 if needed
            if dtype_code == 2:  # Was bfloat16
                tensor = tensor.to(torch.bfloat16)

            tensors.append(tensor.to(device))

        # Parse metadata if present
        # data_size is at offset 28 in each descriptor (after BBBB=4, 4I=16, Q=8)
        expected_data_end = header_total + sum(
            struct.unpack_from(">Q", data, HEADER_SIZE + i * TENSOR_DESC_SIZE + 28)[0]
            for i in range(tensor_count)
        )

        metadata = {}
        if len(data) > expected_data_end:
            try:
                metadata = msgpack.unpackb(data[expected_data_end:])
            except Exception:
                pass  # No metadata or invalid

        return tensors, metadata

    @classmethod
    def serialize_single(
        cls,
        tensor: torch.Tensor,
        sequence_id: str = "",
        position: int = 0,
    ) -> bytes:
        """Convenience method for single tensor with common metadata."""
        return cls.serialize(
            tensor,
            metadata={"sequence_id": sequence_id, "position": position},
        )

    @classmethod
    def deserialize_single(
        cls,
        data: bytes,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convenience method for single tensor."""
        tensors, metadata = cls.deserialize(data, device)
        return tensors[0] if tensors else None, metadata
