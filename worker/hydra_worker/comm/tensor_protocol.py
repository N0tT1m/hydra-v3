"""Custom binary tensor serialization protocol.

Wire format (v2):
    Header (40 bytes):
        Magic(4) | Version(1) | Flags(1) | MsgType(2) |
        SequenceID(8) | BatchID(8) | Timestamp(8) |
        TensorCount(2) | MetadataSize(4) | Reserved(2)

    Per-tensor descriptor (36 bytes, TensorCount of them):
        DType(1) | NDim(1) | Device(1) | Layout(1) |
        Shape[4](16) | DataOffset(8) | DataSize(8)

    Tensor data: raw bytes, contiguous, length = sum(descriptor.DataSize)
    Metadata: msgpack bytes, length = MetadataSize (0 if absent)
    Footer: CRC32(everything above), 4 bytes big-endian

Version history:
    v1: no CRC, no explicit metadata size (metadata ran to end-of-buffer).
    v2 (current): CRC footer + explicit metadata size. v1 is rejected.
"""

import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import msgpack
import structlog


log = structlog.get_logger()


MAGIC = 0x48594452  # "HYDR"
VERSION = 2
HEADER_SIZE = 40
TENSOR_DESC_SIZE = 36
CRC_SIZE = 4

HEADER_FMT = ">IBBHQQQHI2x"
DESC_FMT = ">BBBB4IQQ"

assert struct.calcsize(HEADER_FMT) == HEADER_SIZE, "HEADER_FMT size drift"
assert struct.calcsize(DESC_FMT) == TENSOR_DESC_SIZE, "DESC_FMT size drift"

MSG_TYPE_HIDDEN_STATE = 0x0001
MSG_TYPE_GRADIENT = 0x0002
MSG_TYPE_CONTROL = 0x0003

# bfloat16 is serialized as float32 on the wire (numpy has no bfloat16).
# Deserializer casts back to bfloat16 based on the dtype_code.
DTYPE_MAP = {
    torch.float32: (0, np.float32, 4),
    torch.float16: (1, np.float16, 2),
    torch.bfloat16: (2, np.float32, 4),
    torch.int8: (3, np.int8, 1),
    torch.int32: (4, np.int32, 4),
    torch.int64: (5, np.int64, 8),
}

DTYPE_REVERSE = {
    0: (torch.float32, np.float32),
    1: (torch.float16, np.float16),
    2: (torch.bfloat16, np.float32),
    3: (torch.int8, np.int8),
    4: (torch.int32, np.int32),
    5: (torch.int64, np.int64),
}


class ProtocolError(ValueError):
    """Raised when the wire format is malformed or corrupted."""


@dataclass
class TensorHeader:
    dtype: int
    ndim: int
    device: int
    layout: int
    shape: Tuple[int, ...]
    offset: int
    size: int


@dataclass
class TensorMessage:
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
        tensors: "List[torch.Tensor] | torch.Tensor",
        metadata: Optional[Dict[str, Any]] = None,
        batch_id: int = 0,
        sequence_id: int = 0,
        msg_type: int = MSG_TYPE_HIDDEN_STATE,
    ) -> bytes:
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]

        tensor_count = len(tensors)
        if tensor_count > 0xFFFF:
            raise ProtocolError(f"tensor_count {tensor_count} exceeds uint16 max")

        header_total = HEADER_SIZE + tensor_count * TENSOR_DESC_SIZE

        tensor_data: List[bytes] = []
        for tensor in tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
            if cpu_tensor.dtype == torch.bfloat16:
                cpu_tensor = cpu_tensor.to(torch.float32)
            tensor_data.append(cpu_tensor.numpy().tobytes())

        data_size = sum(len(d) for d in tensor_data)

        meta_bytes = b""
        if metadata:
            meta_bytes = msgpack.packb(metadata, use_bin_type=True)
            if len(meta_bytes) > 0xFFFFFFFF:
                raise ProtocolError("metadata exceeds 4GiB")

        total_size = header_total + data_size + len(meta_bytes) + CRC_SIZE
        buffer = bytearray(total_size)

        timestamp = time.time_ns() & 0xFFFFFFFFFFFFFFFF
        struct.pack_into(
            HEADER_FMT,
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
            len(meta_bytes),
        )

        offset = HEADER_SIZE
        data_offset = 0
        for i, tensor in enumerate(tensors):
            dtype_info = DTYPE_MAP.get(tensor.dtype)
            if dtype_info is None:
                raise ProtocolError(f"unsupported dtype {tensor.dtype}")
            dtype_code = dtype_info[0]

            if tensor.ndim > 4:
                raise ProtocolError(f"tensor ndim {tensor.ndim} exceeds max 4")
            shape = list(tensor.shape) + [0] * (4 - tensor.ndim)
            tsize = len(tensor_data[i])

            struct.pack_into(
                DESC_FMT,
                buffer,
                offset,
                dtype_code,
                tensor.ndim,
                0,  # device (0 = CPU on the wire)
                0,  # layout (0 = contiguous)
                shape[0], shape[1], shape[2], shape[3],
                data_offset,
                tsize,
            )
            offset += TENSOR_DESC_SIZE
            data_offset += tsize

        pos = header_total
        for td in tensor_data:
            buffer[pos:pos + len(td)] = td
            pos += len(td)

        if meta_bytes:
            buffer[pos:pos + len(meta_bytes)] = meta_bytes
            pos += len(meta_bytes)

        crc = zlib.crc32(bytes(buffer[:pos])) & 0xFFFFFFFF
        struct.pack_into(">I", buffer, pos, crc)

        return bytes(buffer)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        if len(data) < HEADER_SIZE + CRC_SIZE:
            raise ProtocolError(
                f"data too short: {len(data)} bytes, need at least {HEADER_SIZE + CRC_SIZE}"
            )

        (
            magic, version, flags, msg_type,
            sequence_id, batch_id, timestamp,
            tensor_count, meta_size,
        ) = struct.unpack_from(HEADER_FMT, data, 0)

        if magic != MAGIC:
            raise ProtocolError(f"invalid magic {hex(magic)} (expected {hex(MAGIC)})")
        if version != VERSION:
            raise ProtocolError(f"unsupported version {version} (expected {VERSION})")

        header_total = HEADER_SIZE + tensor_count * TENSOR_DESC_SIZE
        if len(data) < header_total + CRC_SIZE:
            raise ProtocolError(
                f"data too short for {tensor_count} descriptors: "
                f"{len(data)} bytes, need {header_total + CRC_SIZE}"
            )

        expected_crc = struct.unpack_from(">I", data, len(data) - CRC_SIZE)[0]
        actual_crc = zlib.crc32(data[:-CRC_SIZE]) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ProtocolError(
                f"CRC mismatch: got {hex(actual_crc)}, expected {hex(expected_crc)}"
            )

        descriptors = []
        total_tensor_bytes = 0
        offset = HEADER_SIZE
        for _ in range(tensor_count):
            (dtype_code, ndim, device_code, layout,
             s0, s1, s2, s3,
             data_offset, data_size) = struct.unpack_from(DESC_FMT, data, offset)

            if ndim > 4:
                raise ProtocolError(f"descriptor ndim {ndim} exceeds max 4")
            if dtype_code not in DTYPE_REVERSE:
                raise ProtocolError(f"unknown dtype_code {dtype_code}")
            shape = (s0, s1, s2, s3)[:ndim]
            descriptors.append((dtype_code, ndim, shape, data_offset, data_size))
            total_tensor_bytes += data_size
            offset += TENSOR_DESC_SIZE

        expected_total = header_total + total_tensor_bytes + meta_size + CRC_SIZE
        if len(data) != expected_total:
            raise ProtocolError(
                f"length mismatch: got {len(data)}, expected {expected_total} "
                f"(header={header_total}, tensors={total_tensor_bytes}, "
                f"meta={meta_size}, crc={CRC_SIZE})"
            )

        tensors: List[torch.Tensor] = []
        for dtype_code, ndim, shape, data_offset, data_size in descriptors:
            torch_dtype, np_dtype = DTYPE_REVERSE[dtype_code]
            element_size = np.dtype(np_dtype).itemsize
            expected_elements = 1
            for s in shape:
                expected_elements *= s
            expected_size = expected_elements * element_size
            if data_size != expected_size:
                raise ProtocolError(
                    f"tensor data size mismatch: descriptor says {data_size}, "
                    f"shape {shape} dtype {np_dtype} needs {expected_size}"
                )

            tensor_start = header_total + data_offset
            tensor_end = tensor_start + data_size
            if tensor_end > header_total + total_tensor_bytes:
                raise ProtocolError(
                    f"tensor slice [{tensor_start}:{tensor_end}] exceeds data section"
                )
            tensor_bytes = data[tensor_start:tensor_end]

            arr = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(arr.copy())
            if dtype_code == 2:  # bfloat16
                tensor = tensor.to(torch.bfloat16)
            tensors.append(tensor.to(device))

        metadata: Dict[str, Any] = {}
        if meta_size > 0:
            meta_start = header_total + total_tensor_bytes
            meta_end = meta_start + meta_size
            try:
                metadata = msgpack.unpackb(data[meta_start:meta_end], raw=False)
            except Exception as e:
                raise ProtocolError(f"failed to decode metadata: {e}") from e

        return tensors, metadata

    @classmethod
    def serialize_single(
        cls,
        tensor: torch.Tensor,
        sequence_id: str = "",
        position: int = 0,
    ) -> bytes:
        return cls.serialize(
            tensor,
            metadata={"sequence_id": sequence_id, "position": position},
        )

    @classmethod
    def deserialize_single(
        cls,
        data: bytes,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        tensors, metadata = cls.deserialize(data, device)
        return (tensors[0] if tensors else None), metadata
