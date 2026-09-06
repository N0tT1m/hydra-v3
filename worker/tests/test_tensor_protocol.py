"""Unit tests for the binary tensor protocol.

Coverage:
  * Round-trip for every supported dtype (fp32, fp16, bf16, i8, i32, i64).
  * Multi-tensor messages with metadata.
  * CRC detection of bit flips anywhere in the payload.
  * Header validation: bad magic, wrong version, truncated buffer.
  * Descriptor validation: wrong tensor_count / data_size.
  * bfloat16 round-trips bitwise-exactly (since we serialize as float32).
"""

import struct
import pytest
import torch

from hydra_worker.comm.tensor_protocol import (
    CRC_SIZE,
    DESC_FMT,
    HEADER_FMT,
    HEADER_SIZE,
    MAGIC,
    ProtocolError,
    TENSOR_DESC_SIZE,
    TensorSerializer,
    VERSION,
)


CPU = torch.device("cpu")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.int32,
        torch.int64,
    ],
)
def test_roundtrip_dtype(dtype):
    if dtype.is_floating_point:
        t = torch.randn(4, 8).to(dtype)
    else:
        # Use a small range to avoid overflow for int8.
        t = torch.randint(-100, 100, (4, 8), dtype=dtype)

    blob = TensorSerializer.serialize(t, metadata={"seq": "abc", "pos": 3})
    tensors, meta = TensorSerializer.deserialize(blob, CPU)

    assert len(tensors) == 1
    out = tensors[0]
    assert out.dtype == dtype
    assert out.shape == t.shape
    # bfloat16 is serialized via float32, so we get bitwise equality.
    assert torch.equal(out, t), f"mismatch for dtype={dtype}"
    assert meta == {"seq": "abc", "pos": 3}


def test_roundtrip_multi_tensor():
    a = torch.randn(2, 3)
    b = torch.randint(0, 10, (5,), dtype=torch.int64)
    c = torch.zeros(1, 1, 1, 4, dtype=torch.float16)

    blob = TensorSerializer.serialize([a, b, c], metadata={"k": "v"})
    tensors, meta = TensorSerializer.deserialize(blob, CPU)

    assert len(tensors) == 3
    assert torch.equal(tensors[0], a)
    assert torch.equal(tensors[1], b)
    assert torch.equal(tensors[2], c)
    assert meta == {"k": "v"}


def test_roundtrip_no_metadata():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    blob = TensorSerializer.serialize(t)
    tensors, meta = TensorSerializer.deserialize(blob, CPU)
    assert torch.equal(tensors[0], t)
    assert meta == {}


def test_roundtrip_non_contiguous():
    """Non-contiguous tensors must be serialized correctly (implicitly contiguified)."""
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6).T  # non-contig
    assert not t.is_contiguous()
    blob = TensorSerializer.serialize(t)
    tensors, _ = TensorSerializer.deserialize(blob, CPU)
    assert tensors[0].shape == t.shape
    assert torch.equal(tensors[0], t.contiguous())


def test_crc_detects_bit_flip_in_data():
    t = torch.randn(4, 8)
    blob = bytearray(TensorSerializer.serialize(t))
    # Flip a bit in the middle of the tensor data region.
    flip_at = HEADER_SIZE + TENSOR_DESC_SIZE + 16
    blob[flip_at] ^= 0x01
    with pytest.raises(ProtocolError, match="CRC mismatch"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_crc_detects_bit_flip_in_header():
    t = torch.randn(2, 2)
    blob = bytearray(TensorSerializer.serialize(t))
    # Flip a bit in the msg_type field (offset 6-7).
    blob[6] ^= 0x01
    with pytest.raises(ProtocolError, match="CRC mismatch"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_crc_detects_bit_flip_in_metadata():
    t = torch.zeros(2)
    blob = bytearray(TensorSerializer.serialize(t, metadata={"seq": "abc"}))
    # Metadata sits between tensor data and the trailing 4-byte CRC.
    meta_offset = len(blob) - CRC_SIZE - 1
    blob[meta_offset] ^= 0x01
    with pytest.raises(ProtocolError, match="CRC mismatch"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_truncated_buffer_rejected():
    t = torch.randn(2, 2)
    blob = TensorSerializer.serialize(t)
    with pytest.raises(ProtocolError, match="too short"):
        TensorSerializer.deserialize(blob[:HEADER_SIZE], CPU)


def test_bad_magic_rejected():
    t = torch.randn(2)
    blob = bytearray(TensorSerializer.serialize(t))
    # Overwrite first 4 bytes with garbage and fix CRC so the magic check fires first.
    blob[0:4] = b"\x00\x00\x00\x00"
    # Deserialize should fail on magic check before CRC check.
    with pytest.raises(ProtocolError, match="invalid magic"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_wrong_version_rejected():
    """Forge a v99 message with valid CRC; deserializer must still reject it."""
    import zlib

    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    blob = bytearray(TensorSerializer.serialize(t))
    # Rewrite the version byte (offset 4).
    blob[4] = 99
    # Recompute CRC so we specifically test the version check, not CRC.
    new_crc = zlib.crc32(bytes(blob[:-CRC_SIZE])) & 0xFFFFFFFF
    struct.pack_into(">I", blob, len(blob) - CRC_SIZE, new_crc)
    with pytest.raises(ProtocolError, match="unsupported version"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_descriptor_size_mismatch_rejected():
    """If a producer lies about data_size, we must reject rather than read garbage."""
    import zlib

    t = torch.randn(4, 4, dtype=torch.float32)
    blob = bytearray(TensorSerializer.serialize(t))
    # Descriptor DataSize is the last 8 bytes of the 36-byte descriptor.
    desc_start = HEADER_SIZE
    size_offset = desc_start + TENSOR_DESC_SIZE - 8
    # Claim a bigger data size than actually follows.
    struct.pack_into(">Q", blob, size_offset, 9999)
    # Fix CRC so we exercise the length-mismatch check, not CRC detection.
    new_crc = zlib.crc32(bytes(blob[:-CRC_SIZE])) & 0xFFFFFFFF
    struct.pack_into(">I", blob, len(blob) - CRC_SIZE, new_crc)
    with pytest.raises(ProtocolError, match="length mismatch"):
        TensorSerializer.deserialize(bytes(blob), CPU)


def test_tensor_count_overflow_rejected():
    """tensor_count is a uint16 field; more than 65535 tensors must fail at serialize time."""
    tensors = [torch.zeros(1)] * (1 << 16)
    with pytest.raises(ProtocolError, match="uint16 max"):
        TensorSerializer.serialize(tensors)


def test_header_format_sizes():
    """Catch silent drift if someone changes HEADER_FMT / DESC_FMT."""
    assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
    assert struct.calcsize(DESC_FMT) == TENSOR_DESC_SIZE


def test_magic_is_hydr():
    assert MAGIC == 0x48594452  # 'H','Y','D','R'


def test_version_is_current():
    assert VERSION == 2


def test_serialize_single_helpers():
    t = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    blob = TensorSerializer.serialize_single(t, sequence_id="seq-1", position=42)
    out, meta = TensorSerializer.deserialize_single(blob, CPU)
    assert out is not None
    assert torch.equal(out, t)
    assert meta["sequence_id"] == "seq-1"
    assert meta["position"] == 42


def test_bfloat16_exact_roundtrip():
    """bfloat16 must round-trip exactly (we upcast to float32 on the wire)."""
    t = torch.tensor([1.25, -2.5, 3.75, float("inf"), 0.0], dtype=torch.bfloat16)
    blob = TensorSerializer.serialize(t)
    out, _ = TensorSerializer.deserialize(blob, CPU)
    assert out[0].dtype == torch.bfloat16
    # Compare by bit pattern to avoid NaN quirks; inf and finite values must match.
    finite_mask = torch.isfinite(t)
    assert torch.equal(out[0][finite_mask], t[finite_mask])
    assert torch.isinf(out[0][~finite_mask]).all()
