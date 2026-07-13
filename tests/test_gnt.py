"""
测试 GNT 格式解码器（decoders/gnt.py）
"""

import struct
import random
import numpy as np
import pytest
from pathlib import Path

from py_casia_tool.decoders.gnt import decode_gnt
from py_casia_tool.types import GntChar


# ─────────────── Fixture：伪造合法 GNT 字节流 ───────────────


@pytest.fixture
def fake_gnt_bytes():
    """
    生成一个包含 3 个样本的合法 GNT 字节流。
    每个样本：total_size(4B) + tag(2B) + w(2B) + h(2B) + bitmap(w*h B)
    tag 从 0xB0A1 开始递增，尺寸固定 72x48。
    """
    width = 72
    height = 48
    bitmap_size = width * height

    data = b""
    for i in range(3):
        tag = 0xB0A1 + i
        total_size = 2 + 2 + 2 + bitmap_size

        data += struct.pack("<I", total_size)
        data += struct.pack(">H", tag)
        data += struct.pack("<HH", width, height)
        bitmap = bytes(random.randint(1, 255) for _ in range(bitmap_size))
        data += bitmap

    return data


# ─────────────── 测试 decode_gnt 主功能 ───────────────


class TestDecodeGnt:

    def test_sample_count(self, fake_gnt_bytes):
        """验证解码出的样本数量正确"""
        samples = decode_gnt(fake_gnt_bytes)
        assert len(samples) == 3

    def test_return_type(self, fake_gnt_bytes):
        """验证每个元素都是 GntChar 类型"""
        samples = decode_gnt(fake_gnt_bytes)
        for s in samples:
            assert isinstance(s, GntChar)

    def test_first_sample_fields(self, fake_gnt_bytes):
        """验证第一个样本的标签、宽高、bitmap 长度"""
        samples = decode_gnt(fake_gnt_bytes)
        first = samples[0]
        assert first.tag_code == "啊"
        assert first.width == 72
        assert first.height == 48
        assert first.bitmap.size == 48 * 72

    def test_last_sample_label(self, fake_gnt_bytes):
        """验证最后一个样本的标签递增"""
        samples = decode_gnt(fake_gnt_bytes)
        last = samples[-1]
        assert last.tag_code == "埃"

    def test_empty_input(self):
        """空字节流应返回空列表"""
        assert decode_gnt(b"") == []

    def test_truncated_input_raises(self):
        """不完整的数据应抛出异常（或返回部分样本，取决于设计）"""
        truncated = b"\x0c\x00\x00\x00\xa1\xb0\x48\x00\x30\x00"
        with pytest.raises(Exception):
            decode_gnt(truncated)

    def test_bitmap_not_all_zero(self, fake_gnt_bytes):
        """验证 bitmap 不是全零（因为用了随机数据）"""
        samples = decode_gnt(fake_gnt_bytes)
        for s in samples:
            assert np.any(s.bitmap != 0), "bitmap 不应全零"
