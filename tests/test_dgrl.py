"""
测试 DGRL 格式解码器（decoders/dgrl.py）
"""

import struct
import random
import numpy as np
import pytest

from py_casia_tool.decoders.dgrl import decode_dgrl
from py_casia_tool.types import DgrlLine
from fake_data_generator import generate_dgrl_bytes


@pytest.fixture
def fake_dgrl_bytes():
    """生成一个包含 2 行文本的合法 DGRL 字节流"""
    return generate_dgrl_bytes(image_height=300, image_width=400, line_num=2, chars_per_line=[2, 3])


class TestDecodeDgrl:

    def test_sample_count(self, fake_dgrl_bytes):
        """验证解码出的样本数量正确（2行）"""
        samples = decode_dgrl(fake_dgrl_bytes)
        assert len(samples) == 2

    def test_return_type(self, fake_dgrl_bytes):
        """验证每个元素都是 DgrlLine 类型"""
        samples = decode_dgrl(fake_dgrl_bytes)
        for s in samples:
            assert isinstance(s, DgrlLine)

    def test_image_dimensions(self, fake_dgrl_bytes):
        """验证图像尺寸"""
        samples = decode_dgrl(fake_dgrl_bytes)
        first = samples[0]
        assert first.image_width == 400
        assert first.image_height == 300

    def test_first_sample_fields(self, fake_dgrl_bytes):
        """验证第一行的字段：label 是整行文本"""
        samples = decode_dgrl(fake_dgrl_bytes)
        first = samples[0]
        assert first.label == "啊阿"
        assert first.width == 320
        assert first.height == 48
        assert first.bitmap.shape == (48, 320)

    def test_last_sample_label(self, fake_dgrl_bytes):
        """验证第二行（最后一行）的标签"""
        samples = decode_dgrl(fake_dgrl_bytes)
        last = samples[-1]
        assert last.label == "埃挨哎"

    def test_line_number_increment(self, fake_dgrl_bytes):
        """验证行号递增"""
        samples = decode_dgrl(fake_dgrl_bytes)
        assert samples[0].line_number == 1
        assert samples[1].line_number == 2

    def test_empty_input(self):
        """空字节流应抛出异常"""
        with pytest.raises(ValueError):
            decode_dgrl(b"")

    def test_invalid_format_code(self):
        """无效格式标识应抛出异常"""
        data = struct.pack("<I", 54)
        data += b"INVALID" + b"\x00" * 2
        data += b"#Test\0"
        data += b"GB" + b"\x00" * 18
        data += struct.pack("<HH", 2, 8)
        with pytest.raises(ValueError, match="无效的 DGRL 格式标识"):
            decode_dgrl(data)

    def test_bitmap_not_all_zero(self, fake_dgrl_bytes):
        """验证 bitmap 不是全零"""
        samples = decode_dgrl(fake_dgrl_bytes)
        for s in samples:
            assert np.any(s.bitmap != 0), "bitmap 不应全零"

    def test_bits_per_pixel_1_bitmap_shape(self):
        """验证 bits_per_pixel=1 时 bitmap 形状正确（二值图像解压）"""
        code_length = 2
        bit_depth = 1
        image_height = 300
        image_width = 400
        line_num = 1

        szIllustr = "#DGRL Test\0"
        szCodeType = "GB" + "\x00" * 18
        header_size = 4 + 8 + len(szIllustr) + 20 + 2 + 2

        data = struct.pack("<I", header_size)
        data += b"DGRL" + b"\x00" * 4
        data += szIllustr.encode("ascii")
        data += szCodeType.encode("ascii")
        data += struct.pack("<HH", code_length, bit_depth)
        data += struct.pack("<III", image_height, image_width, line_num)

        char_num = 2
        data += struct.pack("<I", char_num)
        data += struct.pack(">H", 0xB0A1)
        data += struct.pack(">H", 0xB0A2)

        top, left, height, width = 10, 10, 48, 32
        data += struct.pack("<IIII", top, left, height, width)

        bitmap_size = height * (width + 7) // 8
        bitmap = bytes(random.randint(1, 255) for _ in range(bitmap_size))
        data += bitmap

        samples = decode_dgrl(data)
        assert len(samples) == 1
        assert samples[0].bitmap.shape == (height, width)
        assert samples[0].bitmap.dtype == np.uint8
