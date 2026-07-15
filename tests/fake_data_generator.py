"""
伪造数据生成器

提供生成合法 GNT 和 DGRL 格式字节流的函数，用于单元测试。
"""

import struct
import random


def generate_gnt_bytes(num_samples: int = 3, width: int = 72, height: int = 48) -> bytes:
    """
    生成合法的 GNT 格式字节流。

    每个样本结构：total_size(4B) + tag(2B) + width(2B) + height(2B) + bitmap(w*h B)

    Args:
        num_samples: 样本数量，默认 3
        width: 字符宽度，默认 72
        height: 字符高度，默认 48

    Returns:
        bytes: GNT 格式字节流
    """
    bitmap_size = width * height
    data = b""
    for i in range(num_samples):
        tag = 0xB0A1 + i
        total_size = 2 + 2 + 2 + bitmap_size
        data += struct.pack("<I", total_size)
        data += struct.pack(">H", tag)
        data += struct.pack("<HH", width, height)
        bitmap = bytes(random.randint(1, 255) for _ in range(bitmap_size))
        data += bitmap
    return data


def generate_dgrl_bytes(
    image_height: int = 300,
    image_width: int = 400,
    line_num: int = 2,
    chars_per_line: list = None,
) -> bytes:
    """
    生成合法的 DGRL 格式字节流。

    文件头：header_size(4B) + format_code(8B) + szIllustr + szCodeType(20B)
            + code_length(2B) + bit_depth(2B) + image_height(4B) + image_width(4B) + line_num(4B)

    每行结构：char_num(4B) + 所有标签(code_length*char_num) + 位置尺寸(16B) + 位图(h*w)

    Args:
        image_height: 图像高度，默认 300
        image_width: 图像宽度，默认 400
        line_num: 行数，默认 2
        chars_per_line: 每行字符数列表，默认为 [2, 3]

    Returns:
        bytes: DGRL 格式字节流
    """
    if chars_per_line is None:
        chars_per_line = [2, 3]

    code_length = 2
    bit_depth = 8
    szIllustr = "#DGRL Test\0"
    szCodeType = "GB" + "\x00" * 18
    header_size = 4 + 8 + len(szIllustr) + 20 + 2 + 2

    data = struct.pack("<I", header_size)
    data += b"DGRL" + b"\x00" * 4
    data += szIllustr.encode("ascii")
    data += szCodeType.encode("ascii")
    data += struct.pack("<HH", code_length, bit_depth)
    data += struct.pack("<III", image_height, image_width, line_num)

    total_chars = sum(chars_per_line)
    tags = [0xB0A1 + i for i in range(total_chars)]
    tag_index = 0

    for line_idx in range(line_num):
        char_num = chars_per_line[line_idx]
        data += struct.pack("<I", char_num)

        for _ in range(char_num):
            tag = tags[tag_index]
            tag_index += 1
            data += struct.pack(">H", tag)

        top = line_idx * 60 + 10
        left = 10
        height = 48
        width = 320
        data += struct.pack("<IIII", top, left, height, width)

        bitmap = bytes(random.randint(1, 255) for _ in range(width * height))
        data += bitmap

    return data
