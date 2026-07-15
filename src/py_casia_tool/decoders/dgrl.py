from io import BytesIO
import os

import numpy as np

from ..types import DgrlLine
from ..utils.decorators import register_decoder
from ..utils.read_exactly import read_exactly

encoding = {1: "ascii", 2: "gbk", 4: "utf-32-le"}

@register_decoder(suffix=".dgrl", format_name="dgrl")
def decode_dgrl(path_or_bytes, parse_writer_code: bool = False) -> list[DgrlLine]:
    bytes_data = None
    writer_code = None
    if isinstance(path_or_bytes, str):
        if os.path.exists(path_or_bytes):
            with open(path_or_bytes, "rb") as f:
                bytes_data = BytesIO(f.read())
            if parse_writer_code:
                writer_code = os.path.basename(path_or_bytes).split("-")[0]
        else:
            raise FileNotFoundError(f"文件不存在: {path_or_bytes}")
    else:
        bytes_data = BytesIO(path_or_bytes)

    if bytes_data is None:
        raise ValueError("bytes_data is None")

    header_size = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]

    format_code = read_exactly(bytes_data, 8).decode("ascii").rstrip("\x00")
    if not format_code.startswith("DGRL"):
        raise ValueError(f"无效的 DGRL 格式标识: {format_code}")

    illustration_size = header_size - 36
    _illustration = read_exactly(bytes_data, illustration_size).decode("ascii").strip("\x00")

    _code_type = read_exactly(bytes_data, 20).decode("ascii").strip("\x00")
    code_length = np.frombuffer(read_exactly(bytes_data, 2), dtype=np.uint16)[0]
    bits_per_pixel = np.frombuffer(read_exactly(bytes_data, 2), dtype=np.uint16)[0]

    image_height = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]
    image_width = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]
    line_num = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]

    dgrl_lines = []

    for line_idx in range(line_num):
        char_number = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0] #该行的字符数量

        # 读取该行所有字符的标注（code_length * char_number 字节）
        tag_bytes = read_exactly(bytes_data, code_length * char_number)
        label = tag_bytes.decode(encoding[code_length], errors="ignore").replace("\x00", "")

        # 读取该行的位置和尺寸（4×4=16 字节）
        top = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]
        left = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]
        height = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]
        width = np.frombuffer(read_exactly(bytes_data, 4), dtype=np.uint32)[0]

        # H*( (W + 7 ) / 8) or H*W
        bitmap_size = height * (width + 7) // 8 if bits_per_pixel == 1 else height * width
        bitmap_bytes = read_exactly(bytes_data, bitmap_size)

        if bits_per_pixel == 1:
            bitmap_flat = np.frombuffer(bitmap_bytes, dtype=np.uint8)
            bitmap = np.unpackbits(bitmap_flat)[:height * width].reshape(height, width)
        else:
            bitmap = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape(height, width)

        dgrl_lines.append(
            DgrlLine(
                line_number=line_idx + 1,
                image_width=image_width,
                image_height=image_height,
                label=label,
                width=width,
                height=height,
                top=top,
                left=left,
                bitmap=bitmap,
                writer_code=writer_code,
            )
        )

    return dgrl_lines
