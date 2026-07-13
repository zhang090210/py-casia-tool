from ..types import GntChar
import os
from io import BytesIO
import numpy as np


def decode_gnt(path_or_bytes, parse_writer_code: bool = False) -> list[GntChar]:
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

    gnt_chars = []
    while bytes_data.read(1) != "":
        bytes_data.seek(-1, 1)
        total_size_bytes = bytes_data.read(4)
        if len(total_size_bytes) < 4:
            break
        _total_size = np.frombuffer(total_size_bytes, dtype=np.uint32)[0]
        tag_code = bytes_data.read(2).decode("gbk").strip("\x00")
        width = np.frombuffer(bytes_data.read(2), dtype=np.uint16)[0]
        height = np.frombuffer(bytes_data.read(2), dtype=np.uint16)[0]

        bitmap = np.frombuffer(bytes_data.read(width * height), dtype=np.uint8).reshape(height, width)

        gnt_chars.append(GntChar(tag_code, width, height, bitmap, writer_code))

    return gnt_chars
