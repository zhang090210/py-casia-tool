def read_exactly(bytes_data, num_bytes: int) -> bytes:
    """读取指定字节数"""
    read_bytes = bytes_data.read(num_bytes)
    if read_bytes is None or len(read_bytes) < num_bytes:
        raise ValueError(f"Incomplete file content, expected {num_bytes} bytes")
    return read_bytes
