from typing import Callable, Dict, List, Any, Union, Optional
from pathlib import Path

_decoders_by_suffix: Dict[str, str] = {}
_decoders_by_name: Dict[str, Callable] = {}


def register_decoder(suffix: str, format_name: Optional[str] = None):
    if format_name is None:
        format_name = suffix.lstrip(".")

    def decorator(func: Callable):
        _decoders_by_suffix[suffix] = format_name
        _decoders_by_name[format_name] = func
        return func

    return decorator


def decode(source: Union[str, Path, bytes], fmt: Optional[str] = None, parse_writer_code: bool = False) -> List[Any]:
    """
    :param source: 输入数据，可以是文件路径、字节流或字符串
    :param fmt: 当输入字节流时，必须通过 fmt 参数指定格式名
    :param parse_writer_code: 是否解析 writer_code 字符串
    :return: 解码后的数据列表
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix not in _decoders_by_suffix:
            raise ValueError(f"未注册的后缀: {suffix}，可用: {list(_decoders_by_suffix.keys())}")
        format_name = _decoders_by_suffix[suffix]
        decoder = _decoders_by_name[format_name]
        return decoder(str(path), parse_writer_code=parse_writer_code)

    elif isinstance(source, bytes):
        # 输入字节流时不能指定 writer_code
        if fmt is None:
            raise ValueError("当输入字节流时，必须通过 fmt 参数指定格式名")
        if fmt not in _decoders_by_name:
            raise ValueError(f"未注册的格式: {fmt}，可用: {list(_decoders_by_name.keys())}")
        decoder = _decoders_by_name[fmt]
        return decoder(source, parse_writer_code=False)

    else:
        raise TypeError(f"不支持的 source 类型: {type(source)}")
