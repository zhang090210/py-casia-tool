"""
测试 register_decoder 装饰器和 decode 统一入口
"""

import pytest

from py_casia_tool.utils.decorators import register_decoder, decode, _decoders_by_suffix, _decoders_by_name
from py_casia_tool.types import GntChar, DgrlLine
from fake_data_generator import generate_gnt_bytes, generate_dgrl_bytes


class TestRegisterDecoder:

    def test_register_new_decoder(self):
        """注册自定义解码器后应出现在注册表中"""

        @register_decoder(suffix=".test", format_name="test")
        def fake_decoder(data, parse_writer_code=False):
            return ["decoded"]

        assert ".test" in _decoders_by_suffix
        assert _decoders_by_suffix[".test"] == "test"
        assert "test" in _decoders_by_name
        assert _decoders_by_name["test"] is fake_decoder

    def test_register_defaults_format_name(self):
        """不传 format_name 时应从 suffix 推导"""

        @register_decoder(suffix=".foo")
        def fake_decoder(data, parse_writer_code=False):
            return []

        assert _decoders_by_suffix[".foo"] == "foo"
        assert "foo" in _decoders_by_name

    def test_gnt_decoder_registered(self):
        """gnt 解码器应已注册"""
        assert ".gnt" in _decoders_by_suffix
        assert _decoders_by_suffix[".gnt"] == "gnt"
        assert "gnt" in _decoders_by_name

    def test_dgrl_decoder_registered(self):
        """dgrl 解码器应已注册"""
        assert ".dgrl" in _decoders_by_suffix
        assert _decoders_by_suffix[".dgrl"] == "dgrl"
        assert "dgrl" in _decoders_by_name


class TestDecodeDispatch:

    def test_decode_bytes_with_fmt(self):
        """bytes 输入 + 指定 fmt 应正确分发"""

        @register_decoder(suffix="._t1", format_name="_t1")
        def fake_decoder(data, parse_writer_code=False):
            return [data]

        payload = b"hello"
        result = decode(payload, fmt="_t1")
        assert result == [payload]

    def test_decode_path_by_suffix(self, tmp_path):
        """文件路径输入应按后缀自动分发"""

        @register_decoder(suffix="._t2", format_name="_t2")
        def fake_decoder(path_or_bytes, parse_writer_code=False):
            return [path_or_bytes, parse_writer_code]

        test_file = tmp_path / "sample._t2"
        test_file.write_bytes(b"data")

        result = decode(str(test_file), parse_writer_code=True)
        assert result[0] == str(test_file)
        assert result[1] is True

    def test_decode_pathlib_path(self, tmp_path):
        """Path 对象也应支持"""

        @register_decoder(suffix="._t3", format_name="_t3")
        def fake_decoder(path_or_bytes, parse_writer_code=False):
            return ["ok"]

        test_file = tmp_path / "sample._t3"
        test_file.write_bytes(b"data")

        result = decode(test_file)
        assert result == ["ok"]

    def test_decode_bytes_without_fmt_raises(self):
        """bytes 输入不指定 fmt 应报错"""
        with pytest.raises(ValueError, match="必须通过 fmt 参数指定格式名"):
            decode(b"hello")

    def test_decode_bytes_unknown_fmt_raises(self):
        """bytes 输入指定未知 fmt 应报错"""
        with pytest.raises(ValueError, match="未注册的格式"):
            decode(b"hello", fmt="nonexistent")

    def test_decode_path_unknown_suffix_raises(self, tmp_path):
        """文件路径未知后缀应报错"""
        test_file = tmp_path / "sample.unknown"
        test_file.write_bytes(b"data")
        with pytest.raises(ValueError, match="未注册的后缀"):
            decode(str(test_file))

    def test_decode_unsupported_source_type_raises(self):
        """不支持的 source 类型应报 TypeError"""
        with pytest.raises(TypeError, match="不支持的 source 类型"):
            decode(12345)

    def test_decode_bytes_passes_parse_writer_code_false(self):
        """bytes 输入时 parse_writer_code 应强制为 False"""

        @register_decoder(suffix="._t4", format_name="_t4")
        def fake_decoder(data, parse_writer_code=False):
            return [parse_writer_code]

        result = decode(b"data", fmt="_t4", parse_writer_code=True)
        assert result[0] is False


# ─────────────── Fixtures：伪造合法字节流 ───────────────


@pytest.fixture
def fake_gnt_bytes():
    """生成包含 3 个样本的合法 GNT 字节流"""
    return generate_gnt_bytes(num_samples=3, width=72, height=48)


@pytest.fixture
def fake_dgrl_bytes():
    """生成包含 2 行文本的合法 DGRL 字节流"""
    return generate_dgrl_bytes(image_height=300, image_width=400, line_num=2, chars_per_line=[2, 3])


# ─────────────── 通过 decode 统一入口解码 GNT ───────────────


class TestDecodeGntViaEntry:

    def test_decode_gnt_bytes(self, fake_gnt_bytes):
        """通过 decode(bytes, fmt='gnt') 解码 GNT"""
        samples = decode(fake_gnt_bytes, fmt="gnt")
        assert len(samples) == 3
        assert all(isinstance(s, GntChar) for s in samples)
        assert samples[0].tag_code == "啊"
        assert samples[-1].tag_code == "埃"

    def test_decode_gnt_file(self, fake_gnt_bytes, tmp_path):
        """通过 decode(路径) 自动识别 .gnt 后缀解码"""
        test_file = tmp_path / "1246-f.gnt"
        test_file.write_bytes(fake_gnt_bytes)
        samples = decode(str(test_file))
        assert len(samples) == 3
        assert samples[0].tag_code == "啊"

    def test_decode_gnt_file_with_writer_code(self, fake_gnt_bytes, tmp_path):
        """通过 decode(路径, parse_writer_code=True) 解析 writer_code"""
        test_file = tmp_path / "1246-f.gnt"
        test_file.write_bytes(fake_gnt_bytes)
        samples = decode(str(test_file), parse_writer_code=True)
        assert samples[0].writer_code == "1246"

    def test_decode_gnt_pathlib(self, fake_gnt_bytes, tmp_path):
        """通过 decode(Path) 解码 GNT"""
        test_file = tmp_path / "sample.gnt"
        test_file.write_bytes(fake_gnt_bytes)
        samples = decode(test_file)
        assert len(samples) == 3

    def test_decode_gnt_bytes_ignores_writer_code(self, fake_gnt_bytes):
        """bytes 输入时 writer_code 应为 None"""
        samples = decode(fake_gnt_bytes, fmt="gnt", parse_writer_code=True)
        assert samples[0].writer_code is None


# ─────────────── 通过 decode 统一入口解码 DGRL ───────────────


class TestDecodeDgrlViaEntry:

    def test_decode_dgrl_bytes(self, fake_dgrl_bytes):
        """通过 decode(bytes, fmt='dgrl') 解码 DGRL"""
        samples = decode(fake_dgrl_bytes, fmt="dgrl")
        assert len(samples) == 2
        assert all(isinstance(s, DgrlLine) for s in samples)
        assert samples[0].label == "啊阿"
        assert samples[-1].label == "埃挨哎"

    def test_decode_dgrl_file(self, fake_dgrl_bytes, tmp_path):
        """通过 decode(路径) 自动识别 .dgrl 后缀解码"""
        test_file = tmp_path / "741-P14.dgrl"
        test_file.write_bytes(fake_dgrl_bytes)
        samples = decode(str(test_file))
        assert len(samples) == 2
        assert samples[0].image_width == 400
        assert samples[0].image_height == 300

    def test_decode_dgrl_file_with_writer_code(self, fake_dgrl_bytes, tmp_path):
        """通过 decode(路径, parse_writer_code=True) 解析 writer_code"""
        test_file = tmp_path / "741-P14.dgrl"
        test_file.write_bytes(fake_dgrl_bytes)
        samples = decode(str(test_file), parse_writer_code=True)
        assert samples[0].writer_code == "741"

    def test_decode_dgrl_pathlib(self, fake_dgrl_bytes, tmp_path):
        """通过 decode(Path) 解码 DGRL"""
        test_file = tmp_path / "sample.dgrl"
        test_file.write_bytes(fake_dgrl_bytes)
        samples = decode(test_file)
        assert len(samples) == 2

    def test_decode_dgrl_bitmap_shape(self, fake_dgrl_bytes):
        """验证 bitmap 形状正确"""
        samples = decode(fake_dgrl_bytes, fmt="dgrl")
        for s in samples:
            assert s.bitmap.shape == (s.height, s.width)
