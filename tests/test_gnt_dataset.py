"""
测试 GntDataset 类
"""

import struct
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from py_casia_tool.datasets.gnt import GntDataset


def create_fake_gnt_file(filepath: str, num_samples: int = 3) -> None:
    """创建伪造的 GNT 文件"""
    width, height = 72, 48
    bitmap_size = width * height
    data = b""
    for i in range(num_samples):
        tag = 0xB0A1 + i
        total_size = 2 + 2 + 2 + bitmap_size
        data += struct.pack("<I", total_size)
        data += struct.pack(">H", tag)
        data += struct.pack("<HH", width, height)
        data += bytes(random.randint(1, 255) for _ in range(bitmap_size))
    with open(filepath, "wb") as f:
        f.write(data)


class TestGntDatasetInit:

    def test_init_with_single_file(self):
        """用单个 GNT 文件初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            assert len(dataset) == 3
            assert dataset.num_classes == 3
            assert len(dataset.gnt_files) == 1

    def test_init_with_multiple_files(self):
        """用多个 GNT 文件初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=2)
            create_fake_gnt_file(str(Path(tmpdir) / "1247-f.gnt"), num_samples=3)
            dataset = GntDataset(tmpdir)
            assert len(dataset) == 5
            assert len(dataset.gnt_files) == 2

    def test_no_gnt_files_raises(self):
        """目录中无 .gnt 文件应报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                GntDataset(tmpdir)

    def test_max_samples_limit(self):
        """验证 max_samples 参数限制样本数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=10)
            dataset = GntDataset(tmpdir, max_samples=4)
            assert len(dataset) == 4

    def test_parse_writer_code(self):
        """验证 parse_writer_code 参数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir, parse_writer_code=True)
            assert dataset.samples[0].writer_code == "1246"


class TestGntDatasetVocab:

    def test_vocab_built_correctly(self):
        """验证词汇表构建正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            assert "啊" in dataset.char_to_idx
            assert "阿" in dataset.char_to_idx
            assert "埃" in dataset.char_to_idx
            assert set(dataset.idx_to_char) == {"啊", "阿", "埃"}
            assert len(dataset.char_to_idx) == 3

    def test_vocab_unique(self):
        """验证词汇表无重复字符"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=5)
            create_fake_gnt_file(str(Path(tmpdir) / "1247-f.gnt"), num_samples=3)
            dataset = GntDataset(tmpdir)
            assert len(dataset.char_to_idx) == len(set(dataset.char_to_idx.keys()))

    def test_idx_to_char_consistent(self):
        """验证 idx_to_char 与 char_to_idx 一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            for char, idx in dataset.char_to_idx.items():
                assert dataset.idx_to_char[idx] == char


class TestGntDatasetAccess:

    def test_getitem_returns_image_and_label(self):
        """验证 __getitem__ 返回图像和标签"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            image, label = dataset[0]
            assert isinstance(image, np.ndarray)
            assert image.shape == (48, 72)
            assert isinstance(label, int)
            assert 0 <= label < dataset.num_classes

    def test_getitem_out_of_range(self):
        """索引越界应报 IndexError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            with pytest.raises(IndexError):
                _ = dataset[100]

    def test_getitem_with_transform(self):
        """验证 transform 被应用"""
        def resize_transform(image):
            return np.resize(image, (28, 28))

        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir, transform=resize_transform)
            image, _ = dataset[0]
            assert image.shape == (28, 28)

    def test_getitem_with_target_transform(self):
        """验证 target_transform 被应用"""
        def one_hot_transform(label):
            return np.array([1 if i == label else 0 for i in range(3)])

        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir, target_transform=one_hot_transform)
            _, label = dataset[0]
            assert isinstance(label, np.ndarray)
            assert label.shape == (3,)

    def test_get_char(self):
        """验证 get_char 返回原始 GntChar 对象"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            char = dataset.get_char(0)
            assert char.tag_code == "啊"
            assert char.width == 72
            assert char.height == 48


class TestGntDatasetProperties:

    def test_len(self):
        """验证 __len__ 返回样本数量"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=5)
            dataset = GntDataset(tmpdir)
            assert len(dataset) == 5

    def test_num_classes(self):
        """验证 num_classes 返回类别数量"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=10)
            dataset = GntDataset(tmpdir)
            assert dataset.num_classes == 10

    def test_repr(self):
        """验证 __repr__ 输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            repr_str = repr(dataset)
            assert "GntDataset" in repr_str
            assert "num_samples=3" in repr_str
            assert "num_classes=3" in repr_str
