"""
测试 dataset 和 dataloader 封装
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from py_casia_tool.datasets import GntDataset, DgrlDataset
from fake_data_generator import generate_gnt_bytes, generate_dgrl_bytes


def create_fake_gnt_file(filepath: str, num_samples: int = 3) -> None:
    """创建伪造的 GNT 文件"""
    data = generate_gnt_bytes(num_samples=num_samples, width=72, height=48)
    with open(filepath, "wb") as f:
        f.write(data)


def create_fake_dgrl_file(filepath: str) -> None:
    """创建伪造的 DGRL 文件"""
    data = generate_dgrl_bytes(image_height=300, image_width=400, line_num=2, chars_per_line=[2, 3])
    with open(filepath, "wb") as f:
        f.write(data)


class TestGntDataset:

    def test_init_with_fake_data(self):
        """用伪造数据初始化 GntDataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            assert len(dataset) == 3
            assert dataset.num_classes == 3

    def test_vocab_built_correctly(self):
        """验证词汇表构建正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            assert "啊" in dataset.char_to_idx
            assert "阿" in dataset.char_to_idx
            assert "埃" in dataset.char_to_idx
            assert set(dataset.idx_to_char) == {"啊", "阿", "埃"}

    def test_getitem_returns_image_and_label(self):
        """验证 __getitem__ 返回图像和标签"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            image, label = dataset[0]
            assert isinstance(image, np.ndarray)
            assert image.shape == (48, 72)
            assert isinstance(label, int)
            assert label == 0

    def test_max_samples_limit(self):
        """验证 max_samples 参数限制样本数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"), num_samples=5)
            dataset = GntDataset(tmpdir, max_samples=2)
            assert len(dataset) == 2

    def test_no_gnt_files_raises(self):
        """目录中无 .gnt 文件应报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                GntDataset(tmpdir)

    def test_repr(self):
        """验证 __repr__ 输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_gnt_file(str(Path(tmpdir) / "1246-f.gnt"))
            dataset = GntDataset(tmpdir)
            repr_str = repr(dataset)
            assert "GntDataset" in repr_str
            assert "num_samples=3" in repr_str
            assert "num_classes=3" in repr_str


class TestDgrlDataset:

    def test_init_with_fake_data(self):
        """用伪造数据初始化 DgrlDataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert len(dataset) == 2
            assert dataset.num_classes == 5

    def test_vocab_built_correctly(self):
        """验证词汇表构建正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert "啊" in dataset.char_to_idx
            assert "阿" in dataset.char_to_idx
            assert "埃" in dataset.char_to_idx
            assert "挨" in dataset.char_to_idx
            assert "哎" in dataset.char_to_idx

    def test_getitem_returns_image_and_label(self):
        """验证 __getitem__ 返回图像和标签"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            image, label = dataset[0]
            assert isinstance(image, np.ndarray)
            assert image.shape == (48, 320)
            assert isinstance(label, int)

    def test_no_dgrl_files_raises(self):
        """目录中无 .dgrl 文件应报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                DgrlDataset(tmpdir)
