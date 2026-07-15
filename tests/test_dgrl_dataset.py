"""
测试 DgrlDataset 类
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from py_casia_tool.datasets.dgrl import DgrlDataset
from fake_data_generator import generate_dgrl_bytes


def create_fake_dgrl_file(filepath: str) -> None:
    """创建伪造的 DGRL 文件"""
    data = generate_dgrl_bytes(image_height=300, image_width=400, line_num=2, chars_per_line=[2, 3])
    with open(filepath, "wb") as f:
        f.write(data)


class TestDgrlDatasetInit:

    def test_init_with_single_file(self):
        """用单个 DGRL 文件初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert len(dataset) == 2

    def test_init_with_multiple_files(self):
        """用多个 DGRL 文件初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P15.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert len(dataset) == 4

    def test_no_dgrl_files_raises(self):
        """目录中无 .dgrl 文件应报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                DgrlDataset(tmpdir)

    def test_max_samples_limit(self):
        """验证 max_samples 参数限制样本数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir, max_samples=1)
            assert len(dataset) == 1

    def test_parse_writer_code(self):
        """验证 parse_writer_code 参数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir, parse_writer_code=True)
            assert dataset.samples[0].writer_code == "741"


class TestDgrlDatasetVocab:

    def test_vocab_built_correctly(self):
        """验证词汇表构建正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert "啊" in dataset.char_to_idx
            assert "阿" in dataset.char_to_idx
            assert "埃" in dataset.char_to_idx

    def test_vocab_unique(self):
        """验证词汇表去重"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert len(dataset.char_to_idx) == len(set(dataset.char_to_idx.keys()))

    def test_idx_to_char_consistent(self):
        """验证 idx_to_char 与 char_to_idx 一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            for char, idx in dataset.char_to_idx.items():
                assert dataset.idx_to_char[idx] == char


class TestDgrlDatasetAccess:

    def test_getitem_returns_image_and_label(self):
        """验证 __getitem__ 返回图像和标签索引"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            image, label = dataset[0]
            assert isinstance(image, np.ndarray)
            assert image.shape == (48, 320)
            assert isinstance(label, int)
            assert label == dataset.char_to_idx["啊"]

    def test_getitem_out_of_range(self):
        """索引越界应报 IndexError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            with pytest.raises(IndexError):
                _ = dataset[100]

    def test_getitem_with_transform(self):
        """验证 transform 被应用"""

        def normalize_transform(image):
            return image / 255.0

        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir, transform=normalize_transform)
            image, _ = dataset[0]
            assert image.dtype == np.float64
            assert np.max(image) <= 1.0
            assert np.min(image) >= 0.0

    def test_getitem_with_target_transform(self):
        """验证 target_transform 被应用"""

        def label_plus_one(label):
            return label + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir, target_transform=label_plus_one)
            _, label = dataset[0]
            assert isinstance(label, int)
            assert label == dataset.char_to_idx["啊"] + 1

    def test_get_line(self):
        """验证 get_line 返回原始 DgrlLine 对象"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            line = dataset.get_line(0)
            assert line.line_number == 1
            assert line.label == "啊阿"


class TestDgrlDatasetProperties:

    def test_len(self):
        """验证 __len__ 返回样本数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert len(dataset) == 2

    def test_num_classes(self):
        """验证 num_classes 返回类别数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            assert dataset.num_classes == 5

    def test_repr(self):
        """验证 __repr__ 输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_fake_dgrl_file(str(Path(tmpdir) / "741-P14.dgrl"))
            dataset = DgrlDataset(tmpdir)
            repr_str = repr(dataset)
            assert "DgrlDataset" in repr_str
            assert "num_samples=2" in repr_str
            assert "num_classes=5" in repr_str
