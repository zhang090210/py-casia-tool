"""
DGRL 离线手写文本行数据集封装

DGRL (Offline Handwritten Text Line) 格式是 CASIA 手写汉字数据库的文本行格式，
每个 .dgrl 文件包含一行或多行手写文本，每行由多个字符组成。

本模块提供 DgrlDataset 类，用于加载和管理 DGRL 格式数据，
支持词汇表构建、数据变换、样本限制等功能，适用于深度学习训练。
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from unstable import unstable

from ..decoders.dgrl import decode_dgrl
from ..types import DgrlLine


@unstable
class DgrlDataset:
    """
    DGRL 数据集类，封装 .dgrl 文件的加载和访问接口

    支持从目录中自动扫描所有 .dgrl 文件，加载文本行数据并构建词汇表，
    提供类似 PyTorch Dataset 的 __getitem__ 和 __len__ 接口。

    注意：DGRL 格式中每个样本是一行文本（包含多个字符），
    label 为整行文本字符串，词汇表从所有行的单个字符中构建。

    Attributes:
        root_dir: 数据集根目录路径
        dgrl_files: 扫描到的所有 .dgrl 文件列表（已排序）
        samples: 加载的所有字符样本列表（每个样本是 DgrlLine 对象）
        char_to_idx: 字符到索引的映射字典
        idx_to_char: 索引到字符的映射列表
        num_classes: 类别数量（词汇表大小）
        transform: 图像变换函数
        target_transform: 标签变换函数
        parse_writer_code: 是否解析书写者代码
        max_samples: 最大加载样本数（None 表示不限制）
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        parse_writer_code: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        初始化 DgrlDataset

        Args:
            root_dir: 包含 .dgrl 文件的目录路径
            transform: 应用于图像的变换函数，输入 np.ndarray，输出任意类型
            target_transform: 应用于标签的变换函数，输入 int，输出任意类型
            parse_writer_code: 是否从文件名中解析书写者代码（如 "741-P14.dgrl" → "741"）
            max_samples: 限制加载的最大样本数，用于快速验证或调试
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.parse_writer_code = parse_writer_code
        self.max_samples = max_samples

        # 扫描并排序目录中的所有 .dgrl 文件
        self.dgrl_files = sorted(self.root_dir.glob("*.dgrl"))
        if not self.dgrl_files:
            raise FileNotFoundError(f"在 {root_dir} 中未找到 .dgrl 文件")

        # 加载所有样本
        self.samples: List[DgrlLine] = []
        self._load_all()

        # 构建词汇表（字符到索引的映射）
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: List[str] = []
        self._build_vocab()

    def _load_all(self) -> None:
        """
        加载目录中所有 .dgrl 文件的文本行数据

        遍历 dgrl_files 列表，逐个解码文件并收集样本。
        每个 .dgrl 文件包含多个文本行，每行包含多个字符。
        所有字符按顺序展平存储在 samples 列表中。
        若设置了 max_samples，达到限制后提前停止。
        """
        for dgrl_file in self.dgrl_files:
            lines = decode_dgrl(str(dgrl_file), parse_writer_code=self.parse_writer_code)
            self.samples.extend(lines)
            # 检查样本数量限制
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                self.samples = self.samples[: self.max_samples]
                break

    def _build_vocab(self) -> None:
        """
        从所有行的标签中构建字符级词汇表

        将每行 label 拆分为单个字符，提取所有唯一字符并按 Unicode 排序，
        建立字符到索引的双向映射。
        """
        chars = sorted(set(char for s in self.samples for char in s.label))
        for idx, char in enumerate(chars):
            self.char_to_idx[char] = idx
        self.idx_to_char = chars

    def __len__(self) -> int:
        """返回数据集样本总数（行数）"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        获取指定索引的样本（一行文本）

        Args:
            idx: 样本索引

        Returns:
            (image, label): 行图像数组和标签索引
                - image: 形状为 (height, width) 的 np.ndarray
                - label: 行首字符在词汇表中的索引（int）
        """
        sample = self.samples[idx]
        image = sample.bitmap
        first_char = sample.label[0] if sample.label else ""
        label = self.char_to_idx.get(first_char, 0)

        # 应用图像变换
        if self.transform:
            image = self.transform(image)
        # 应用标签变换
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_line(self, idx: int) -> DgrlLine:
        """
        获取指定索引的原始 DgrlLine 对象

        与 __getitem__ 不同，此方法返回完整的 DgrlLine 结构，
        包含原始 bitmap、label、width、height、top、left、
        line_number、image_width、image_height 等所有信息。

        Args:
            idx: 样本索引

        Returns:
            DgrlLine: 原始文本行数据对象
        """
        return self.samples[idx]

    @property
    def num_classes(self) -> int:
        """返回词汇表大小（类别数量）"""
        return len(self.char_to_idx)

    def __repr__(self) -> str:
        """返回数据集的字符串表示"""
        return f"DgrlDataset(root_dir={self.root_dir}, " f"num_samples={len(self)}, num_classes={self.num_classes})"
