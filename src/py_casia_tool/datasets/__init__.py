"""
datasets 模块

本模块提供 CASIA 手写汉字数据集的封装类和 DataLoader 工厂函数，
支持 GNT（离线孤立字符）和 DGRL（离线手写文本行）两种格式。

主要组件：
- GntDataset: GNT 格式数据集类
- DgrlDataset: DGRL 格式数据集类
- create_gnt_dataloader: 创建 GNT 数据集的 DataLoader
- create_dgrl_dataloader: 创建 DGRL 数据集的 DataLoader

使用示例：
    from py_casia_tool.datasets import GntDataset, create_gnt_dataloader

    # 直接使用 Dataset
    dataset = GntDataset("path/to/gnt/files")
    image, label = dataset[0]

    # 使用 DataLoader
    dataloader = create_gnt_dataloader("path/to/gnt/files", batch_size=32)
"""

from .gnt import GntDataset
from .dgrl import DgrlDataset

__all__ = ["GntDataset", "DgrlDataset"]
