import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class M5InventoryDataset(Dataset):
    """
    M5 沃尔玛库存数据集加载器
    负责读取特征并返回: (features, true_demand, cost_params)
    """
    def __init__(self, data_path: str, mode: str = 'train'):
        # 伪代码：实际需要读取 data+baseline_model 预处理后的数据
        self.data_path = data_path
        self.mode = mode
        
    def __len__(self) -> int:
        return 1000  # 示例长度
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        获取单条(单个 SKU 或一个 Batch SKU)的数据
        返回:
            features: 预测模型所需的输入特征 (如历史销量, 价格, 节假日)
            true_demand: 真实的未来需求量 D_{it}
            cost_params: 包含该 SKU 的成本参数字典 (c_h, c_u, c_f, volume, price)
        """
        # 返回 dummy data 作为接口定义
        # 特征维度和数值根据 data+baseline_model.ipynb 中最近 28 天的销量来定
        features = torch.randn(28)  # 过去 28 天的特征
        true_demand = torch.tensor(5.0)  # 真实需求
        cost_params = {
            'c_h': 0.5,    # 单位持有成本
            'c_u': 2.0,    # 单位缺货成本 (利润)
            'c_f': 10.0,   # 固定订货成本
            'v_i': 1.0,    # 单位体积
            'p_i': 5.0     # 采购价格
        }
        return features, true_demand, cost_params

def get_dataloader(batch_size: int = 32) -> DataLoader:
    """
    获取 DataLoader
    """
    # 路径指向上一级的 dataset 目录
    dataset = M5InventoryDataset('../dataset/')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
