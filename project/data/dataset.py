import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces import SKUCostParams
from data.category import compute_adi, compute_cv2, classify_type

class M5InventoryDataset(Dataset):
    """
    M5 沃尔玛库存数据集加载器
    负责读取特征并返回: (features, true_demand, cost_params)
    """
    def __init__(self, data_path: str, mode: str = 'train', seq_len: int = 28):
        """
        Args:
            data_path: dataset 文件夹路径 (例如: '../dataset')
            mode: 'train' 或 'test'
            seq_len: 历史窗口长度 (默认使用过去 28 天的销量作为特征)
        """
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        
        # 1. 加载核心销售数据
        sales_path = os.path.join(data_path, 'sales_train_evaluation.csv')
        price_path = os.path.join(data_path, 'sell_prices.csv')
        
        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"未找到销售数据文件: {sales_path}")
            
        print("Loading M5 dataset... This might take a while.")
        self.sales_df = pd.read_csv(sales_path)
        self.prices_df = pd.read_csv(price_path)
        
        # 获取所有表示天数的列名 (d_1, d_2, ... d_1941)
        self.d_cols = [col for col in self.sales_df.columns if col.startswith('d_')]
        
        # 定义特征和目标的时间窗口
        # M5 最终评估是在 d_1914 到 d_1941 (最后 28 天)
        # 简单起见，我们将 d_1914 作为要预测的目标天，它的特征是前面 seq_len 天
        # 如果是训练集，我们可以选择更多的天数来构造样本，这里为了演示，简化为每行 SKU 只取一个固定窗口样本
        if mode == 'train':
            self.target_day_idx = 1913 - 28 # d_1886 作为目标，前面 28 天作特征
        else:
            self.target_day_idx = 1913 # d_1914 作为目标，前面 28 天作特征 (即 d_1886 到 d_1913)
            
        # 计算该 SKU 在目标周的平均价格，用于模拟业务参数
        # (真实情况下，应该按照 item_id, store_id, wm_yr_wk 去 sell_prices 查，这里做简化映射)
        print("Pre-calculating average prices per SKU-Store...")
        self.avg_prices = self.prices_df.groupby(['item_id', 'store_id'])['sell_price'].mean().to_dict()

        print("Classifying data types based on ADI and CV2...")
        # Calculate ADI and CV2 for each SKU based on historical sales
        def calculate_category(row_sales):
            adi = compute_adi(row_sales)
            cv2 = compute_cv2(row_sales)
            return classify_type(adi, cv2)
            
        # We only use historical sales columns for this calculation
        sales_only = self.sales_df[self.d_cols].values
        # Using numpy array calculation for speed instead of pandas apply
        categories = []
        for i in range(sales_only.shape[0]):
            categories.append(calculate_category(sales_only[i]))

        # 使用 copy() 来避免 PerformanceWarning (DataFrame is highly fragmented)
        self.sales_df = self.sales_df.copy()
        self.sales_df['category_idx'] = categories

        print(f"Dataset initialized with {len(self.sales_df)} SKUs.")

    def __len__(self) -> int:
        return len(self.sales_df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SKUCostParams]:
        """
        获取单条(单个 SKU 或一个 Batch SKU)的数据
        返回: (features, category_idx, true_demand, cost_params)
        """
        row = self.sales_df.iloc[idx]
        
        item_id = row['item_id']
        store_id = row['store_id']
        cat_id = row['cat_id']
        category_idx = torch.tensor(row['category_idx'], dtype=torch.long)
        
        # 1. 提取输入特征 (过去 seq_len 天的销量)
        feature_start_idx = self.target_day_idx - self.seq_len
        feature_end_idx = self.target_day_idx
        feature_cols = [f'd_{i}' for i in range(feature_start_idx + 1, feature_end_idx + 1)]
        
        # 转为 numpy 数组并转换为 tensor
        features_np = row[feature_cols].values.astype(np.float32)
        features = torch.tensor(features_np, dtype=torch.float32)
        
        # 2. 提取真实需求 (目标天的销量)
        target_col = f'd_{self.target_day_idx + 1}'
        true_demand_np = np.float32(row[target_col])
        true_demand = torch.tensor([true_demand_np], dtype=torch.float32)
        
        # 3. 构造业务成本参数 (模拟计算逻辑)
        # 获取该 SKU 在该门店的历史平均价格
        p_i = self.avg_prices.get((item_id, store_id), 5.0)
        
        # 假设持仓成本是采购价的 20% / 52周
        c_h = p_i * 0.20 / 52.0
        
        # 假设售价是采购价的 1.5 倍，缺货成本即为损失的毛利
        sell_price = p_i * 1.5
        c_u = sell_price - p_i
        
        # 假设固定订货成本为 10.0
        c_f = 10.0
        
        # 假设体积 v_i 根据类别简单映射
        v_i = 1.0
        if 'HOBBIES' in cat_id:
            v_i = 0.5
        elif 'HOUSEHOLD' in cat_id:
            v_i = 2.0
        elif 'FOODS' in cat_id:
            v_i = 0.8
            
        cost_params = SKUCostParams(
            item_id=item_id,
            store_id=store_id,
            c_h=float(c_h),
            c_u=float(c_u),
            c_f=float(c_f),
            v_i=float(v_i),
            p_i=float(p_i)
        )
        
        return features, category_idx, true_demand, cost_params

def get_dataloader(data_path: str, batch_size: int = 32, mode: str = 'train') -> DataLoader:
    """
    获取 DataLoader
    为了避免在单测中加载整个真实数据集太慢，如果在 data_path 找不到真实文件，
    我们将回退到一个简单的 DummyDataset。
    """
    sales_path = os.path.join(data_path, 'sales_train_evaluation.csv')
    # 自定义 collate_fn 因为 DataLoader 默认无法直接把 List[SKUCostParams] 给 stack 起来
    def custom_collate(batch):
        features = torch.stack([item[0] for item in batch])
        category_idx = torch.stack([item[1] for item in batch])
        demands = torch.stack([item[2] for item in batch])
        # 将多个 dataclass 直接放在列表中
        cost_params_list = [item[3] for item in batch]
        return features, category_idx, demands, cost_params_list

    if not os.path.exists(sales_path):
        print(f"Warning: Real dataset not found at {sales_path}. Using Dummy Dataset for testing.")
        class DummyDataset(Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                features = torch.randn(28)
                category_idx = torch.randint(0, 4, (1,)).squeeze()
                true_demand = torch.tensor([5.0])
                cost_params = SKUCostParams(
                    item_id=f"dummy_{idx}", store_id="dummy_store",
                    c_h=0.5, c_u=2.0, c_f=10.0, v_i=1.0, p_i=5.0
                )
                return features, category_idx, true_demand, cost_params
        return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    dataset = M5InventoryDataset(data_path, mode=mode)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
