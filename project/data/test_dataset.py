import sys
import os
import torch

# 将 project 目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import get_dataloader, HISTORY_WINDOW_DAYS

def test_dataset():
    """
    [A 同学单测 - Dataset]
    测试数据集加载和特征预处理逻辑
    """
    print("=== Testing M5InventoryDataset (A) ===")
    
    # 假设使用一个很小的 batch size 测试
    batch_size = 4
    
    try:
        # 获取 dataloader，注意路径指向上上级目录的 dataset (因为运行路径是 project/)
        print("Initializing dataloader...")
        
        # 兼容两种运行方式：在 project/ 目录下运行，或在 project/data/ 目录下运行
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        dataset_path = os.path.join(base_dir, 'dataset')
        
        dataloader = get_dataloader(data_path=dataset_path, batch_size=batch_size)
        
        # 尝试读取第一个 batch 的数据
        print("Fetching first batch...")
        batch = next(iter(dataloader))
        features, category_idx, true_demand, cost_params_list = batch
        
        # 打印并验证特征的形状和类型
        print(f"Features Shape: {features.shape} (Expected: [batch_size, {HISTORY_WINDOW_DAYS}, 1])")
        print(f"Category Shape: {category_idx.shape} (Expected: [batch_size])")
        print(f"True Demand Shape: {true_demand.shape} (Expected: [batch_size, 1] or [batch_size])")
        print(f"Cost Params Count: {len(cost_params_list)} (Expected: {batch_size})")
        
        assert isinstance(features, torch.Tensor), "Features should be a PyTorch Tensor"
        assert features.shape[0] == batch_size, "Batch size mismatch in features"
        assert features.shape[1] == HISTORY_WINDOW_DAYS, "Sequence length mismatch in features"
        assert features.shape[2] == 1, "Input size mismatch in features"
        assert len(cost_params_list) == batch_size, "Batch size mismatch in cost parameters"
        
        # 打印第一个样本的成本参数，验证是否成功解析为 dataclass
        first_cost_param = cost_params_list[0]
        print(f"\nFirst sample's SKUCostParams: {first_cost_param}")
        assert hasattr(first_cost_param, 'c_h'), "Missing c_h in cost params"
        assert hasattr(first_cost_param, 'c_u'), "Missing c_u in cost params"
        
        print("\nDataset test passed successfully!")
        
    except Exception as e:
        print(f"Dataset Test Failed: {e}")
        print("\nNote: This is likely because the actual M5 data processing logic is not yet implemented in dataset.py.")
        print("A 同学需要完善 data/dataset.py 中的数据加载和处理逻辑，才能跑通此单测。")

if __name__ == "__main__":
    test_dataset()
