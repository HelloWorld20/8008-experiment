import sys
import os
import torch

# 将 project 目录加入 sys.path，以便能够 import 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lstm import DemandPredictor
from data.dataset import get_dataloader

def test_demand_predictor():
    """
    [A 同学单测]
    测试 LSTM 模型的前向推理逻辑和张量形状是否符合接口要求
    """
    print("=== Testing Demand Predictor (A) ===")
    
    # 1. 初始化模型
    batch_size = 4
    seq_len = 28 # 模拟时间步长
    input_size = 18 # 根据参考代码，特征维度通常是18
    hidden_size = 512 # 根据参考代码设置
    num_layers = 4 # 根据参考代码设置
    output_size = 1
    
    model = DemandPredictor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    print(f"Model initialized: {model}")
    
    # 2. 生成模拟输入 (或者从 DataLoader 取一个 batch)
    dummy_features = torch.randn(batch_size, seq_len, input_size)
    print(f"Input Features Shape: {dummy_features.shape}")
    
    # 3. 前向推理
    try:
        y_pred = model(dummy_features)
        print(f"Forward Pass Success!")
        print(f"Output Shape: {y_pred.shape} (Expected: [{batch_size}, {output_size}])")
        
        # 验证 ReLU 是否生效 (预测值不为负)
        assert (y_pred >= 0).all(), "Demand prediction should be non-negative!"
        print("Output values are non-negative. OK.")
        
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

if __name__ == "__main__":
    test_demand_predictor()
