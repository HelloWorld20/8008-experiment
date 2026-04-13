import sys
import os
import torch

# 将 project 目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import get_dataloader
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from environment.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel
from train.loop import train_predict_and_optimize

def run_end_to_end_test():
    print("=== Starting End-to-End Pipeline Test ===")
    
    # 1. 准备 DataLoader (使用 DummyDataset 确保没有数据依赖也能跑通)
    # 这里我们只取 5 个 batch 进行快速验证
    print("[1/5] Initializing DataLoader...")
    dataloader = get_dataloader(
        data_path="dummy_dir", 
        batch_size=8
    )
    
    # 获取一个 batch 以探测 input_size
    batch = next(iter(dataloader))
    features = batch[0]
    category_idx = batch[1]
    
    # 我们的 features shape 目前可能是 (batch_size, seq_len)
    # 对于 LSTM 来说，我们需要把它转成 (batch_size, seq_len, input_size) 
    # 或者如果序列里的每一步就是一个标量特征，那么 input_size = 1，或者如果它就是个全连接网络，input_size=seq_len
    # 在当前的 LSTM 设计里，我们假设每个 timestep 有多少特征。
    # 这里如果 shape 只有两维，说明传过来的是 28 天的一维特征，那对于 LSTM 的 input_size 其实是 1（每个时间步1个特征）
    # 或者我们在 LSTM 内部使用了 seq_len 当作 input_size 并只看作一步 (取决于 A 同学的网络)
    # 按照 lstm.py 的 forward 逻辑：如果只有两维，他会做 unsqueeze(1) 把 seq_len=1，所以 input_size=28
    if len(features.shape) == 2:
        input_size = features.shape[1] # 因为在 lstm.py 第60行会 unsqueeze(1) 变成 (batch, 1, seq_len)
    else:
        input_size = features.shape[2]
        
    print(f"      Feature input_size detected: {input_size}")
    
    # 2. 初始化四大组件
    print("[2/5] Initializing Components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Using device: {device}")
    
    # A同学: 预测模型
    predictor = DemandPredictor(input_size=input_size, hidden_size=64, num_layers=2).to(device)
    # B同学: 求解器
    solver = ABCASolver(pop_size=20, max_iter=30)
    # B同学: 环境
    env = InventoryEnvironment()
    # C同学: 代理模型
    surrogate = SurrogateModel()
    
    # 3. 开始端到端训练
    print("[3/5] Starting End-to-End Training Loop...")
    print("-" * 50)
    # 使用 2 个 epoch 进行快速测试
    train_predict_and_optimize(
        dataloader=dataloader,
        predictor=predictor,
        solver=solver,
        env=env,
        surrogate=surrogate,
        epochs=2,
        device=device
    )
    print("-" * 50)
    print("[4/5] Training Loop Completed Successfully!")
    
    # 4. 验证最终状态
    print("[5/5] Final Status Check:")
    print(f"      Surrogate Trained: {surrogate.is_trained}")
    # 测试预测模型是否正常工作
    test_out = predictor(features.to(device), category_idx.to(device))
    print(f"      Predictor Output Shape: {test_out.shape}")
    print("=== All End-to-End Tests Passed ===")

if __name__ == "__main__":
    run_end_to_end_test()