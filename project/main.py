import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from data.dataset import get_dataloader
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from environment.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel
from train.loop import train_predict_and_optimize

def main():
    """
    Predict-and-Optimize 框架全局入口文件
    负责初始化所有模块并启动端到端训练流水线。
    """
    parser = argparse.ArgumentParser(description="End-to-End Predict-and-Optimize Training")
    parser.add_argument('--epochs', type=int, default=10, help='epochs数量 (default: 10)')
    parser.add_argument('--report_to', type=str, default='none', help='是否上报指标到特定平台，如 "wandb"')
    parser.add_argument('--exp_name', type=str, default='8008', help='实验/项目名称，当 report_to=wandb 时作为 wandb project name')
    args = parser.parse_args()

    print("Initializing Project Components...")
    
    # 检测运行设备: 优先使用 CUDA (NVIDIA GPU), 其次 MPS (Apple Silicon), 默认 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")
    
    # 1. 初始化 DataLoader
    # 批量大小设置为 32
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(base_dir), 'dataset')
    dataloader = get_dataloader(data_path=dataset_path, batch_size=32)
    print(f"DEBUG: DataLoader length: {len(dataloader)}")
    
    # 2. 初始化预测模型 (A同学负责)
    # 假设输入特征为 28 维 (对应最近 28 天的销量等历史特征)，隐藏层 64，输出 1 维 (未来的需求量)
    predictor = DemandPredictor(input_size=28, hidden_size=64, output_size=1).to(device)
    
    # 3. 初始化求解器与环境 (B同学负责)
    solver = ABCASolver(max_iter=50, pop_size=30)
    env = InventoryEnvironment()
    
    # 4. 初始化代理模型 (C同学负责)
    surrogate = SurrogateModel()
    
    print(f"Starting End-to-End Predict-and-Optimize Training Loop for {args.epochs} epochs...")
    train_predict_and_optimize(
        dataloader=dataloader, 
        predictor=predictor, 
        solver=solver, 
        env=env, 
        surrogate=surrogate, 
        epochs=args.epochs,
        device=device,
        report_to=args.report_to,
        exp_name=args.exp_name
    )

if __name__ == "__main__":
    main()
