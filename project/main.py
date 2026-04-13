import torch
from data.dataset import get_dataloader
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from env.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel
from train.loop import train_predict_and_optimize

def main():
    """
    Predict-and-Optimize 框架全局入口文件
    负责初始化所有模块并启动端到端训练流水线。
    """
    print("Initializing Project Components...")
    
    # 1. 初始化 DataLoader
    # 批量大小设置为 32
    dataloader = get_dataloader(batch_size=32)
    
    # 2. 初始化预测模型 (A同学负责)
    # 假设输入特征为 28 维 (对应最近 28 天的销量等历史特征)，隐藏层 64，输出 1 维 (未来的需求量)
    predictor = DemandPredictor(input_size=28, hidden_size=64, output_size=1)
    
    # 3. 初始化求解器与环境 (B同学负责)
    solver = ABCASolver(max_iter=50, pop_size=30)
    env = InventoryEnvironment()
    
    # 4. 初始化代理模型 (C同学负责)
    surrogate = SurrogateModel()
    
    # 5. 启动端到端训练循环 (C同学统筹)
    print("Starting End-to-End Predict-and-Optimize Training Loop...")
    train_predict_and_optimize(
        dataloader=dataloader,
        predictor=predictor,
        solver=solver,
        env=env,
        surrogate=surrogate,
        epochs=10
    )

if __name__ == "__main__":
    main()
