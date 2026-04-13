import sys
import os
import torch
import numpy as np

# 将 project 目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surrogate.model import SurrogateModel, SurrogateAutogradFunction

def test_surrogate_and_autograd():
    """
    [C 同学单测]
    测试 LightGBM 代理模型的拟合以及自定义 PyTorch Autograd 的伪梯度回传
    """
    print("=== Testing Surrogate Model & Autograd (C) ===")
    
    # 1. 模拟历史数据缓冲池 (预热阶段产生的数据)
    num_samples = 100
    # 假设预测的需求量服从正态分布，均值 10，方差 2
    y_pred_history = np.random.normal(loc=10.0, scale=2.0, size=(num_samples, ))
    
    # 假设真实的代价函数是一个非线性的凹函数：(x-12)^2 + 10 加上一些噪声
    # 模拟：当预测量等于 12 时，总成本最低 (即最优)
    true_cost_history = (y_pred_history - 12.0)**2 + 10.0 + np.random.normal(scale=1.0, size=(num_samples, ))
    
    # 2. 训练 Surrogate Model
    print(f"Training Surrogate Model with {num_samples} historical samples...")
    surrogate = SurrogateModel()
    surrogate.train_surrogate(y_pred_history, true_cost_history)
    print("Surrogate Model trained successfully.")
    
    # 3. 验证 Surrogate 预测能力
    test_y = np.array([10.0, 12.0, 14.0])
    pred_costs = surrogate.predict_cost(test_y)
    print(f"\nSurrogate Predictions:")
    print(f"y_pred=10.0 -> Estimated Cost: {pred_costs[0]:.2f}")
    print(f"y_pred=12.0 -> Estimated Cost: {pred_costs[1]:.2f} (Should be lowest)")
    print(f"y_pred=14.0 -> Estimated Cost: {pred_costs[2]:.2f}")
    
    # 4. 测试 PyTorch Autograd 桥接 (伪梯度回传)
    print("\nTesting PyTorch Autograd Bridge (Finite Difference Gradient)...")
    
    # A 同学传来的预测张量，要求导！
    # 假设当前预测量为 10.0 (此时导数应该为负，因为向右移动到 12.0 成本会下降)
    y_pred_tensor = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)
    print(f"Initial y_pred_tensor: {y_pred_tensor.item()}, requires_grad={y_pred_tensor.requires_grad}")
    
    # 前向传播 (经过 SurrogateAutogradFunction)
    cost_tensor = SurrogateAutogradFunction.apply(y_pred_tensor, surrogate)
    print(f"Forward pass Surrogate cost tensor: {cost_tensor.item():.2f}")
    
    # 反向传播
    cost_tensor.backward()
    
    # 检查梯度是否成功回传给 y_pred_tensor
    print(f"Backward pass successful!")
    print(f"Gradient w.r.t y_pred_tensor: {y_pred_tensor.grad.item():.4f}")
    
    # 验证梯度方向
    # 因为我们在 10.0 处，最优解在 12.0 处，所以导数应该为负数（沿着负梯度方向更新将增大 y，逼近 12）
    if y_pred_tensor.grad.item() < 0:
        print("Gradient direction is CORRECT (Negative, pushing y_pred upwards to 12.0).")
    else:
        print("Warning: Gradient direction might be incorrect based on dummy data.")

if __name__ == "__main__":
    test_surrogate_and_autograd()
