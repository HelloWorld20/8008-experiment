import torch
import numpy as np
import lightgbm as lgb
from typing import Tuple

class SurrogateModel:
    """
    [Step 5] 代理模型 (C同学负责)
    使用 LightGBM 或小型神经网络，拟合 (y_pred) -> true_cost 的映射，从而提供平滑的梯度
    """
    def __init__(self):
        # 初始化 LightGBM 回归器作为代理模型
        self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
        self.is_trained = False
        
    def train_surrogate(self, y_pred_history: np.ndarray, true_cost_history: np.ndarray):
        """
        使用历史的预测量和真实的代价训练代理模型
        
        参数:
            y_pred_history: 过去的预测需求量数据
            true_cost_history: 这些预测量经过 Solver 和 Env 产生的真实代价
        """
        # 拟合 mapping: y_pred -> true_cost
        # 注意：如果有更多特征 (如 item_id, season) 也可以加进来
        self.model.fit(y_pred_history.reshape(-1, 1), true_cost_history)
        self.is_trained = True
        
    def predict_cost(self, y_pred: np.ndarray) -> np.ndarray:
        """预测成本 (前向推理)"""
        if not self.is_trained:
            raise ValueError("Surrogate model is not trained yet!")
        return self.model.predict(y_pred.reshape(-1, 1))

class SurrogateAutogradFunction(torch.autograd.Function):
    """
    为不可导的 Solver+Env 链路提供代理梯度 (C同学核心工作)
    这是一个自定义的 PyTorch Autograd Function
    """
    @staticmethod
    def forward(ctx, y_pred_tensor, surrogate_model):
        """
        前向传播：直接返回 surrogate 预测的 cost，并保存张量用于反向传播
        """
        y_pred_np = y_pred_tensor.detach().cpu().numpy()
        
        # 使用代理模型预测 cost
        cost_np = surrogate_model.predict_cost(y_pred_np)
        cost_tensor = torch.tensor(cost_np, dtype=torch.float32, device=y_pred_tensor.device)
        
        # 保存上下文变量用于反向传播
        ctx.save_for_backward(y_pred_tensor)
        ctx.surrogate_model = surrogate_model
        
        return cost_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：利用代理模型估算梯度 d(Cost)/d(y_pred)
        """
        y_pred_tensor, = ctx.saved_tensors
        surrogate_model = ctx.surrogate_model
        
        # 此处使用有限差分法 (Finite Difference) 来近似计算代理模型的梯度
        epsilon = 1e-4
        y_np = y_pred_tensor.detach().cpu().numpy()
        
        cost_plus = surrogate_model.predict_cost(y_np + epsilon)
        cost_minus = surrogate_model.predict_cost(y_np - epsilon)
        
        # 计算伪梯度 = (f(x+e) - f(x-e)) / 2e
        surrogate_grad = (cost_plus - cost_minus) / (2 * epsilon)
        
        # 将 Numpy Array 转回 PyTorch Tensor
        grad_tensor = torch.tensor(surrogate_grad, dtype=torch.float32, device=y_pred_tensor.device)
        
        # 链式法则：将上游传来的梯度(grad_output) 乘以 当前层的代理梯度
        # 注意: 如果 grad_output 是多维的，需要进行匹配的广播操作
        grad_tensor = grad_tensor.view_as(grad_output)
        
        # 对应 forward 的输入参数，依次返回它们的梯度。surrogate_model 不需要梯度，返回 None。
        return grad_output * grad_tensor, None
