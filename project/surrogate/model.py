import torch
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings

# 忽略 sklearn 特征名称警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class SurrogateModel:
    """
    [Step 5] 代理模型 (C同学负责)
    使用 HistGradientBoostingRegressor (代替 LightGBM 解决 macOS 上的 segfault 问题)
    拟合 (y_pred) -> true_cost 的映射，从而提供平滑的梯度
    """
    def __init__(self):
        # 初始化 回归器作为代理模型
        # 调参：为了使预测曲面更加平滑（有利于计算有限差分梯度），限制树深和叶子节点数
        self.model = HistGradientBoostingRegressor(
            max_iter=100, 
            learning_rate=0.05,
            max_depth=5,
            max_leaf_nodes=15,
            min_samples_leaf=5,
            random_state=42
        )
        self.is_trained = False
        
    def train_surrogate(self, y_pred_history: np.ndarray, true_cost_history: np.ndarray):
        """
        使用历史的预测量和真实的代价训练代理模型
        
        参数:
            y_pred_history: 过去的预测需求量数据
            true_cost_history: 这些预测量经过 Solver 和 Env 产生的真实代价
        """
        # 拟合 mapping: y_pred -> true_cost
        # 注意：这里 y_pred 是一维特征。在实际复杂业务中，可能会把 item_id 的 embedding、季节特征等也拼进来
        print("DEBUG: Calling LightGBM fit()...")
        try:
            self.model.fit(y_pred_history.reshape(-1, 1), true_cost_history)
            print("DEBUG: LightGBM fit() finished.")
        except Exception as e:
            print(f"DEBUG: LightGBM fit() raised exception: {e}")
            import traceback
            traceback.print_exc()
            raise e
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
        # 为了兼容多维 Tensor (如 batch_size x seq_len x output_dim)
        original_shape = y_pred_tensor.shape
        y_pred_flat = y_pred_tensor.detach().view(-1).cpu().numpy()
        
        # 使用代理模型预测 cost
        cost_np = surrogate_model.predict_cost(y_pred_flat)
        
        # 恢复原始形状并转为 Tensor
        cost_tensor = torch.tensor(cost_np, dtype=torch.float32, device=y_pred_tensor.device).view(original_shape)
        
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
        
        # 展平输入以供 LightGBM 计算
        original_shape = y_pred_tensor.shape
        y_np = y_pred_tensor.detach().view(-1).cpu().numpy()
        
        # 此处使用中心有限差分法 (Central Finite Difference) 来近似计算代理模型的梯度
        # epsilon 不能太小，因为树模型预测曲面是阶梯状的；稍微大一点能跨越台阶获得宏观梯度
        epsilon = 0.1 
        
        cost_plus = surrogate_model.predict_cost(y_np + epsilon)
        cost_minus = surrogate_model.predict_cost(y_np - epsilon)
        
        # 计算伪梯度 = (f(x+e) - f(x-e)) / (2e)
        surrogate_grad = (cost_plus - cost_minus) / (2 * epsilon)
        
        # 将 Numpy Array 转回 PyTorch Tensor，并恢复形状
        grad_tensor = torch.tensor(surrogate_grad, dtype=torch.float32, device=y_pred_tensor.device).view(original_shape)
        
        # 链式法则：将上游传来的梯度(grad_output) 乘以 当前层的代理梯度
        final_grad = grad_output * grad_tensor
        
        # 对应 forward 的输入参数依次返回梯度。
        # y_pred_tensor 需要梯度 (返回 final_grad)
        # surrogate_model 不是 Tensor，不需要梯度 (返回 None)
        return final_grad, None
