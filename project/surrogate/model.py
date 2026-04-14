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
        # 增加 L2 正则化以防止过拟合极其噪杂的成本表面
        self.model = HistGradientBoostingRegressor(
            max_iter=100, 
            learning_rate=0.05,
            max_depth=3,          # 进一步降低深度，强制平滑
            max_leaf_nodes=7,     # 减少叶子节点
            min_samples_leaf=20,  # 增加叶子样本数，抗噪
            l2_regularization=1.0,
            random_state=42
        )
        self.is_trained = False
        
    def train_surrogate(self, y_pred_history: np.ndarray, context_history: np.ndarray, true_cost_history: np.ndarray):
        """
        使用历史的预测量和真实的代价训练代理模型
        
        参数:
            y_pred_history: 过去的预测需求量数据
            context_history: 过去的业务上下文特征 (c_h, c_u, c_f, p_i, v_i)
            true_cost_history: 这些预测量经过 Solver 和 Env 产生的真实代价
        """
        # 拟合 mapping: (y_pred, context) -> true_cost
        # 这样代理模型才能知道在不同成本参数下，同一个 y_pred 为什么会导致不同的 cost
        print("DEBUG: Calling LightGBM fit()...")
        try:
            X = np.column_stack([y_pred_history, context_history])
            self.model.fit(X, true_cost_history)
            print("DEBUG: LightGBM fit() finished.")
        except Exception as e:
            print(f"DEBUG: LightGBM fit() raised exception: {e}")
            import traceback
            traceback.print_exc()
            raise e
        self.is_trained = True
        
    def predict_cost(self, y_pred: np.ndarray, context: np.ndarray) -> np.ndarray:
        """预测成本 (前向推理)"""
        if not self.is_trained:
            raise ValueError("Surrogate model is not trained yet!")
        X = np.column_stack([y_pred, context])
        return self.model.predict(X)

class SurrogateAutogradFunction(torch.autograd.Function):
    """
    为不可导的 Solver+Env 链路提供代理梯度 (C同学核心工作)
    这是一个自定义的 PyTorch Autograd Function
    """
    @staticmethod
    def forward(ctx, y_pred_tensor, context_tensor, surrogate_model):
        """
        前向传播：直接返回 surrogate 预测的 cost，并保存张量用于反向传播
        """
        # 为了兼容多维 Tensor (如 batch_size x seq_len x output_dim)
        original_shape = y_pred_tensor.shape
        y_pred_flat = y_pred_tensor.detach().view(-1).cpu().numpy()
        context_flat = context_tensor.detach().cpu().numpy()
        
        # 使用代理模型预测 cost
        cost_np = surrogate_model.predict_cost(y_pred_flat, context_flat)
        
        # 恢复原始形状并转为 Tensor
        cost_tensor = torch.tensor(cost_np, dtype=torch.float32, device=y_pred_tensor.device).view(original_shape)
        
        # 保存上下文变量用于反向传播
        ctx.save_for_backward(y_pred_tensor, context_tensor)
        ctx.surrogate_model = surrogate_model
        
        return cost_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：利用代理模型估算梯度 d(Cost)/d(y_pred)
        """
        y_pred_tensor, context_tensor = ctx.saved_tensors
        surrogate_model = ctx.surrogate_model
        
        # 展平输入以供 LightGBM 计算
        original_shape = y_pred_tensor.shape
        y_np = y_pred_tensor.detach().view(-1).cpu().numpy()
        context_np = context_tensor.detach().cpu().numpy()
        
        # 此处使用中心有限差分法 (Central Finite Difference) 来近似计算代理模型的梯度
        # 增大 epsilon，跨越树模型的阶梯状表面，获取宏观趋势
        epsilon = 0.5 
        
        cost_plus = surrogate_model.predict_cost(y_np + epsilon, context_np)
        cost_minus = surrogate_model.predict_cost(y_np - epsilon, context_np)
        
        # 计算伪梯度 = (f(x+e) - f(x-e)) / (2e)
        surrogate_grad = (cost_plus - cost_minus) / (2 * epsilon)
        
        # 为了防止梯度爆炸，对伪梯度进行裁剪 (Gradient Clipping)
        surrogate_grad = np.clip(surrogate_grad, -5.0, 5.0)
        
        # 将 Numpy Array 转回 PyTorch Tensor，并恢复形状
        grad_tensor = torch.tensor(surrogate_grad, dtype=torch.float32, device=y_pred_tensor.device).view(original_shape)
        
        # 链式法则：将上游传来的梯度(grad_output) 乘以 当前层的代理梯度
        final_grad = grad_output * grad_tensor
        
        # 对应 forward 的输入参数依次返回梯度。
        # y_pred_tensor 需要梯度 (返回 final_grad)
        # context_tensor 不需要梯度 (返回 None)
        # surrogate_model 不是 Tensor，不需要梯度 (返回 None)
        return final_grad, None, None
