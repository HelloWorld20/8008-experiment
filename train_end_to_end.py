import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os

# 将 CNN+LSTM 目录添加到 sys.path 以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
cnn_lstm_dir = os.path.join(current_dir, 'CNN+LSTM')
if cnn_lstm_dir not in sys.path:
    sys.path.append(cnn_lstm_dir)

from abca import InventoryABC

# -----------------------------------------------------------------------------
# 1. 真实的网络模型：来自 02-learning-pytorch-lstm.py 的 LSTM2
# -----------------------------------------------------------------------------
class LSTM2(nn.Module):
    '''
    多层 LSTM 及多层感知机 (MLP) 组合模型类。
    用于从时间序列特征中提取并预测未来需求。
    '''
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.LSTM2 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        
        self.fc3= nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        device = x.device
        h_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
        
        # 获取最后一层 LSTM 的隐状态
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        
        x0 = self.fc1(final_state)
        if x0.size(0) > 1:  # BatchNorm1d 要求 batch_size > 1
            x0 = self.bn1(x0)
        x0 = self.dp1(x0)
        x0 = self.relu(x0)
        
        x0 = self.fc2(x0)
        if x0.size(0) > 1:
            x0 = self.bn2(x0)
        x0 = self.dp2(x0)
        x0 = self.relu(x0)
        
        out = self.fc3(x0)
        return out

# -----------------------------------------------------------------------------
# 2. 决策求解器 (ABCA Solver)
# -----------------------------------------------------------------------------
class ABCASolver:
    """
    对接真实的 InventoryABC 类
    """
    def __init__(self, budget, capacity):
        self.budget = budget
        self.capacity = capacity
        
    def solve(self, y_pred_array, df_sku_info):
        """
        运行 ABCA 算法寻找最佳订货量 Q_star。
        y_pred_array: 预测的需求数组 [Batch_size]
        df_sku_info: 包含 price, volume, initial_inventory 的 DataFrame
        """
        df = df_sku_info.copy()
        # 限制预测值，防止负数或极端值导致搜索空间错误
        df['predicted_demand'] = np.maximum(0, y_pred_array)
        
        # 重新计算搜索空间 Q_min 和 Q_max
        df['Q_min'] = np.maximum(0, np.floor(df['predicted_demand'] - df['initial_inventory'] - 5).astype(int))
        df['Q_max'] = np.maximum(df['Q_min'] + 1, np.ceil(df['predicted_demand'] - df['initial_inventory'] + 5).astype(int))

        # 实例化真实的 ABCA。训练循环中调小参数以加快单次迭代速度。
        abc = InventoryABC(
            df=df, 
            pop_size=5,      # 较小的种群
            max_iters=10,    # 较少的迭代次数
            limit=3, 
            budget=self.budget, 
            capacity=self.capacity
        )
        # 屏蔽终端的大量打印输出
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            best_solution, best_fit, best_cost = abc.optimize()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        return best_solution

# -----------------------------------------------------------------------------
# 3. 真实的业务环境 (Inventory Environment)
# -----------------------------------------------------------------------------
class InventoryEnvironment:
    """
    接收决策 Q_star 和真实的未来需求 y_true，计算业务的真实损失 (True Cost)。
    """
    def __init__(self, holding_cost_rate, penalty_cost_rate):
        self.holding_cost_rate = holding_cost_rate
        self.penalty_cost_rate = penalty_cost_rate
        
    def calculate_true_cost(self, Q_star, y_true, prices):
        excess = np.maximum(0, Q_star - y_true)
        shortage = np.maximum(0, y_true - Q_star)
        
        holding_cost = np.sum(excess * prices * self.holding_cost_rate)
        penalty_cost = np.sum(shortage * prices * self.penalty_cost_rate)
        
        return holding_cost + penalty_cost

# -----------------------------------------------------------------------------
# 4. 代理模型 (Surrogate Model)
# -----------------------------------------------------------------------------
class SurrogateModel(nn.Module):
    """
    为了解决 ABCA 不可导的问题，用一个可导的网络来拟合映射：y_pred -> predicted_cost。
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 输出一个标量的预测成本
        )
        
    def forward(self, y_pred):
        return self.net(y_pred)

# -----------------------------------------------------------------------------
# 5. 端到端主训练循环 (Predict-then-Optimize Training Loop)
# -----------------------------------------------------------------------------
def train_end_to_end():
    batch_size = 32
    seq_length = 28
    input_dim = 18
    
    # 1. 初始化模块
    # 根据 02-learning-pytorch-lstm.py: num_classes=1, input_size=18, hidden_size=512, num_layers=4
    print("1. 初始化真实 LSTM2 模型...")
    predictor_net = LSTM2(num_classes=1, input_size=input_dim, hidden_size=512, num_layers=4)
    optimizer_predictor = optim.Adam(predictor_net.parameters(), lr=0.001)
    
    print("2. 初始化真实 ABCASolver...")
    abca_solver = ABCASolver(budget=50000000, capacity=2000000)
    
    print("3. 初始化业务环境和代理模型...")
    env = InventoryEnvironment(holding_cost_rate=0.1, penalty_cost_rate=0.5)
    
    surrogate_net = SurrogateModel(input_dim=batch_size) 
    optimizer_surrogate = optim.Adam(surrogate_net.parameters(), lr=0.005)
    mse_loss_fn = nn.MSELoss()

    num_epochs = 5
    print("\n🚀 开始端到端 Predict-then-Optimize 训练...")
    
    # 模拟一组商品的基础属性 (M5 数据集特征)
    df_sku_info = pd.DataFrame({
        'price': np.random.uniform(1.0, 50.0, size=batch_size),
        'volume': np.random.uniform(0.1, 5.0, size=batch_size),
        'initial_inventory': np.random.randint(0, 5, size=batch_size)
    })
    
    for epoch in range(num_epochs):
        # 模拟获取一个 Batch 的时序数据
        # 形状: [32, 28, 18] -> 32个样本，28天时间步，18个特征
        batch_X = torch.randn(batch_size, seq_length, input_dim) 
        # 真实的未来销量
        batch_y_true = np.random.randint(1, 20, size=(batch_size, 1))
        
        # =====================================================================
        # 阶段 A：训练代理模型 (Surrogate Model)
        # =====================================================================
        
        # 1. 用当前的预测网络给出销量预测
        with torch.no_grad():
            predictor_net.eval()
            y_pred_tensor = predictor_net(batch_X)
            y_pred_numpy = y_pred_tensor.numpy().flatten()
            
        # 2. 求解不可导的黑盒决策 Q_star
        Q_star = abca_solver.solve(y_pred_numpy, df_sku_info)
        
        # 3. 在环境中计算真实的业务损失
        true_cost = env.calculate_true_cost(
            Q_star, 
            batch_y_true.flatten(), 
            df_sku_info['price'].values
        )
        true_cost_tensor = torch.tensor([true_cost], dtype=torch.float32)
        
        # 4. 训练代理模型，拟合 y_pred -> true_cost 的映射
        surrogate_net.train()
        optimizer_surrogate.zero_grad()
        surrogate_input = y_pred_tensor.view(1, -1)
        predicted_cost = surrogate_net(surrogate_input)
        
        surrogate_loss = mse_loss_fn(predicted_cost.squeeze(), true_cost_tensor.squeeze())
        surrogate_loss.backward()
        optimizer_surrogate.step()
        
        # =====================================================================
        # 阶段 B：训练需求预测网络 (Predictor NN)
        # =====================================================================
        
        # 1. 再次前向传播 (开启梯度计算)
        predictor_net.train()
        optimizer_predictor.zero_grad()
        y_pred_tensor_train = predictor_net(batch_X)
        
        # 2. 让代理模型评估预测值将产生的业务成本
        surrogate_net.eval()
        surrogate_input_train = y_pred_tensor_train.view(1, -1)
        estimated_task_loss = surrogate_net(surrogate_input_train)
        
        # 3. 反向传播：目标是让预测网络产生能使成本最小化的预测值
        task_loss = estimated_task_loss.mean() 
        task_loss.backward()
        optimizer_predictor.step()
        
        # 打印日志
        print(f"Epoch [{epoch+1:03d}/{num_epochs}] | "
              f"Surrogate Fit Loss: {surrogate_loss.item():>8.2f} | "
              f"Task (True Cost) Loss: {task_loss.item():>8.2f}")

    print("\n✅ 端到端训练结束！两个文件成功串联。")

if __name__ == "__main__":
    train_end_to_end()
