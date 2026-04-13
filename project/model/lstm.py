import torch
import torch.nn as nn

class DemandPredictor(nn.Module):
    """
    [Step 2] 需求预测神经网络 (A同学负责)
    使用 LSTM 或其他时间序列模型预测未来的需求量 y_{it}
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DemandPredictor, self).__init__()
        # 使用 LSTM 提取时间序列特征
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 全连接层映射到需求预测值
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算预测需求量
        
        参数:
            x (torch.Tensor): 输入特征张量, 形状 (batch_size, seq_len, input_size) 
                              或 (batch_size, input_size)
            
        返回:
            y_pred (torch.Tensor): 预测的需求量, 形状 (batch_size, output_size)
        """
        # 如果输入没有序列维度，则扩展一维
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        # 使用 ReLU 保证需求量为非负数
        y_pred = torch.relu(out)
        return y_pred
