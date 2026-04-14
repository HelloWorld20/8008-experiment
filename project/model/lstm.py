import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandPredictor(nn.Module):
    """
    [Step 2] 需求预测神经网络 (A同学负责)
    使用 LSTM 或其他时间序列模型预测未来的需求量 y_{it}
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, output_size: int = 1, use_category_embedding: bool = True, embedding_dim: int = 4):
        """
        初始化 DemandPredictor 模型。
        
        参数:
            input_size (int): 输入特征维度（默认为18，或根据具体特征工程而定）
            hidden_size (int): LSTM 隐藏层维度（大幅调小，防止过拟合代理梯度）
            num_layers (int): LSTM 层数（默认为 1）
            output_size (int): 预测输出维度（默认为 1）
            use_category_embedding (bool): 是否使用类别特征 Embedding（默认为 True）
            embedding_dim (int): 类别特征 Embedding 维度
        """
        super(DemandPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_category_embedding = use_category_embedding
        
        if self.use_category_embedding:
            # 4种类别: smooth, erratic, intermittent, lumpy
            self.category_emb = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)
            input_size += embedding_dim
            
        # 降维的轻量级 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # 使用 LayerNorm 替代 BatchNorm，减少小 batch 和长尾分布带来的抖动。
        self.fc1 = nn.Linear(hidden_size, 32)
        self.ln1 = nn.LayerNorm(32)
        self.dp1 = nn.Identity()
        
        self.fc2 = nn.Linear(32, 16)
        self.ln2 = nn.LayerNorm(16)
        self.dp2 = nn.Identity()
        
        self.fc3 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, category_idx: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播计算预测需求量
        
        参数:
            x (torch.Tensor): 输入特征张量, 推荐形状为 (batch_size, seq_len, input_size)
                              兼容旧格式 (batch_size, input_size)
            category_idx (torch.Tensor): 类别索引特征, 形状 (batch_size,)
            
        返回:
            y_pred (torch.Tensor): 预测的需求量, 形状 (batch_size, output_size)
        """
        if self.use_category_embedding and category_idx is not None:
            # emb: (batch_size, embedding_dim)
            emb = self.category_emb(category_idx)
            
            if len(x.shape) == 2:
                # x 形状为 (batch_size, input_size)，直接在特征维度拼接
                x = torch.cat([x, emb], dim=-1)
            elif len(x.shape) == 3:
                # x 形状为 (batch_size, seq_len, input_size)
                # 扩展 emb 的序列维度: (batch_size, 1, embedding_dim) -> (batch_size, seq_len, embedding_dim)
                emb_expanded = emb.unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x, emb_expanded], dim=-1)

        # 兼容旧数据格式，避免历史脚本直接报错。
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # 传入 LSTM 进行序列处理
        # LSTM 返回: out (batch_size, seq_len, hidden_size), (hn, cn)
        # 其中 hn 形状为 (num_layers, batch_size, hidden_size)
        _, (hn, _) = self.lstm(x)
        
        # 取最后一层 LSTM（即第 num_layers 层）的隐状态作为整个序列的特征表示
        # hn[-1] 形状: (batch_size, hidden_size)
        final_state = hn[-1]
        
        # 第一个全连接模块：Linear -> LayerNorm -> ReLU
        out = self.fc1(final_state)
        out = self.ln1(out)
        out = self.dp1(out)
        out = self.relu(out)
        
        # 第二个全连接模块：Linear -> LayerNorm -> ReLU
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.dp2(out)
        out = self.relu(out)
        
        # 输出层映射到最终需求量
        out = self.fc3(out)
        
        # 需求量应为非负数，使用 Softplus 替代 ReLU，避免 Dying ReLU 问题导致梯度消失
        # Softplus(x) = log(1 + exp(x))，平滑且在 x<0 时也有微小梯度
        y_pred = F.softplus(out)
        
        return y_pred
