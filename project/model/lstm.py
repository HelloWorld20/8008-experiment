import torch
import torch.nn as nn

class DemandPredictor(nn.Module):
    """
    [Step 2] 需求预测神经网络 (A同学负责)
    使用 LSTM 或其他时间序列模型预测未来的需求量 y_{it}
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 4, output_size: int = 1, use_category_embedding: bool = True, embedding_dim: int = 16):
        """
        初始化 DemandPredictor 模型。
        
        参数:
            input_size (int): 输入特征维度（默认为18，或根据具体特征工程而定）
            hidden_size (int): LSTM 隐藏层维度（默认为 512）
            num_layers (int): LSTM 层数（默认为 4）
            output_size (int): 预测输出维度（默认为 1）
            use_category_embedding (bool): 是否使用类别特征 Embedding（默认为 True）
            embedding_dim (int): 类别特征 Embedding 维度（默认为 16）
        """
        super(DemandPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_category_embedding = use_category_embedding
        
        if self.use_category_embedding:
            # 4种类别: smooth, erratic, intermittent, lumpy
            self.category_emb = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)
            input_size += embedding_dim
            
        # 多层 LSTM 用于提取时间序列特征，加入 Dropout 防止过拟合
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # 对应参考代码中的多层全连接网络及 BatchNorm, Dropout
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, category_idx: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播计算预测需求量
        
        参数:
            x (torch.Tensor): 输入特征张量, 形状 (batch_size, seq_len, input_size) 
                              或 (batch_size, input_size)
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

        # 如果输入没有序列维度，则扩展一维: (batch_size, input_size) -> (batch_size, 1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # 传入 LSTM 进行序列处理
        # LSTM 返回: out (batch_size, seq_len, hidden_size), (hn, cn)
        # 其中 hn 形状为 (num_layers, batch_size, hidden_size)
        _, (hn, _) = self.lstm(x)
        
        # 取最后一层 LSTM（即第 num_layers 层）的隐状态作为整个序列的特征表示
        # hn[-1] 形状: (batch_size, hidden_size)
        final_state = hn[-1]
        
        # 第一个全连接模块：Linear -> BatchNorm -> Dropout -> ReLU
        out = self.fc1(final_state)
        # 注意: 如果 batch_size = 1，BatchNorm1d 可能会报错。如果是训练模式且 batch_size=1，则跳过 BatchNorm 或通过 eval() 规避。
        # 这里为了稳健性，加一个条件判断
        if out.shape[0] > 1 or not self.training:
            out = self.bn1(out)
        out = self.dp1(out)
        out = self.relu(out)
        
        # 第二个全连接模块：Linear -> BatchNorm -> Dropout -> ReLU
        out = self.fc2(out)
        if out.shape[0] > 1 or not self.training:
            out = self.bn2(out)
        out = self.dp2(out)
        out = self.relu(out)
        
        # 输出层映射到最终需求量
        out = self.fc3(out)
        
        # 需求量应为非负数
        y_pred = self.relu(out)
        
        return y_pred
