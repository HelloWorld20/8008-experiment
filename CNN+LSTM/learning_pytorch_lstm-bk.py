# -*- coding: utf-8 -*-
"""
M5 Forecasting - PyTorch LSTM Implementation
M5 预测 - PyTorch LSTM 实现

该脚本基于 learning-pytorch-lstm-deep-learning-with-m5-data.ipynb 转换而来。
主要功能：
1. 加载 M5 预测数据集 (sell_prices.csv, calendar.csv, sales_train_validation.csv, sample_submission.csv)。
2. 数据预处理：内存优化、日期索引设置、归一化。
3. 定义并训练 LSTM 模型进行时间序列预测。
   - 实验 1: 简单的单层 LSTM。
   - 实验 2: 多层 LSTM。
   - 实验 3: 包含更多特征（滞后特征、滚动均值/标准差、星期几嵌入）的复杂 LSTM。
4. 模型评估与结果可视化。

注意：
- 数据路径默认为 '../dataset/'，请根据实际情况调整。
- 训练轮数 (EPOCHS) 默认设置为 5 以便快速测试，实际训练请改为 500 或更多。
- 图像将保存为 PNG 文件而不是直接显示。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime as dt
from torch.autograd import Variable
import random 
import os
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time 
from itertools import cycle

# 尝试导入进度条库，如果 fastprogress 不存在则使用 tqdm
try:
    from fastprogress import master_bar, progress_bar
except ImportError:
    from tqdm import tqdm
    def progress_bar(iterable):
        return tqdm(iterable)

# 忽略警告
warnings.filterwarnings('ignore')

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# 设置绘图风格
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# =================配置参数=================
# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集路径 (相对于脚本位置)
INPUT_DIR_PATH = os.path.join(SCRIPT_DIR, '../dataset/')
# 训练轮数 (测试时设为 5，实际训练建议 500)
NUM_EPOCHS = 500
# 设备配置 (优先使用 GPU/MPS)
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

SEED = 1345

# =================工具函数=================

def reduce_mem_usage(df, verbose=True):
    """
    减少 DataFrame 的内存占用。
    通过将数值列转换为更小的数据类型 (如 int8, float16) 来节省内存。
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def read_data():
    """读取 CSV 数据文件"""
    print("Reading files...")
    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df

def seed_everything(seed):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def sliding_windows(data, seq_length):
    """
    创建滑动窗口序列。
    输入: data (数组), seq_length (窗口长度)
    输出: x (输入序列), y (目标值)
    """
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

def sliding_windows_mutli_features(data, seq_length):
    """
    创建多特征的滑动窗口序列。
    输入: data (二维数组, 列为特征), seq_length (窗口长度)
    输出: x (输入序列), y (目标值, 第一列为标签)
    """
    x = []
    y = []
    for i in range((data.shape[0])-seq_length-1):
        _x = data[i:(i+seq_length),:] ## 所有列作为特征
        _y = data[i+seq_length,0]      ## 第0列作为标签 (sales)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y).reshape(-1,1)

def init_weights(m):
    """初始化模型权重"""
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# =================模型定义=================

class LSTM(nn.Module):
    """简单的 LSTM 模型 (实验 1)"""
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = self.dropout(out)
        return out

class LSTM2(nn.Module):
    """复杂的 LSTM 模型 (实验 3)"""
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=num_layers, batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
        
        # 取最后一层的最后时间步的输出
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        
        x0 = self.fc1(final_state)
        x0 = self.bn1(x0)
        x0 = self.dp1(x0)
        x0 = self.relu(x0)
        
        x0 = self.fc2(x0)
        x0 = self.bn2(x0)
        x0 = self.dp2(x0)
        x0 = self.relu(x0)
        
        out = self.fc3(x0)
        return out

# =================主程序=================

if __name__ == "__main__":
    seed_everything(SEED)
    
    # 1. 读取数据
    _, calendar_df, sales_train_validation_df, _ = read_data()

    # 2. 数据预处理
    print("Preprocessing data...")
    # 创建日期索引
    date_index = calendar_df['date']
    dates = date_index[0:1913]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

    # 创建每个项目的每日销售额数据框
    sales_train_validation_df['item_store_id'] = sales_train_validation_df.apply(lambda x: x['item_id']+'_'+x['store_id'], axis=1)
    DF_Sales = sales_train_validation_df.loc[:, 'd_1':'d_1913'].T
    DF_Sales.columns = sales_train_validation_df['item_store_id'].values

    # 设置日期为索引
    DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
    DF_Sales.index = pd.to_datetime(DF_Sales.index)
    # print(DF_Sales.head())

    # 选择一个特定的时间序列进行演示 (Index 6780)
    index = 6780
    if index >= DF_Sales.shape[1]:
        index = 0 # Fallback
    y = pd.DataFrame(DF_Sales.iloc[:, index])
    y = pd.DataFrame(y).set_index([dates_list])
    TS_selected = y 
    y.index = pd.to_datetime(y.index)
    
    # 绘制选定的时间序列
    ax = y.plot(figsize=(30, 9), color='red')
    ax.set_facecolor('lightgrey')
    plt.title('Sales Demand Selected Time Series Over Time', fontsize=23)
    plt.savefig('time_series_plot.png')
    print("Saved time_series_plot.png")

    # 3. 实验 1: 简单的 LSTM
    print("\n=== Experiment 1: Simple LSTM ===")
    
    # 数据归一化
    data = np.array(y)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(data.reshape(-1, 1))

    # 绘制归一化前后的分布
    fig, axs = plt.subplots(2)
    fig.suptitle('Data Distribution Before and After Normalization', fontsize=19)
    pd.DataFrame(data).plot(kind='hist', ax=axs[0], alpha=.4, figsize=[12,6], legend=False, title='Before Normalization', color='red') 
    pd.DataFrame(train_data_normalized).plot(kind='hist', ax=axs[1], figsize=[12,6], alpha=.4, legend=False, title='After Normalization', color='blue')
    plt.savefig('normalization_dist.png')
    print("Saved normalization_dist.png")

    # 创建滑动窗口数据
    seq_length = 28
    x, y_data = sliding_windows(train_data_normalized, seq_length)
    print("Data shape:", x.shape, y_data.shape)

    # 划分训练集和测试集
    train_size = int(len(y_data) * 0.67)
    test_size = len(y_data) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y_data)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y_data[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y_data[train_size:len(y_data)])))

    # 初始化模型
    input_size = 1
    hidden_size = 512
    num_layers = 1
    num_classes = 1
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)

    # 损失函数和优化器
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7, eps=1e-08)

    # 训练循环
    print("Training Simple LSTM...")
    for epoch in progress_bar(range(NUM_EPOCHS)): 
        lstm.train()
        outputs = lstm(trainX.to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, trainY.to(device))
        loss.backward()
        optimizer.step()
        
        lstm.eval()
        valid = lstm(testX.to(device))
        vall_loss = criterion(valid, testY.to(device))
        scheduler.step(vall_loss)
        
        if epoch % 50 == 0:
            print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " % (epoch, loss.cpu().item(), vall_loss.cpu().item()))

    # 预测结果可视化
    lstm.eval()
    train_predict = lstm(dataX.to(device))
    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.data.numpy()

    # 反归一化
    data_predict = scaler.inverse_transform(data_predict)
    dataY_plot = scaler.inverse_transform(dataY_plot)

    # 绘制预测结果
    df_predict = pd.DataFrame(data_predict)
    df_predict = df_predict.set_index([dates_list[:-29]])
    df_labels = pd.DataFrame(dataY_plot)
    df_labels = df_labels.set_index([dates_list[:-29]])

    figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.axvline(x=dates_list[train_size], c='r')
    plt.plot(df_labels[0])
    plt.plot(df_predict[0])
    plt.legend(['Prediction', 'Time Series'], fontsize=21)
    plt.suptitle('Time-Series Prediction Entire Set (Simple LSTM)', fontsize=23)
    plt.savefig('prediction_simple_lstm.png')
    print("Saved prediction_simple_lstm.png")

    # 4. 实验 3: 复杂特征工程 + 复杂 LSTM
    print("\n=== Experiment 3: Complex LSTM with Features ===")
    
    # 特征工程
    DF = TS_selected.copy()
    colnames = DF.columns
    DF = DF.rename(columns={colnames[0]:'sales'})
    
    # 添加滞后特征 (Lags)
    start_time = time.time()
    for i in (1, 7, 14, 28, 365):
        DF['lag_'+str(i)] = DF['sales'].transform(lambda x: x.shift(i))
    
    # 添加滚动统计特征 (Rolling Mean/Std)
    for i in [7, 14, 28, 60, 180, 365]:
        DF['rolling_mean_'+str(i)] = DF['sales'].transform(lambda x: x.shift(28).rolling(i).mean())
        DF['rolling_std_'+str(i)]  = DF['sales'].transform(lambda x: x.shift(28).rolling(i).std())
    
    print('Feature engineering time: %0.2f min' % ((time.time() - start_time) / 60))

    # 处理缺失值
    DF = DF.replace('nan', np.nan).fillna(0)

    # 归一化所有特征
    DF_normlized = DF.copy(deep=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    scaled_data = scaler.fit_transform(DF) 
    y_scaler.fit_transform(DF['sales'].values.reshape(-1, 1))
    DF_normlized.iloc[:,:] = scaled_data

    # 添加日期特征 (星期几嵌入)
    DF_normlized = DF_normlized.reset_index()
    DF_normlized = DF_normlized.rename(columns={'index':'date'})
    DF_normlized['date'] = DF_normlized['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    DF_normlized = DF_normlized.merge(calendar_df[['date','weekday']], on='date')

    # 手动添加星期几的嵌入向量 (Embedding)
    # 这里使用的是硬编码的值，模拟嵌入层
    weekday_map = {
        'Sunday':    [0.4, -0.3, 0.6, 0.1],
        'Monday':    [0.2, 0.2, 0.5, -0.3],
        'Tuesday':   [0.1, -1.0, 1.3, 0.9],
        'Wednesday': [-0.6, 0.5, 1.2, 0.7],
        'Thursday':  [0.9, 0.2, -0.1, 0.6],
        'Friday':    [0.4, 1.1, 0.3, -1.5],
        'Saturday':  [0.3, -0.2, 0.6, 0.0]
    }
    
    for k in range(1, 5):
        DF_normlized[f'wd{k}'] = DF_normlized['weekday'].map(lambda x: weekday_map.get(x, [0,0,0,0])[k-1])

    # 选择特征
    features = ["sales", "lag_7", "lag_1", "lag_28", "lag_365", 
                "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
                "rolling_mean_28", "rolling_std_28", "rolling_mean_60", "rolling_std_60",
                "wd1", "wd2", "wd3", "wd4"] # Removed duplicate 'lag_28'
    
    data_with_features = DF_normlized[features].to_numpy()
    
    # 创建多特征滑动窗口
    x, y = sliding_windows_mutli_features(data_with_features, seq_length=28)
    print("X_data shape:", x.shape)
    print("y_data shape:", y.shape)

    # 划分训练/测试集
    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    # 初始化复杂 LSTM 模型
    input_size = 17 # Number of features
    hidden_size = 512
    num_layers = 4 # Increased layers
    num_classes = 1
    
    lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)
    lstm.apply(init_weights)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, min_lr=1e-7, eps=1e-08)

    best_val_loss = 100
    
    print("Training Complex LSTM...")
    for epoch in progress_bar(range(NUM_EPOCHS)): 
        lstm.train()
        outputs = lstm(trainX.to(device))
        optimizer.zero_grad()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1)
        
        loss = criterion(outputs, trainY.to(device))
        loss.backward()
        optimizer.step()
        
        lstm.eval()
        valid = lstm(testX.to(device))
        vall_loss = criterion(valid, testY.to(device))
        scheduler.step(vall_loss)
        
        if vall_loss.cpu().item() < best_val_loss:
            torch.save(lstm.state_dict(), 'best_model.pt')
            best_val_loss = vall_loss.cpu().item()
            # print("Saved best model epoch:", epoch, "val loss is:", best_val_loss)
        
        if epoch % 50 == 0:
            print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " % (epoch, loss.cpu().item(), vall_loss.cpu().item()))

    # 加载最佳模型并预测
    if os.path.exists('best_model.pt'):
        lstm.load_state_dict(torch.load('best_model.pt'))
        print("Loaded best model.")

    lstm.eval()
    train_predict = lstm(dataX.to(device))
    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.data.numpy()

    # 反归一化
    data_predict = y_scaler.inverse_transform(data_predict)
    dataY_plot = y_scaler.inverse_transform(dataY_plot.reshape(-1, 1))

    # 绘制最终预测结果
    df_predict = pd.DataFrame(data_predict)
    # 注意索引长度匹配
    df_predict = df_predict.set_index([dates_list[:len(df_predict)]])
    
    df_labels = pd.DataFrame(dataY_plot)
    df_labels = df_labels.set_index([dates_list[:len(df_labels)]])

    figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(df_labels[0])
    plt.plot(df_predict[0])
    plt.legend(['Prediction', 'Time Series'], fontsize=21)
    plt.suptitle('Time-Series Prediction Entire Set (Complex LSTM)', fontsize=23)
    plt.savefig('prediction_complex_lstm.png')
    print("Saved prediction_complex_lstm.png")
    
    print("Done!")
