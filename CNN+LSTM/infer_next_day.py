'''
使用说明
- 作用：加载已训练好的多特征 LSTM+MLP 模型，对最近 28 天窗口进行一次推理，输出下一日预测值（已逆归一化）。
- 依赖：best_model.pt、M5 数据集（calendar.csv、sales_train_validation.csv）。
- 命令行示例：
  uv run python CNN+LSTM/infer_next_day.py --model-path /data/weijianghong/workspace/8008/CNN+LSTM/best_model.pt --data-dir /data/weijianghong/workspace/8008/dataset/ --full
  
  uv run python CNN+LSTM/infer_next_day.py --model-path CNN+LSTM/best_model.pt --data-dir /data/weijianghong/workspace/8008/dataset/ --subset-nrows 200 --index 10 --seq-len 28
- 参数说明：
  --model-path     模型文件路径，默认 CNN+LSTM/best_model.pt
  --data-dir       数据目录路径，默认 /data/weijianghong/workspace/8008/dataset/
  --full           是否使用全量数据（加上该开关则全量读取）
  --subset-nrows   使用子集时的行数，默认 100
  --index          选择的商品列索引，默认 10
  --seq-len        滑窗长度，默认 28
'''
import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reduce_mem_usage(df, verbose=True):
    '''
    压缩 DataFrame 数值列的内存占用，按范围降级到更小的整数或浮点类型。
    '''
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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data(data_dir, run_full, subset_nrows):
    '''
    读取所需 CSV 数据，支持按开关选择全量或子集。
    返回 calendar_df 与 sales_train_validation_df。
    '''
    calendar_df = pd.read_csv(os.path.join(data_dir, 'calendar.csv'))
    calendar_df = reduce_mem_usage(calendar_df, verbose=False)
    sales_train_validation_df = pd.read_csv(
        os.path.join(data_dir, 'sales_train_validation.csv'),
        nrows=None if run_full else subset_nrows
    )
    return calendar_df, sales_train_validation_df


def build_ts_selected(calendar_df, sales_df, index=10):
    '''
    基于给定 index 选择单个商品的时间序列，并生成日期列表。
    返回 TS_selected 与 dates_list。
    '''
    date_index = calendar_df['date']
    dates = date_index[0:1913]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    sales_df['item_store_id'] = sales_df.apply(lambda x: x['item_id'] + '_' + x['store_id'], axis=1)
    df_sales = sales_df.loc[:, 'd_1':'d_1913'].T
    df_sales.columns = sales_df['item_store_id'].values
    df_sales = pd.DataFrame(df_sales).set_index([dates_list])
    df_sales.index = pd.to_datetime(df_sales.index)
    y = pd.DataFrame(df_sales.iloc[:, index])
    y = pd.DataFrame(y).set_index([dates_list])
    y.index = pd.to_datetime(y.index)
    TS_selected = y
    return TS_selected, dates_list


def build_features_df(TS_selected, dates_list, calendar_df):
    '''
    基于单条时间序列构建滞后与滚动特征，做数值归一化并添加 weekday 嵌入。
    返回 DF_normlized 与 y_scaler。
    '''
    DF = TS_selected.copy()
    colnames = DF.columns
    DF = DF.rename(columns={colnames[0]: 'sales'})
    for i in (1, 7, 14, 28, 365):
        DF['lag_' + str(i)] = DF['sales'].transform(lambda x: x.shift(i))
    DF = DF.set_index([dates_list])
    for i in [7, 14, 28, 60, 180, 365]:
        DF['rolling_mean_' + str(i)] = DF['sales'].transform(lambda x: x.shift(28).rolling(i).mean())
        DF['rolling_std_' + str(i)] = DF['sales'].transform(lambda x: x.shift(28).rolling(i).std())
    DF = DF.replace('nan', np.nan).fillna(0)
    DF_normlized = DF.copy(deep=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    numeric_cols = DF_normlized.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        DF_normlized[col] = DF_normlized[col].astype(np.float64)
    DF_normlized.loc[:, numeric_cols] = scaler.fit_transform(DF_normlized.loc[:, numeric_cols])
    y_scaler.fit(DF['sales'].values.reshape(-1, 1))
    DF_normlized = DF_normlized.reset_index()
    DF_normlized = DF_normlized.rename(columns={'index': 'date'})
    DF_normlized['date'] = DF_normlized['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    DF_normlized = DF_normlized.merge(calendar_df[['date', 'weekday']], on='date')
    DF_normlized[['wd1', 'wd2', 'wd3', 'wd4']] = 0.0
    DF_normlized.loc[DF_normlized['weekday'] == 'Sunday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.4, -0.3, 0.6, 0.1]
    DF_normlized.loc[DF_normlized['weekday'] == 'Monday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.2, 0.2, 0.5, -0.3]
    DF_normlized.loc[DF_normlized['weekday'] == 'Tuesday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.1, -1.0, 1.3, 0.9]
    DF_normlized.loc[DF_normlized['weekday'] == 'Wednesday', ['wd1', 'wd2', 'wd3', 'wd4']] = [-0.6, 0.5, 1.2, 0.7]
    DF_normlized.loc[DF_normlized['weekday'] == 'Thursday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.9, 0.2, -0.1, 0.6]
    DF_normlized.loc[DF_normlized['weekday'] == 'Friday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.4, 1.1, 0.3, -1.5]
    DF_normlized.loc[DF_normlized['weekday'] == 'Saturday', ['wd1', 'wd2', 'wd3', 'wd4']] = [0.3, -0.2, 0.6, 0.0]
    return DF_normlized, y_scaler


class LSTM2(nn.Module):
    '''
    多特征 LSTM + MLP 模型：LSTM 提取序列特征，MLP 端包含 BN/Dropout/Relu，输出连续值。
    '''
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = 1
        self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        前向传播：输入形状 (B, T, F)，输出形状 (B, 1)。
        '''
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
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


def load_and_infer_next_day(model_path, DF_normlized, y_scaler, seq_length=28):
    '''
    加载模型权重，使用最近 seq_length 天的多特征窗口进行一次推理，返回下一日预测值（原始尺度）。
    '''
    features_cols = ["sales", "lag_7", "lag_1", "lag_28", "lag_365", "rolling_mean_7",
                     "rolling_std_7", "rolling_mean_14", "rolling_std_14", "rolling_mean_28",
                     "rolling_std_28", "rolling_mean_60", "rolling_std_60", "lag_28", "wd1", "wd2", "wd3", "wd4"]
    model = LSTM2(num_classes=1, input_size=18, hidden_size=512, num_layers=4).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    data_with_features = DF_normlized[features_cols].to_numpy()
    if data_with_features.shape[0] < seq_length:
        raise ValueError("可用样本长度不足以构造推理窗口")
    window = data_with_features[-seq_length:, :]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        yhat = model(x)
    pred = y_scaler.inverse_transform(yhat.cpu().numpy())
    return float(pred[0, 0])


def main():
    '''
    命令行入口：读取数据与特征，加载模型，打印下一日预测值。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=os.path.join('CNN+LSTM', 'best_model.pt'))
    parser.add_argument('--data-dir', type=str, default='/data/weijianghong/workspace/8008/dataset/')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--subset-nrows', type=int, default=100)
    parser.add_argument('--index', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=28)
    args = parser.parse_args()

    calendar_df, sales_df = read_data(args.data_dir, args.full, args.subset_nrows)
    TS_selected, dates_list = build_ts_selected(calendar_df, sales_df, index=args.index)
    DF_normlized, y_scaler = build_features_df(TS_selected, dates_list, calendar_df)
    value = load_and_infer_next_day(args.model_path, DF_normlized, y_scaler, seq_length=args.seq_len)
    print("下一日预测值:", value)


if __name__ == '__main__':
    main()
