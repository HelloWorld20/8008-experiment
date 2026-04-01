##########################Load Libraries  ####################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle
import datetime as dt
from torch.autograd import Variable
import random 
import os
from matplotlib.pyplot import figure
from fastprogress import progress_bar
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time 
import uuid

# %matplotlib inline

#from gensim.models import Word2Vec
#import gensim.downloader as api

pd.set_option('display.max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

 

 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_DIR_PATH = '/data/weijianghong/workspace/8008/dataset/'
RUN_FULL_DATASET = True    # 是否运行完整数据集，True 表示运行，False 表示仅运行子集
SUBSET_NROWS = 100 # 仅运行数据集的前 100 行，用于调试或快速测试
STORE_ID = 'CA_1'

def reduce_mem_usage(df, verbose=True):
    '''
    减少 pandas DataFrame 的内存占用。
    遍历所有列，将数值类型向下转换为更小的内存类型（如 int64 转 int8 等），从而优化内存使用。
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data():
    '''
    读取 M5 数据集中的各种 CSV 文件（如 sell_prices, calendar, sales_train_validation 等），
    并对其调用 reduce_mem_usage 进行内存优化压缩。
    '''
    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv', nrows=None if RUN_FULL_DATASET else SUBSET_NROWS)
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv', nrows=None if RUN_FULL_DATASET else SUBSET_NROWS)
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv', nrows=None if RUN_FULL_DATASET else SUBSET_NROWS)
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df
    

_,  calendar_df, sales_train_validation_df, _ = read_data()
if STORE_ID is not None:
    sales_train_validation_df = sales_train_validation_df.loc[sales_train_validation_df['store_id'] == STORE_ID].copy()

#Create date index
# 获取日历数据中的日期列，取前1913天作为训练和验证的数据范围
date_index = calendar_df['date']
dates = date_index[0:1913]
# 将字符串日期转换为 datetime.date 对象，方便后续作为时间序列索引
dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

# Create a data frame for items sales per day with item ids (with Store Id) as columns names  and dates as the index 
# 将 item_id 和 store_id 拼接生成唯一的 'item_store_id'
sales_train_validation_df['item_store_id'] = sales_train_validation_df.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)
# 提取 d_1 到 d_1913 的每日销量数据，并转置矩阵，使得行变为日期，列变为各个商品
DF_Sales = sales_train_validation_df.loc[:,'d_1':'d_1913'].T
DF_Sales.columns = sales_train_validation_df['item_store_id'].values

#Set Dates as index 
# 将之前生成的日期列表设置为 DataFrame 的索引，并转化为 pandas 的 DatetimeIndex
DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
DF_Sales.index = pd.to_datetime(DF_Sales.index)
DF_Sales.head()

#Select arbitrary index and plot the time series
# 随意选择一列（即某个商品的时间序列数据）进行后续的分析和建模
index = 10
y = pd.DataFrame(DF_Sales.iloc[:,index])
y = pd.DataFrame(y).set_index([dates_list])
TS_selected = y 
y.index = pd.to_datetime(y.index)
ax = y.plot(figsize=(30, 9),color='red')
ax.set_facecolor('lightgrey')
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.legend(fontsize=20)
plt.title(label = 'Sales Demand Selected Time Series Over Time',fontsize = 23)
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
os.makedirs('plots', exist_ok=True)
plt.savefig(f"plots/plot_{uuid.uuid4().hex[:8]}.png")
plt.close()


#del calendar_df, sales_train_validation_df,DF_Sales
#gc.collect()

SEED = 1345
def seed_everything(seed):
    '''
    设置全局随机种子，确保实验可重复性。
    包括 Python random, NumPy, PyTorch 及其 CUDA 后端的随机种子设置。
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


def init_weights(m):
    '''
    PyTorch 模型权重初始化函数。
    遍历模型的所有参数，并使用均匀分布对其进行初始化。
    '''
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        

# Re-Use the Time Series we have selected earlier
DF = TS_selected
colnames = DF.columns
DF = DF.rename(columns={colnames[0]:'sales'})
DF.tail()

start_time = time.time()
# 创建滞后特征（Lag Features）。
# 通过 shift 操作，把过去（如1天前、7天前、28天前、365天前等）的销量作为当前时间点的特征。
# 这有助于模型捕获短期的自相关性、周期性的规律和年度的季节性模式。
for i in (1,7,14,28,365):
    print('Shifting:', i)
    DF['lag_'+str(i)] = DF['sales'].transform(lambda x: x.shift(i))
print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))


 
DF = DF.set_index([dates_list])
Product = "Time Series"

################Create Plot ##############################################
fig, axs = plt.subplots(6, 1, figsize=(33, 16))
axs = axs.flatten()
ax_idx = 0

for i in (0,1,7,14,28,365):
    if i == 0:
        ax = DF['sales'].plot(fontsize = 21,
                     legend =False,
                     color=next(color_cycle),
                     ax=axs[ax_idx])
        ax.set_ylabel("Sales Demand",fontsize = 21)
        ax.set_xlabel("Date",fontsize = 21)
        ax.set_title(fontsize = 21,label = Product)

        ax_idx += 1
    else : 
        ax = DF[f'lag_{i}'].plot(fontsize = 21,
                     legend =False,
                     color=next(color_cycle),
                     ax=axs[ax_idx])
        ax.set_ylabel("Sales Demand",fontsize = 21)
        ax.set_xlabel("Date",fontsize = 21)
        ax.set_title(fontsize = 21,label = Product+f'  Lag {i}')

        ax_idx += 1
    
   
   
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig(f"plots/plot_{uuid.uuid4().hex[:8]}.png")
plt.close()
################Create Plot End##############################################

# 创建滚动统计特征（Rolling Features）。
# 注意这里的 .shift(28)：由于模型可能是为了预测未来 28 天的销量，
# 所以当前特征绝不能包含未来 28 天内的任何信息，以避免“数据泄露 (Data Leakage)”。
# 在偏移 28 天之后，再计算过去不同时间窗口（7天, 14天, 28天...）内的均值和标准差。
for i in [7,14,28,60,180,365]:
    print('Rolling period:', i)
    DF['rolling_mean_'+str(i)] = DF['sales'].transform(lambda x: x.shift(28).rolling(i).mean())
    DF['rolling_std_'+str(i)]  = DF['sales'].transform(lambda x: x.shift(28).rolling(i).std())


print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))
DF.head()

DF = DF.replace('nan', np.nan).fillna(0)
DF.head()

DF_normlized = DF.copy(deep=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
numeric_cols = DF_normlized.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    DF_normlized[col] = DF_normlized[col].astype(np.float64)
DF_normlized.loc[:, numeric_cols] = scaler.fit_transform(DF_normlized.loc[:, numeric_cols])
y_scaler.fit(DF['sales'].values.reshape(-1, 1))
DF_normlized.head()

DF_normlized = DF_normlized.reset_index()
DF_normlized = DF_normlized.rename(columns={'index':'date'})
DF_normlized.head()
DF_normlized['date'] = DF_normlized['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
DF_normlized = DF_normlized.merge(calendar_df[['date','weekday']],on='date')
DF_normlized.head()

## Adding the embedded vectors 
DF_normlized[['wd1','wd2','wd3','wd4']] = 0.0
DF_normlized.loc[DF_normlized['weekday'] =='Sunday',   ['wd1','wd2','wd3','wd4']] = [0.4, -0.3, 0.6,  0.1]
DF_normlized.loc[DF_normlized['weekday'] =='Monday',   ['wd1','wd2','wd3','wd4']] = [0.2,  0.2, 0.5, -0.3]
DF_normlized.loc[DF_normlized['weekday'] =='Tuesday',  ['wd1','wd2','wd3','wd4']] = [0.1, -1.0, 1.3,  0.9]
DF_normlized.loc[DF_normlized['weekday'] =='Wednesday',['wd1','wd2','wd3','wd4']] = [-0.6, 0.5, 1.2, 0.7]
DF_normlized.loc[DF_normlized['weekday'] =='Thursday', ['wd1','wd2','wd3','wd4']] = [0.9,  0.2, -0.1, 0.6]
DF_normlized.loc[DF_normlized['weekday'] =='Friday',   ['wd1','wd2','wd3','wd4']] = [0.4,  1.1, 0.3, -1.5]
DF_normlized.loc[DF_normlized['weekday'] =='Saturday', ['wd1','wd2','wd3','wd4']] = [0.3, -0.2, 0.6,  0.0]


fig, axs = plt.subplots(2)
 
fig.suptitle('rolling_mean_14 - Data Distribution Before and After Normalization ',fontsize = 19)
pd.DataFrame(DF['rolling_mean_14']).plot(kind='hist',ax = axs[0] , alpha=.4 , figsize=[12,6], legend = False,title = ' Before Normalization',color ='red') 
pd.DataFrame(DF_normlized['rolling_mean_14']).plot(kind='hist', ax = axs[1] ,figsize=[12,6], alpha=.4 , legend = False,title = ' After Normalization'\
                                         ,color = 'blue')

###  This function creates a sliding window or sequences of 28 days and one day label ####
###  For Multiple features                                                            ####
def sliding_windows_mutli_features(data, seq_length):
    '''
    为多特征数据创建时间序列的滑动窗口。
    根据给定的序列长度（seq_length），将输入数据划分为特征窗口（x）和对应的目标标签（y）。
    目标标签默认取自第0列。
    '''
    x = []
    y = []

    for i in range((data.shape[0])-seq_length-1):
        _x = data[i:(i+seq_length),:] ## 16 columns for features  
        _y = data[i+seq_length,0] ## column 0 contains the labbel
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y).reshape(-1,1)

# Select only the features and the target for prediction  
data_with_features = DF_normlized[["sales","lag_7","lag_1","lag_28","lag_365","rolling_mean_7",\
"rolling_std_7","rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28","rolling_mean_60","rolling_std_60",'lag_28','wd1','wd2','wd3','wd4']].to_numpy()             

#data_with_features = DF_normlized['sales'].to_numpy().reshape(-1,1)
data_with_features.shape

x , y = sliding_windows_mutli_features(data_with_features,seq_length=28)
print("X_data shape is",x.shape) # X_data shape is (1884, 28, 18)
print("y_data shape is",y.shape) # y_data shape is (1884, 1)

"""
1913 - 28 - 1 = 1884：扣除最后一个窗口（28天）得到的样本数量。
28：代表时间步长，即每个样本包含 28 天的特征。
18：代表特征数量，即每个样本包含 18 个特征。
"""



train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


print("train shape is:",trainX.size())          # train shape is: torch.Size([1262, 28, 18])
print("train label shape is:",trainY.size())    # train label shape is: torch.Size([1262, 1])
print("test shape is:",testX.size())            # test shape is: torch.Size([622, 28, 18])
print("test label shape is:",testY.size())      # test label shape is: torch.Size([622, 1])

class LSTM2(nn.Module):
    '''
    自定义的多层 LSTM 及多层感知机 (MLP) 组合模型类。
    包含多层 LSTM 用于提取时间序列特征，并在输出端使用包含 Batch Normalization 和 Dropout 的多个全连接层，以防止过拟合。
    '''

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.batch_size = 1
        #self.seq_length = seq_length
        
        self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True,dropout = 0.2)
       
        
        
        self.fc1 = nn.Linear(hidden_size,256)
        self.bn1 = nn.BatchNorm1d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(256, 128)
            
        self.bn2 = nn.BatchNorm1d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        self.fc3= nn.Linear(128, 1)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        '''
        前向传播函数。
        处理多特征输入 x，经过多层 LSTM 提取时间序列特征，
        最后通过包含批归一化 (BatchNorm) 和 Dropout 的多个全连接层输出最终预测。
        '''
        # 初始化多层 LSTM 的隐状态 h_1 和细胞状态 c_1
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
         
        
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
       
        # 传入 LSTM 进行序列处理，hn 保存了各个层在最后一个时间步的隐状态
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
     
        #print("hidden state shpe is:",hn.size())
        y = hn.view(-1, self.hidden_size)
        
        # 获取最后一层 LSTM（即第 num_layers 层）的隐状态作为整个序列的特征表示
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        #print("final state shape is:",final_state.shape)
        
        # 第一个全连接模块：Linear -> BatchNorm -> Dropout -> ReLU
        # BatchNorm 用于加速收敛并缓解内部协变量偏移，ReLU 提供非线性表达能力
        x0 = self.fc1(final_state)
        x0 = self.bn1(x0)
        x0 = self.dp1(x0)
        x0 = self.relu(x0)
        
        # 第二个全连接模块
        x0 = self.fc2(x0)
        x0 = self.bn2(x0)
        x0 = self.dp2(x0)
        
        x0 = self.relu(x0)
        
        # 输出层，将高维特征映射为单一的连续预测值
        out = self.fc3(x0)
        #print(out.size())
        return out
    
   

num_epochs = 100
learning_rate = 1e-3
input_size = 18
hidden_size = 512
num_layers = 4
num_classes = 1

best_val_loss = 100

lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
lstm.to(device)


lstm.apply(init_weights)

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=50, factor =0.5 ,min_lr=1e-7, eps=1e-08)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model

for epoch in progress_bar(range(num_epochs)): 
    # 切换为训练模式，启用 BatchNorm 和 Dropout
    lstm.train()
    print(trainX.head())
    outputs = lstm(trainX.to(device))
    
    # 清空累积的梯度
    optimizer.zero_grad()
    
    # 防止梯度爆炸：在反向传播前限制梯度的最大范数（Clip Gradient Norm）
    torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1)
    # obtain the loss function
    loss = criterion(outputs, trainY.to(device))
    
    # 反向传播计算梯度
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1)
    
    # 更新网络参数
    optimizer.step()
    
    # 切换为评估模式，计算验证集上的表现
    lstm.eval()
    valid = lstm(testX.to(device))
    vall_loss = criterion(valid, testY.to(device))
    
    # 学习率调度器根据验证集上的损失表现自动调整学习率
    scheduler.step(vall_loss)
    #scheduler.step()
    
    # 模型保存逻辑：如果当前验证集的损失低于历史最低损失，则保存当前模型的状态字典
    if vall_loss.cpu().item() < best_val_loss:
         torch.save(lstm.state_dict(), 'best_model.pt')
         print("saved best model epoch:",epoch,"val loss is:",vall_loss.cpu().item())
         best_val_loss = vall_loss.cpu().item()
        
    
    if epoch % 50 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))

######Prediction###############
lstm.load_state_dict(torch.load('best_model.pt'))

lstm.eval()
train_predict = lstm(dataX.to(device))
data_predict = train_predict.cpu().data.numpy()
dataY_plot = dataY.data.numpy()
print(data_predict.shape)
print(dataY_plot.shape)


## Inverse Normalize 
data_predict = y_scaler.inverse_transform(data_predict)
dataY_plot = y_scaler.inverse_transform(dataY_plot.reshape(-1, 1))

## Add dates
df_predict = pd.DataFrame(data_predict)
df_predict = df_predict.set_index([dates_list[:-29]])
df_labels = pd.DataFrame(dataY_plot)
df_labels = df_labels.set_index([dates_list[:-29]])

# Plot 
figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
plt.axvline(x=dates_list[train_size], c='r')
plt.plot( df_labels[0])
plt.plot(df_predict[0])
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction Entire Set',fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
os.makedirs('plots', exist_ok=True)
plt.savefig(f"plots/plot_{uuid.uuid4().hex[:8]}.png")
plt.close()



#######Plot the test set ##########################
figure(num=None, figsize=(23, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_labels.iloc[-testX.size()[0]:][0])
plt.plot(df_predict.iloc[-testX.size()[0]:][0])
plt.legend(['Prediction','Time Series'],fontsize = 21)
plt.suptitle('Time-Series Prediction Test',fontsize = 23)
plt.xticks(fontsize=21 )
plt.yticks(fontsize=21 )
plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
plt.xlabel(xlabel = 'Date',fontsize = 21)
os.makedirs('plots', exist_ok=True)
plt.savefig(f"plots/plot_{uuid.uuid4().hex[:8]}.png")
plt.close()

np.sqrt(((dataY_plot[-testX.size()[0]:] - data_predict[-testX.size()[0]:] ) ** 2).mean())
