# learning-pytorch-lstm.py 流水线总览

本文概述当前训练脚本的端到端流程、关键模块与可配置开关，便于理解与复用。

## 开关与路径

- 设备选择：CUDA 可用则自动使用 GPU，否则使用 CPU。参见 [learning-pytorch-lstm.py:36](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L36)。
- 数据路径与规模控制：
  - 数据目录 INPUT_DIR_PATH。参见 [learning-pytorch-lstm.py:38](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L38)。
  - RUN_FULL_DATASET 控制是否读全量；SUBSET_NROWS 控制读子集时的行数。参见 [learning-pytorch-lstm.py:39-40](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L39-L40)。
  - STORE_ID 仅保留指定门店的数据（例如 'CA_1'）。参见 [learning-pytorch-lstm.py:41](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L41) 与筛选位置 [learning-pytorch-lstm.py:97-98](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L97-L98)。

## 数据读取与预处理

- 读取 CSV 并做内存压缩：sell_prices、calendar、sales_train_validation，受 RUN_FULL_DATASET/SUBSET_NROWS 控制。参见 [read_data](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L76-L92) 与压缩函数 [reduce_mem_usage](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L42-L69)。
- 构造日期索引（使用 calendar 前 1913 天）并生成 Python 日期列表 dates_list。参见 [learning-pytorch-lstm.py:102-105](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L102-L105)。
- 在 sales_train_validation_df 中拼接 item_id 与 store_id 得到 item_store_id，并将 d_1..d_1913 转置为以日期为索引、商品为列的 DF_Sales。参见 [learning-pytorch-lstm.py:108-117](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L108-L117)。
- 仅选择一家门店的数据（由 STORE_ID 控制），之后的流程均在该门店范围内进行。参见 [learning-pytorch-lstm.py:97-98](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L97-L98)。

## 单商品时间序列选取

- 通过 index 选择一个商品列作为时间序列 y，并用于后续特征工程与建模。参见 [learning-pytorch-lstm.py:122-126](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L122-L126)。

## 特征工程

- 滞后特征（Lag）：为 sales 构造 lag_1、lag_7、lag_14、lag_28、lag_365。参见 [learning-pytorch-lstm.py:175-181](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L175-L181)。
- 可视化当前销量与各 Lag 序列，输出到 plots。参见 [learning-pytorch-lstm.py:189-225](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L189-L225) 与保存位置 [learning-pytorch-lstm.py:219-224](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L219-L224)。
- 滚动统计特征（Rolling）：为避免数据泄露，先整体 shift(28)，再计算窗口 7/14/28/60/180/365 的 rolling_mean 与 rolling_std。参见 [learning-pytorch-lstm.py:232-234](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L232-L234)。
- 缺失处理与归一化：
  - 将 'nan' 替换为 np.nan 并填 0；复制得到 DF_normlized。参见 [learning-pytorch-lstm.py:240-243](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L240-L243)。
  - 仅对数值列进行 MinMaxScaler(-1,1) 缩放，先显式转成 float 再缩放，避免把浮点写回整型列引发错误。参见 [learning-pytorch-lstm.py:244-249](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L244-L249)。
  - 目标值使用独立的 y_scaler 仅做 fit，以便之后逆归一化。参见 [learning-pytorch-lstm.py:245-250](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L245-L250)。
- 日期与星期特征：重置索引为 'date'，与 calendar 合并 weekday；添加四维的星期嵌入向量 wd1~wd4。参见 [learning-pytorch-lstm.py:252-267](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L252-L267)。

## 滑窗与数据集

- 构造多特征滑窗数据（seq_length=28），标签默认取第 0 列（sales）。参见函数 [sliding_windows_mutli_features](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L280-L290)。
- 选择用于训练的特征列并转为 numpy：
  - 列表包含 ["sales","lag_7","lag_1","lag_28","lag_365","rolling_mean_7","rolling_std_7","rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28","rolling_mean_60","rolling_std_60","lag_28","wd1","wd2","wd3","wd4"]。
  - 注意 lag_28 在列表中出现两次，这是与当前模型的 input_size=18 对齐的设计。参见 [learning-pytorch-lstm.py:298-299](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L298-L299)。
- 划分训练/测试集（约 67/33），并转为张量。参见 [learning-pytorch-lstm.py:308-318](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L308-L318)。

## 模型结构与训练

- 模型：LSTM2（多层 LSTM + 两层 MLP，含 BatchNorm/Dropout/ReLU）。参见类定义 [learning-pytorch-lstm.py:326-400](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L326-L400)。
- 超参：num_epochs=100、learning_rate=1e-3、input_size=18、hidden_size=512、num_layers=4、num_classes=1。参见 [learning-pytorch-lstm.py:404-409](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L404-L409)。
- 训练流程：
  - 初始化权重（均匀分布），MSE 损失，Adam 优化器，ReduceLROnPlateau 调度器。参见 [learning-pytorch-lstm.py:417-425](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L417-L425)。
  - 循环训练：前向、梯度清零、梯度裁剪、反向传播、优化步，计算验证集损失并按需降低学习率。参见 [learning-pytorch-lstm.py:429-456](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L429-L456)。
  - 模型保存：若验证损失刷新最优，则保存当前 state_dict 至 best_model.pt。参见 [learning-pytorch-lstm.py:459-462](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L459-L462)。

## 推理与可视化

- 加载最优权重，整集前向得到预测与标签，并用 y_scaler 逆归一化；分别绘制整集与测试区间曲线图到 plots。参见 [learning-pytorch-lstm.py:468-520](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm.py#L468-L520)。
- 独立推理脚本：提供单次“下一日预测”的脚本，含命令行参数与数据读取流程，见 [infer_next_day.py](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/infer_next_day.py) 与其使用说明 [infer_next_day.py:1-15](file:///Users/leon.w/workspace/cityu/8008/CNN+LSTM/infer_next_day.py#L1-L15)。

## 快速运行

- 训练：
  ```bash
  uv run python CNN+LSTM/learning-pytorch-lstm.py
  ```
- 单步推理（下一日预测）：
  ```bash
  uv run python CNN+LSTM/infer_next_day.py --model-path /Users/leon.w/workspace/cityu/8008/CNN+LSTM/best_model.pt --data-dir /data/weijianghong/workspace/8008/dataset/ --full
  ```
