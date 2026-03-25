#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列预测：EDA、特征工程和建模
Time Series Forecasting: EDA, Feature Engineering and Modelling

这个脚本将Jupyter Notebook转换为可执行的Python脚本
This script converts Jupyter Notebook to executable Python script
"""

import os
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')

# 导入LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError:
    print("警告：LightGBM未安装，建模功能将不可用")
    LGBMRegressor = None
    lgb = None

def load_data():
    """
    加载数据
    Load data from CSV files
    """
    print("正在加载数据...")
    
    # 本地数据文件路径
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据集目录位于脚本目录的上一级目录的 dataset 文件夹中
    dataset_dir = os.path.join(os.path.dirname(script_dir), 'dataset')
    
    print(f"数据目录: {dataset_dir}")
    
    local_files = [
        os.path.join(dataset_dir, 'sales_train_evaluation.csv'),
        os.path.join(dataset_dir, 'calendar.csv'),
        os.path.join(dataset_dir, 'sell_prices.csv')
    ]
    
    # 检查本地文件是否存在
    all_files_exist = True
    for file_path in local_files:
        if not os.path.exists(file_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        print("使用本地数据文件")
        sales = pd.read_csv(local_files[0])
        calendar = pd.read_csv(local_files[1])
        prices = pd.read_csv(local_files[2])
    else:
        print("错误：本地数据文件不存在，请确保以下文件存在：")
        for file_path in local_files:
            print(f"  - {file_path}")
        raise FileNotFoundError("本地数据文件缺失")
    
    sales.name = 'sales'
    calendar.name = 'calendar'
    prices.name = 'prices'
    
    return sales, calendar, prices


def add_zero_sales(sales):
    """
    为测试期间添加零销售数据
    Add zero sales for test period (days 1942-1969)
    """
    print("为测试期间添加零销售数据...")
    for d in range(1942, 1970):
        col = 'd_' + str(d)
        sales[col] = 0
        sales[col] = sales[col].astype(np.int16)
    return sales


def downcast(df):
    """
    数据类型优化以减少内存使用
    Downcast data types to reduce memory usage
    """
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    
    return df


def optimize_memory(sales, calendar, prices):
    """
    优化所有数据框的内存使用
    Optimize memory usage for all dataframes
    """
    print("优化内存使用...")
    
    # 记录优化前的内存使用
    sales_bd = np.round(sales.memory_usage().sum()/(1024*1024), 1)
    calendar_bd = np.round(calendar.memory_usage().sum()/(1024*1024), 1)
    prices_bd = np.round(prices.memory_usage().sum()/(1024*1024), 1)
    
    print(f"优化前内存使用: sales {sales_bd}MB, calendar {calendar_bd}MB, prices {prices_bd}MB")
    
    # 执行优化
    sales = downcast(sales)
    prices = downcast(prices)
    calendar = downcast(calendar)
    
    # 记录优化后的内存使用
    sales_ad = np.round(sales.memory_usage().sum()/(1024*1024), 1)
    calendar_ad = np.round(calendar.memory_usage().sum()/(1024*1024), 1)
    prices_ad = np.round(prices.memory_usage().sum()/(1024*1024), 1)
    
    print(f"优化后内存使用: sales {sales_ad}MB, calendar {calendar_ad}MB, prices {prices_ad}MB")
    
    return sales, calendar, prices


def melt_data(sales):
    """
    将宽格式数据转换为长格式
    Convert wide format data to long format
    """
    print("正在转换数据格式...")
    df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                  var_name='d', value_name='sold').dropna()
    return df


def merge_data(df, calendar, prices):
    """
    合并所有数据源
    Merge all data sources
    """
    print("正在合并数据...")
    df = pd.merge(df, calendar, on='d', how='left')
    df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left')
    return df


def visualize_item_distribution(sales):
    """
    可视化商品分布
    Visualize item distribution
    """
    print("生成商品分布可视化...")
    group = sales.groupby(['state_id','store_id','cat_id','dept_id'],as_index=False)['item_id'].count().dropna()
    group['USA'] = 'United States of America'
    group.rename(columns={'state_id':'State','store_id':'Store','cat_id':'Category','dept_id':'Department','item_id':'Count'},inplace=True)
    
    fig = px.treemap(group, path=['USA', 'State', 'Store', 'Category', 'Department'], values='Count',
                      color='Count', color_continuous_scale=px.colors.sequential.Sunset,
                      title='Walmart: Distribution of items')
    fig.update_layout(template='seaborn')
    fig.show()


def analyze_price_distribution(df):
    """
    分析商品价格分布
    Analyze price distribution
    """
    print("分析商品价格分布...")
    # 按商店分析价格分布
    group_price_store = df.groupby(['state_id','store_id','item_id'],as_index=False)['sell_price'].mean().dropna()
    fig = px.violin(group_price_store, x='store_id', color='state_id', y='sell_price', 
                     box=True, hover_name='item_id')
    fig.update_xaxes(title_text='Store')
    fig.update_yaxes(title_text='Selling Price($)')
    fig.update_layout(template='seaborn',title='Distribution of Items prices wrt Stores',
                     legend_title_text='State')
    fig.show()


def analyze_sales_trends(df):
    """
    分析销售趋势
    Analyze sales trends
    """
    print("分析销售趋势...")
    # 按类别分析销售趋势
    sales_by_category = df.groupby(['cat_id','date'],as_index=False)['sold'].sum()
    
    fig = px.line(sales_by_category, x='date', y='sold', color='cat_id',
                  title='Sales Trends by Category')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Total Sales')
    fig.update_layout(template='seaborn')
    fig.show()


def analyze_seasonality(df):
    """
    分析季节性
    Analyze seasonality
    """
    print("分析季节性...")
    # 添加月份和星期信息
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
    
    # 按月份分析销售
    monthly_sales = df.groupby('month',as_index=False)['sold'].sum()
    fig1 = px.bar(monthly_sales, x='month', y='sold', title='Monthly Sales Pattern')
    fig1.update_xaxes(title_text='Month')
    fig1.update_yaxes(title_text='Total Sales')
    fig1.update_layout(template='seaborn')
    fig1.show()
    
    # 按星期分析销售
    weekday_sales = df.groupby('weekday',as_index=False)['sold'].sum()
    # 重新排序星期
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_sales['weekday'] = pd.Categorical(weekday_sales['weekday'], categories=weekday_order, ordered=True)
    weekday_sales = weekday_sales.sort_values('weekday')
    
    fig2 = px.bar(weekday_sales, x='weekday', y='sold', title='Weekly Sales Pattern')
    fig2.update_xaxes(title_text='Day of Week')
    fig2.update_yaxes(title_text='Total Sales')
    fig2.update_layout(template='seaborn')
    fig2.show()


def prepare_feature_engineering(df):
    """
    准备特征工程
    Prepare feature engineering
    """
    print("准备特征工程...")
    
    # 1. 首先将所有需要的列转换为分类类型
    print("转换列为分类类型...")
    categorical_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 2. 存储类别映射以便后续使用
    print("存储类别映射...")
    d_id = dict(zip(df.id.cat.codes, df.id))
    d_item_id = dict(zip(df.item_id.cat.codes, df.item_id))
    d_dept_id = dict(zip(df.dept_id.cat.codes, df.dept_id))
    d_cat_id = dict(zip(df.cat_id.cat.codes, df.cat_id))
    d_store_id = dict(zip(df.store_id.cat.codes, df.store_id))
    d_state_id = dict(zip(df.state_id.cat.codes, df.state_id))
    
    # 3. 清理内存
    print("清理内存...")
    gc.collect()
    
    # 4. 标签编码
    print("执行标签编码...")
    
    # 调试：显示数据类型
    print("编码前的数据类型:")
    print(df.dtypes)
    
    # 将d列转换为数字
    df['d'] = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
    
    # 将所有字符串列转换为类别类型，然后编码
    string_cols = ['weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
            print(f"编码字符串列: {col}")
    
    # 删除不需要的列
    cols_to_drop = ['date']  # 日期列的特征已经提取
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"删除列: {col}")
    
    # 将类别变量转换为编码
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, col_type in enumerate(types):
        col_name = cols[i]
        if col_type.name == 'category':
            df[col_name] = df[col_name].cat.codes
            print(f"编码分类列: {col_name}")
        elif col_type.name == 'bool':
            # 将布尔列转换为整数
            df[col_name] = df[col_name].astype(np.int8)
            print(f"编码布尔列: {col_name}")
    
    # 调试：显示编码后的数据类型
    print("编码后的数据类型:")
    print(df.dtypes)
    
    return df, d_id, d_item_id, d_dept_id, d_cat_id, d_store_id, d_state_id


def introduce_lag_features(df):
    """
    引入滞后特征
    Introduce lag features
    """
    print("引入滞后特征...")
    lags = [1, 2, 3, 6, 12, 24, 36]
    
    for lag in lags:
        df[f'sold_lag_{lag}'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                          as_index=False)['sold'].shift(lag).astype(np.float16)
    
    return df


def create_mean_encodings(df):
    """
    创建均值编码特征
    Create mean encoding features
    """
    print("创建均值编码特征...")
    
    # 按不同维度计算均值编码
    encodings = [
        ('item_id', df.groupby('item_id', as_index=False)['sold'].mean()),
        ('state_id', df.groupby('state_id', as_index=False)['sold'].mean()),
        ('store_id', df.groupby('store_id', as_index=False)['sold'].mean()),
        ('cat_id', df.groupby('cat_id', as_index=False)['sold'].mean()),
        ('dept_id', df.groupby('dept_id', as_index=False)['sold'].mean()),
        (['cat_id', 'dept_id'], df.groupby(['cat_id', 'dept_id'], as_index=False)['sold'].mean()),
        (['store_id', 'item_id'], df.groupby(['store_id', 'item_id'], as_index=False)['sold'].mean()),
        (['cat_id', 'item_id'], df.groupby(['cat_id', 'item_id'], as_index=False)['sold'].mean()),
        (['dept_id', 'item_id'], df.groupby(['dept_id', 'item_id'], as_index=False)['sold'].mean()),
        (['state_id', 'store_id'], df.groupby(['state_id', 'store_id'], as_index=False)['sold'].mean()),
        (['state_id', 'store_id', 'cat_id'], df.groupby(['state_id', 'store_id', 'cat_id'], as_index=False)['sold'].mean()),
        (['store_id', 'cat_id', 'dept_id'], df.groupby(['store_id', 'cat_id', 'dept_id'], as_index=False)['sold'].mean())
    ]
    
    for group_cols, group_df in encodings:
        if isinstance(group_cols, list):
            merge_cols = group_cols
            feature_name = 'sold_mean_' + '_'.join(group_cols)
        else:
            merge_cols = [group_cols]
            feature_name = 'sold_mean_' + group_cols
        
        df = pd.merge(df, group_df.rename(columns={'sold': feature_name}), 
                     on=merge_cols, how='left')
    
    return df


def train_lightgbm_models(df, d_store_id):
    """
    训练LightGBM模型
    Train LightGBM models for each store
    """
    print("训练LightGBM模型...")
    
    # 获取商店ID
    stores = df['store_id'].unique().tolist()
    
    # 创建空的预测结果DataFrame
    valid_preds_df = pd.DataFrame()
    eval_preds_df = pd.DataFrame()
    
    # 分store来训练。
    for store in stores:
        store_df = df[df['store_id'] == store]
        
        # 分割训练、验证和测试数据
        X_train = store_df[store_df['d'] < 1914].drop('sold', axis=1)
        y_train = store_df[store_df['d'] < 1914]['sold']
        X_valid = store_df[(store_df['d'] >= 1914) & (store_df['d'] < 1942)].drop('sold', axis=1)
        y_valid = store_df[(store_df['d'] >= 1914) & (store_df['d'] < 1942)]['sold']
        X_test = store_df[store_df['d'] >= 1942].drop('sold', axis=1)
        
        # 训练模型
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=8,
            num_leaves=50,
            min_child_weight=300,
            random_state=42
        )
        
        print(f'***** 商店预测: {d_store_id[store]} *****')

        model.fit(X_train, y_train, 
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric='rmse', 
                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)])
        
        # 预测
        valid_pred_values = model.predict(X_valid)
        eval_pred_values = model.predict(X_test)
        
        # 创建包含预测结果的临时DataFrame
        valid_temp = pd.DataFrame({
            'id': X_valid['id'],
            'd': X_valid['d'],
            'pred': valid_pred_values
        })
        eval_temp = pd.DataFrame({
            'id': X_test['id'],
            'd': X_test['d'],
            'pred': eval_pred_values
        })
        
        # 添加到总的预测结果中
        valid_preds_df = pd.concat([valid_preds_df, valid_temp], ignore_index=True)
        eval_preds_df = pd.concat([eval_preds_df, eval_temp], ignore_index=True)
        
        # 保存模型
        filename = f'model_{d_store_id[store]}.pkl'
        joblib.dump(model, filename)
        
        # 清理内存
        del model, X_train, y_train, X_valid, y_valid, X_test
        gc.collect()
    
    return valid_preds_df, eval_preds_df


def analyze_feature_importance(df):
    """
    分析特征重要性
    Analyze feature importance
    """
    print("分析特征重要性...")
    
    feature_importance_df = pd.DataFrame()
    features = [f for f in df.columns if f != 'sold']
    
    for filename in os.listdir('.'):
        if 'model_' in filename and filename.endswith('.pkl'):
            # 加载模型
            model = joblib.load(filename)
            store_importance_df = pd.DataFrame()
            store_importance_df["feature"] = features
            store_importance_df["importance"] = model.feature_importances_
            store_importance_df["store"] = filename[6:-4]  # 提取商店名
            feature_importance_df = pd.concat([feature_importance_df, store_importance_df], axis=0)
            del model
    
    # 显示最重要的20个特征
    if not feature_importance_df.empty:
        cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:20].index
        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
        
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM 特征重要性 (按商店平均)')
        plt.tight_layout()
        plt.show()
    
    return feature_importance_df


def prepare_submission(df, valid_preds_df, eval_preds_df, d_id, sales):
    """
    准备提交文件
    Prepare submission file
    """
    print("准备提交文件...")
    
    # 使用预测DataFrame直接进行透视表操作
    validation = valid_preds_df.rename(columns={'pred': 'sold'})
    evaluation = eval_preds_df.rename(columns={'pred': 'sold'})
    
    # 准备验证结果
    validation = pd.pivot(validation, index='id', columns='d', values='sold').reset_index()
    validation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    validation['id'] = validation['id'].map(d_id).str.replace('evaluation', 'validation')
    
    # 准备测试结果
    evaluation = pd.pivot(evaluation, index='id', columns='d', values='sold').reset_index()
    evaluation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    evaluation['id'] = evaluation['id'].map(d_id)
    
    # 合并并保存提交文件
    submit = pd.concat([validation, evaluation]).reset_index(drop=True)
    submit.to_csv('submission.csv', index=False)
    
    print(f"提交文件已保存，包含 {len(submit)} 行数据")
    return submit


def main():
    """
    主函数
    Main function
    """
    print("开始时间序列预测分析...")
    
    # 1. 加载数据
    sales, calendar, prices = load_data()
    
    # 2. 添加测试期间的零销售数据
    sales = add_zero_sales(sales)
    
    # 3. 内存优化
    sales, calendar, prices = optimize_memory(sales, calendar, prices)
    
    # 4. 数据格式转换
    df = melt_data(sales)
    
    # 5. 数据合并
    df = merge_data(df, calendar, prices)
    
    # 6. 对合并后的数据进行内存优化
    print("优化合并后数据的内存使用...")
    df = downcast(df)

    # 7. 商品分布可视化
    # visualize_item_distribution(sales)
    
    # 8. EDA分析
    print("\n=== 开始EDA分析 ===")
    # analyze_price_distribution(df)
    # analyze_sales_trends(df)
    # analyze_seasonality(df)
    
    # 9. 特征工程
    print("\n=== 开始特征工程 ===")
    df, d_id, d_item_id, d_dept_id, d_cat_id, d_store_id, d_state_id = prepare_feature_engineering(df)
    
    # 10. 建模和预测（如果LightGBM可用）
    if LGBMRegressor is not None:
        print("\n=== 开始建模和预测 ===")
        valid_preds, eval_preds = train_lightgbm_models(df, d_store_id)
        
        # 11. 特征重要性分析
        print("\n=== 特征重要性分析 ===")
        analyze_feature_importance(df)
        
        # 12. 准备提交文件
        print("\n=== 准备提交文件 ===")
        submit = prepare_submission(df, valid_preds, eval_preds, d_id, sales)
        print(f"提交文件已生成：submission.csv，包含 {len(submit)} 行数据")
    else:
        print("\n=== LightGBM未安装，跳过建模步骤 ===")
    
    print("\n分析完成！")
    print(f"最终数据框形状: {df.shape}")
    
    # 清理内存
    gc.collect()


if __name__ == "__main__":
    main()


