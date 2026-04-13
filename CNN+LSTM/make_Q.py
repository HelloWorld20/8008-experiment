import numpy as np
import pandas as pd
import os

# =====================================================================
# Step 3: ABCA Solver - 搜索空间构造 (Search Space Construction for Q_it)
# =====================================================================
# 核心目标：为人工蜂群算法（ABCA）定义订货量 Q 的合理搜索范围（上下界），
# 并提供判断某个 Q 是否满足业务约束（如预算、仓储容量）的逻辑。
# =====================================================================

class InventorySearchSpace:
    def __init__(self, num_skus, prices, volumes, initial_inventory, budget, capacity):
        """
        初始化库存优化搜索空间的约束条件。
        
        参数:
        - num_skus: 产品的种类数量 (N)
        - prices: 各个产品的单价数组 (N,)
        - volumes: 各个产品的单位体积数组 (N,)
        - initial_inventory: 各个产品的初始库存量数组 (N,)
        - budget: 总采购预算限制 (B)
        - capacity: 仓库总容量限制 (C)
        """
        self.N = num_skus
        self.prices = np.array(prices)
        self.volumes = np.array(volumes)
        self.inventory = np.array(initial_inventory)
        self.budget = budget
        self.capacity = capacity

    def get_search_bounds(self, y_pred, alpha=2.0, beta=5):
        """
        构造搜索空间 Q 的上下界。
        
        为什么需要上下界？
        如果没有边界，ABCA 会在 [0, 无穷大] 之间盲目搜索，效率极低。
        订货量 Q 的大小应该强依赖于神经网络预测出的需求量 y_pred。
        
        参数:
        - y_pred: 神经网络预测的未来需求量 (N,)
        - alpha, beta: 探索系数，用于控制搜索空间的宽容度。
          (例如：最高订购量 = alpha * 预测需求 + beta)
          
        返回:
        - Q_min: 最小订货量数组 (N,)
        - Q_max: 最大订货量数组 (N,)
        """
        # 1. 最小订货量 Q_min
        # 最保守的策略是：预测需求 - 现有库存，如果库存足够则不订货。
        # 但在 ABCA 初始搜索时，我们通常将下界直接设为 0，允许算法探索不订货的情况。
        Q_min = np.zeros_like(y_pred)
        
        # 2. 最大订货量 Q_max
        # 激进的策略：基于预测值，给予一定的安全缓冲（Safety Stock 思想）。
        # 这里定义为：预测值的 alpha 倍，再加上一个基础常数 beta。
        # 加上 beta 是为了防止 y_pred 预测为 0 时，搜索空间锁死在 0，导致算法无法探索小批量订货的可能性。
        Q_max = (alpha * y_pred + beta).astype(int)
        
        # 修正：确保 Q_max 不能小于 0（极端异常情况下）
        Q_max = np.maximum(Q_max, 0)
        
        return Q_min, Q_max

    def is_feasible(self, Q):
        """
        判断某个生成的订货方案 Q 是否满足业务约束（可行性检查）。
        
        在 ABCA 中，无论是雇佣蜂、观察蜂还是侦察蜂生成了新的解 Q，
        都必须通过此函数检查。如果不满足，则该解无效（给予极大的惩罚或直接丢弃）。
        
        参数:
        - Q: 生成的订货量方案 (N,)
        
        返回:
        - bool: True 表示方案可行，False 表示违反了约束。
        """
        # 约束 1：总采购成本不能超过预算 (Budget Constraint)
        # sum(Price_i * Q_i) <= BUDGET
        total_cost = np.sum(self.prices * Q)
        if total_cost > self.budget:
            return False
            
        # 约束 2：总库存体积不能超过仓库容量 (Capacity Constraint)
        # sum(Volume_i * (Inventory_i + Q_i)) <= CAPACITY
        # 注意：这里我们保守估计，假设货还没卖出去，新货就送到了。
        total_volume = np.sum(self.volumes * (self.inventory + Q))
        if total_volume > self.capacity:
            return False
            
        # 约束 3：订货量不能为负数 (Non-negativity Constraint)
        if np.any(Q < 0):
            return False
            
        return True

    def generate_random_solution(self, Q_min, Q_max):
        """
        在搜索空间内随机生成一个合法的初始解。
        
        这个函数用于 ABCA 算法的初始化阶段（生成初始种群），
        以及侦察蜂（Scout Bee）放弃旧解时随机寻找新解。
        
        参数:
        - Q_min, Q_max: 由 get_search_bounds 提供的上下界。
        
        返回:
        - Q: 一个满足所有约束的合法订货方案 (N,)
        """
        max_attempts = 1000  # 防止死循环
        
        for _ in range(max_attempts):
            # 在 [Q_min, Q_max] 之间均匀随机采样生成 Q
            # 使用 np.random.randint，注意上限需要 +1
            Q_candidate = np.random.randint(Q_min, Q_max + 1)
            
            # 如果生成的方案满足预算和容量约束，则返回
            if self.is_feasible(Q_candidate):
                return Q_candidate
                
        # 如果尝试了多次都找不到合法解（例如预算卡得太死），
        # 则返回全 0 的方案（不订货总是合法的）。
        return np.zeros_like(Q_min)


# =====================================================================
# 测试与使用示例
# =====================================================================
if __name__ == "__main__":
    # 1. 从真实 dataset 目录读取数据
    # 使用 M5 Forecasting 数据集作为基础进行模拟
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    # 我们读取 validation 数据和 price 数据
    sales_path = os.path.join(dataset_dir, 'sales_train_validation.csv')
    prices_path = os.path.join(dataset_dir, 'sell_prices.csv')
    
    print(f"Loading real dataset from: {dataset_dir}")
    
    # 读取全量销售数据
    sales_df = pd.read_csv(sales_path)
    
    # 读取价格数据 (每个商品取其最新的价格)
    prices_df = pd.read_csv(prices_path)
    # 取每个 item_id, store_id 组合的最后一周价格
    latest_prices = prices_df.groupby(['item_id', 'store_id'])['sell_price'].last().reset_index()
    
    # 将价格信息合并到 sales_df 中
    df = pd.merge(sales_df, latest_prices, on=['item_id', 'store_id'], how='left')
    # 处理缺失价格，用平均价填充
    df['sell_price'] = df['sell_price'].fillna(df['sell_price'].mean())
    
    N = len(df)
    prices = df['sell_price'].values
    
    # 真实数据集中没有体积(volume)数据，我们根据价格随机生成一个合理的体积 (例如价格越贵体积可能越大，这里简化处理)
    np.random.seed(42)
    volumes = np.random.uniform(0.1, 5.0, N).round(2)
    
    # 使用倒数第二天的数据作为当前真实库存
    initial_inv = df['d_1912'].values
    
    # 使用最后一天的数据作为"完美神经网络"的预测需求 (y_pred)
    y_pred = df['d_1913'].values
    
    # 设定全局预算和容量限制 (针对全量数据 3万多个 SKU，需要调大预算和容量)
    BUDGET = 50000000  # 5千万预算
    CAPACITY = 2000000 # 2百万容量
    
    # 实例化搜索空间构造器
    search_space = InventorySearchSpace(
        num_skus=N, 
        prices=prices, 
        volumes=volumes, 
        initial_inventory=initial_inv, 
        budget=BUDGET, 
        capacity=CAPACITY
    )
    
    # 3. 构造搜索空间上下界
    Q_min, Q_max = search_space.get_search_bounds(y_pred, alpha=1.5, beta=3)
    
    # 4. 在空间中生成一个合法的初始订货方案 (供 ABCA 的蜜蜂使用)
    initial_Q = search_space.generate_random_solution(Q_min, Q_max)
    
    # 将结果整理为 DataFrame 并存入系统
    result_df = pd.DataFrame({
        'id': df['id'],
        'item_id': df['item_id'],
        'store_id': df['store_id'],
        'price': prices,
        'volume': volumes,
        'initial_inventory': initial_inv,
        'predicted_demand': y_pred,
        'Q_min': Q_min,
        'Q_max': Q_max,
        'feasible_Q_sample': initial_Q
    })
    
    output_path = os.path.join(dataset_dir, 'abc_test_data_results.csv')
    result_df.to_csv(output_path, index=False)
    
    print("\n--- 结果摘要 ---")
    print(f"提取的真实 SKU 数量:         {N}")
    print(f"生成的随机合法订货方案总采购成本: {np.sum(search_space.prices * initial_Q):.2f} (预算: {BUDGET})")
    print(f"生成的随机合法订货方案总占用容量: {np.sum(search_space.volumes * (search_space.inventory + initial_Q)):.2f} (上限: {CAPACITY})")
    print(f"\n✅ 详细的基于真实数据的每个 SKU 的上下界及生成方案已保存至: \n   {output_path}")
