import numpy as np
import pandas as pd
import time
import os

class InventoryABC:
    def __init__(self, df, pop_size=20, max_iters=50, limit=20, budget=50000000, capacity=2000000):
        """
        初始化库存优化的人工蜂群算法 (ABCA)
        
        基于神经网络预测的需求量 y_pred，在 [Q_min, Q_max] 的搜索空间内，
        寻找满足 预算(budget) 和 仓储容量(capacity) 约束的最佳订货方案 Q。
        """
        self.df = df
        self.N = len(df)
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.limit = limit
        
        # 提取相关列信息
        self.prices = df['price'].values
        self.volumes = df['volume'].values
        self.inventory = df['initial_inventory'].values
        self.y_pred = df['predicted_demand'].values
        self.Q_min = df['Q_min'].values
        self.Q_max = df['Q_max'].values
        
        self.budget = budget
        self.capacity = capacity
        
        # 种群初始化
        self.population = []
        self.fitness = np.zeros(self.pop_size)
        self.trials = np.zeros(self.pop_size)
        
        print(f"Initializing population of size {self.pop_size}...")
        for i in range(self.pop_size):
            if i == 0 and 'feasible_Q_sample' in df.columns:
                # 使用上一步生成的已知可行解作为初始种群之一
                sol = df['feasible_Q_sample'].values.copy()
            else:
                sol = self.generate_feasible_solution()
            self.population.append(sol)
            self.fitness[i] = self.calculate_fitness(sol)
            
    def is_feasible(self, Q):
        """检查生成的解 Q 是否满足所有业务约束"""
        cost = np.sum(self.prices * Q)
        if cost > self.budget: return False
        vol = np.sum(self.volumes * (self.inventory + Q))
        if vol > self.capacity: return False
        if np.any(Q < 0): return False
        return True

    def generate_feasible_solution(self):
        """在搜索空间 [Q_min, Q_max] 内随机生成完全合法的初始解"""
        max_attempts = 1000
        for _ in range(max_attempts):
            # 注意: randint 的 high 是开区间，所以需要 +1
            Q = np.random.randint(self.Q_min, self.Q_max + 1)
            if self.is_feasible(Q):
                return Q
        return np.zeros_like(self.Q_min)

    def calculate_fitness(self, Q):
        """
        适应度函数 (Fitness Function)：
        在 Predict-then-Optimize 架构中，我们希望最终做出的决策尽可能符合网络的精准预测。
        惩罚值 (Cost) = sum(|Q - y_pred|)
        适应度 (Fitness) = 1 / (1 + Cost) ，Cost越小，适应度越高。
        """
        cost = np.sum(np.abs(Q - self.y_pred))
        return 1.0 / (1.0 + cost)

    def mutate_solution(self, i):
        """
        变异生成相邻候选解。
        蜜蜂通过参考另一个随机蜜源(解)的信息，来微调自己的位置。
        """
        k = i
        while k == i:
            k = np.random.randint(self.pop_size)
            
        V_i = self.population[i].copy()
        
        # 在标准的 ABCA 中，通常只改变一个维度。
        # 但因为我们有几万个 SKU，每次只改变一个维度收敛极慢。
        # 改进：我们每次随机变异 5% 的 SKU 维度 (多维度搜索)
        mutation_rate = 0.05
        mask = np.random.rand(self.N) < mutation_rate
        
        phi = np.random.uniform(-1, 1, size=self.N)
        # 变异公式: V_{i,d} = X_{i,d} + phi * (X_{i,d} - X_{k,d})
        V_i[mask] = V_i[mask] + (phi[mask] * (V_i[mask] - self.population[k][mask])).astype(int)
        
        # 边界截断，确保不会超出搜索空间定义的 Q_min 和 Q_max
        V_i = np.clip(V_i, self.Q_min, self.Q_max)
        
        return V_i

    def employed_bees_phase(self):
        """雇佣蜂阶段：每只雇佣蜂在其蜜源附近进行一次局部搜索"""
        for i in range(self.pop_size):
            V_i = self.mutate_solution(i)
            # 如果变异出来的解满足约束条件，才去评估它
            if self.is_feasible(V_i):
                fit_V = self.calculate_fitness(V_i)
                # 贪婪选择：如果新解比旧解好，则替换；否则试探次数+1
                if fit_V > self.fitness[i]:
                    self.population[i] = V_i
                    self.fitness[i] = fit_V
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
            else:
                self.trials[i] += 1

    def onlooker_bees_phase(self):
        """观察蜂阶段：根据蜜源的丰富程度(适应度)，按概率选择蜜源进行局部搜索"""
        total_fitness = np.sum(self.fitness)
        if total_fitness == 0:
            probs = np.ones(self.pop_size) / self.pop_size
        else:
            probs = self.fitness / total_fitness
            
        t = 0
        i = 0
        while t < self.pop_size:
            # 轮盘赌选择机制
            if np.random.rand() < probs[i]:
                V_i = self.mutate_solution(i)
                if self.is_feasible(V_i):
                    fit_V = self.calculate_fitness(V_i)
                    if fit_V > self.fitness[i]:
                        self.population[i] = V_i
                        self.fitness[i] = fit_V
                        self.trials[i] = 0
                    else:
                        self.trials[i] += 1
                else:
                    self.trials[i] += 1
                t += 1
            i = (i + 1) % self.pop_size

    def scout_bees_phase(self):
        """侦察蜂阶段：如果某个解连续 limit 次都没有改进，放弃它，随机产生一个全新的解"""
        for i in range(self.pop_size):
            if self.trials[i] > self.limit:
                self.population[i] = self.generate_feasible_solution()
                self.fitness[i] = self.calculate_fitness(self.population[i])
                self.trials[i] = 0

    def optimize(self):
        """执行 ABCA 优化主循环"""
        best_Q = None
        best_fitness = -1
        best_cost = float('inf')
        
        for it in range(self.max_iters):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()
            
            # 记录当前代的最优解
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > best_fitness:
                best_fitness = self.fitness[current_best_idx]
                best_Q = self.population[current_best_idx].copy()
                best_cost = 1.0 / best_fitness - 1.0
                
            print(f"Iteration {it+1:03d}/{self.max_iters} | Best Total MAE (Error): {best_cost:.2f}")
            
        return best_Q, best_fitness, best_cost

if __name__ == '__main__':
    # 1. 配置文件路径，读取搜索空间
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(base_dir), 'dataset', 'abc_test_data_results.csv')
    
    print(f"Loading search space dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 2. 参数设置
    POP_SIZE = 10        # 种群规模 (针对3万多 SKU，适当减小以保证执行效率)
    MAX_ITERS = 100      # 迭代次数 (增加到100次以获得更好的收敛效果)
    LIMIT = 10           # 允许失败的局部试探次数，超过则变为侦察蜂
    BUDGET = 50000000    # 预算约束 (与 make_Q.py 生成空间时一致)
    CAPACITY = 2000000   # 容量约束
    
    # 3. 运行算法
    start_time = time.time()
    abc = InventoryABC(
        df=df, 
        pop_size=POP_SIZE, 
        max_iters=MAX_ITERS, 
        limit=LIMIT, 
        budget=BUDGET, 
        capacity=CAPACITY
    )
    
    best_solution, best_fit, best_cost = abc.optimize()
    end_time = time.time()
    
    # 4. 打印并保存结果
    print("\n" + "="*50)
    print("--- ABCA Optimization Completed ---")
    print(f"Total Run Time: {end_time - start_time:.2f} seconds")
    print(f"Best Fitness:   {best_fit:.8f}")
    print(f"Best Cost:      {best_cost:.2f} (Total sum of |Q - y_pred|)")
    
    # 校验最终解是否满足物理约束
    final_cost = np.sum(abc.prices * best_solution)
    final_vol = np.sum(abc.volumes * (abc.inventory + best_solution))
    print(f"Final Cost Use: {final_cost:.2f} / Budget: {BUDGET}")
    print(f"Final Vol Use:  {final_vol:.2f} / Capacity: {CAPACITY}")
    print("="*50 + "\n")
    
    # 保存结果到 CSV
    df['ABCA_Best_Q'] = best_solution
    output_path = os.path.join(os.path.dirname(base_dir), 'dataset', 'abca_final_solution.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Saved optimized ABCA solution to:\n   {output_path}")
