import numpy as np
from typing import List
from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput, SolverOutput

class ABCASolver:
    """
    [Step 3] 人工蜂群运筹求解器 (B同学负责)
    基于预测的需求量 y_pred 和 全局约束，搜索最优的订单量 Q_it
    """
    def __init__(self, max_iter: int = 100, pop_size: int = 50, limit: int = 20):
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.limit = limit  # 侦查蜂重置阈值
        
    def _evaluate_cost(self, Q: np.ndarray, y_pred: np.ndarray, global_constraints: GlobalConstraints) -> float:
        """
        评估某个解的成本 (适应度函数，越小越好)
        包括：持有成本(h)、缺货成本(u)、固定订货成本(f) 以及 违反全局约束的惩罚项
        """
        # 业务成本计算 (向量化提速)
        holding_cost = np.sum(self._c_h * np.maximum(0, Q - y_pred))
        shortage_cost = np.sum(self._c_u * np.maximum(0, y_pred - Q))
        order_cost = np.sum(self._c_f * (Q > 0))
        total_cost = holding_cost + shortage_cost + order_cost
        
        # 资源消耗计算
        total_volume = np.sum(Q * self._v_i)
        total_budget = np.sum(Q * self._p_i)
        
        # 惩罚项 (罚函数法处理约束)
        penalty = 0.0
        if total_volume > global_constraints.V_max:
            penalty += 1000 * (total_volume - global_constraints.V_max)
        if total_budget > global_constraints.B_total:
            penalty += 1000 * (total_budget - global_constraints.B_total)
            
        return total_cost + penalty

    def solve(self, 
              predictor_out: PredictorOutput, 
              cost_params: List[SKUCostParams], 
              global_constraints: GlobalConstraints) -> SolverOutput:
        """
        求解最优订货量 Q_it
        
        参数:
            predictor_out (PredictorOutput): A 同学预测输出的数据类
            cost_params (List[SKUCostParams]): 包含各 SKU 成本参数的列表, 与 y_pred 对应
            global_constraints (GlobalConstraints): 全局约束，如 V_max 和 B_total
            
        返回:
            SolverOutput: 包含求解出的最优离散订货量的接口类
        """
        y_pred = predictor_out.y_pred
        n_items = len(y_pred)
        
        # 预提取参数为 Numpy 数组，大幅加速后续评估
        self._c_h = np.array([p.c_h for p in cost_params])
        self._c_u = np.array([p.c_u for p in cost_params])
        self._c_f = np.array([p.c_f for p in cost_params])
        self._v_i = np.array([p.v_i for p in cost_params])
        self._p_i = np.array([p.p_i for p in cost_params])
        
        # --- ABCA 核心逻辑 ---
        
        # 1. 初始化种群
        # 初始解以预测值为中心随机扰动
        population = np.zeros((self.pop_size, n_items), dtype=np.int32)
        fitness = np.zeros(self.pop_size)
        trials = np.zeros(self.pop_size, dtype=np.int32)
        
        for i in range(self.pop_size):
            # 围绕预测需求量生成初始解，保证非负
            population[i] = np.maximum(0, np.random.normal(loc=y_pred, scale=y_pred * 0.2)).astype(np.int32)
            fitness[i] = self._evaluate_cost(population[i], y_pred, global_constraints)
            
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_cost = fitness[best_idx]
        
        for iter_idx in range(self.max_iter):
            # 2. 雇佣蜂阶段 (Employed Bees)
            for i in range(self.pop_size):
                # 随机选择一个不等于当前蜜蜂的其他蜜蜂
                partner_idx = np.random.randint(0, self.pop_size - 1)
                if partner_idx >= i:
                    partner_idx += 1
                    
                # 随机选择一个维度进行扰动
                j = np.random.randint(0, n_items)
                phi = np.random.uniform(-1, 1)
                
                new_solution = population[i].copy()
                # 更新公式: v_{ij} = x_{ij} + phi * (x_{ij} - x_{kj})
                new_solution[j] = new_solution[j] + int(phi * (population[i][j] - population[partner_idx][j]))
                new_solution[j] = max(0, new_solution[j]) # 保证非负
                
                new_cost = self._evaluate_cost(new_solution, y_pred, global_constraints)
                
                # 贪心选择
                if new_cost < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_cost
                    trials[i] = 0
                else:
                    trials[i] += 1
                    
            # 3. 观察蜂阶段 (Onlooker Bees)
            # 计算选择概率 (成本越低，概率越大。这里转换为适应度: 1 / (1 + cost))
            fit_values = 1.0 / (1.0 + fitness)
            probs = fit_values / np.sum(fit_values)
            
            t = 0
            i = 0
            while t < self.pop_size:
                if np.random.rand() < probs[i]:
                    t += 1
                    partner_idx = np.random.randint(0, self.pop_size - 1)
                    if partner_idx >= i:
                        partner_idx += 1
                        
                    j = np.random.randint(0, n_items)
                    phi = np.random.uniform(-1, 1)
                    
                    new_solution = population[i].copy()
                    new_solution[j] = new_solution[j] + int(phi * (population[i][j] - population[partner_idx][j]))
                    new_solution[j] = max(0, new_solution[j])
                    
                    new_cost = self._evaluate_cost(new_solution, y_pred, global_constraints)
                    
                    if new_cost < fitness[i]:
                        population[i] = new_solution
                        fitness[i] = new_cost
                        trials[i] = 0
                    else:
                        trials[i] += 1
                
                i = (i + 1) % self.pop_size
                
            # 记录全局最优
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_cost:
                best_cost = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
                
            # 4. 侦查蜂阶段 (Scout Bees)
            for i in range(self.pop_size):
                if trials[i] >= self.limit:
                    # 重新随机生成一个解
                    population[i] = np.maximum(0, np.random.normal(loc=y_pred, scale=y_pred * 0.5)).astype(np.int32)
                    fitness[i] = self._evaluate_cost(population[i], y_pred, global_constraints)
                    trials[i] = 0

        Q_it = best_solution
        
        return SolverOutput(Q_it=Q_it)
