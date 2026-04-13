import numpy as np
from typing import List
from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput, SolverOutput

class ABCASolver:
    """
    [Step 3] 人工蜂群运筹求解器 (B同学负责)
    基于预测的需求量 y_pred 和 全局约束，搜索最优的订单量 Q_it
    """
    def __init__(self, max_iter: int = 100, pop_size: int = 50):
        self.max_iter = max_iter
        self.pop_size = pop_size
        
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
        
        # TODO: B同学需要在这里实现 ABCA (人工蜂群) 的核心搜索逻辑
        # 1. 初始化 (s, S) 策略或直接初始化 Q_it 种群
        # 2. 雇佣蜂阶段 (局部邻域搜索)
        # 3. 观察蜂阶段 (根据 Cost 适应度分配资源，检查 global_constraints)
        # 4. 侦查蜂阶段 (重置陷入局部的解)
        
        # 暂时返回 dummy data 作为接口占位
        # 在这里我们做个简单的贪心策略：订货量 = 向上取整的预测需求量
        Q_it = np.ceil(y_pred).astype(np.int32)
        
        # 可以简单模拟一下全局约束的处理
        # 例如：总体积如果超出 global_constraints.V_max，按比例缩小订单
        # ...
        
        return SolverOutput(Q_it=Q_it)
