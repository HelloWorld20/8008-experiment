import sys
import os
import numpy as np

# 将 project 目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput
from solver.abca import ABCASolver

def test_solver():
    """
    [B 同学单测 - Solver]
    单独测试运筹求解器(Solver)的逻辑
    """
    print("=== Testing ABCASolver (B) ===")
    batch_size = 3
    
    # 1. 模拟 A 同学传来的预测输出 (PredictorOutput)
    dummy_y_pred = np.array([5.2, 10.8, 2.5], dtype=np.float32)
    predictor_out = PredictorOutput(y_pred=dummy_y_pred)
    print(f"Mock Predictor Output (y_pred): {dummy_y_pred}")
    
    # 2. 模拟业务参数 (cost_params) 和全局约束
    cost_params_list = [
        SKUCostParams(item_id="item_0", store_id="store_A", c_h=1.0, c_u=5.0, c_f=10.0, v_i=2.0, p_i=20.0),
        SKUCostParams(item_id="item_1", store_id="store_A", c_h=0.5, c_u=2.0, c_f=5.0,  v_i=1.0, p_i=10.0),
        SKUCostParams(item_id="item_2", store_id="store_A", c_h=2.0, c_u=10.0,c_f=20.0, v_i=5.0, p_i=50.0),
    ]
    global_constraints = GlobalConstraints(V_max=100.0, B_total=1000.0)
    
    # 3. 初始化并测试求解器 (ABCA)
    solver = ABCASolver()
    try:
        solver_out = solver.solve(predictor_out, cost_params_list, global_constraints)
        print(f"Solver Executed. Optimal Q_it: {solver_out.Q_it}")
        print(f"Solver Output Type: {type(solver_out.Q_it)} (Expected: numpy.ndarray int32)")
        
        assert isinstance(solver_out.Q_it, np.ndarray), "Solver must return a numpy array!"
        assert solver_out.Q_it.dtype == np.int32, "Decision variables (Q_it) must be integers!"
        assert solver_out.Q_it.shape == (batch_size, ), f"Q_it shape must be ({batch_size}, )!"
        
    except Exception as e:
        print(f"Solver Failed: {e}")

if __name__ == "__main__":
    test_solver()
