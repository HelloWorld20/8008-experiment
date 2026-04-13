import sys
import os
import numpy as np

# 将 project 目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces import SKUCostParams, SolverOutput
from environment.inventory import InventoryEnvironment

def test_environment():
    """
    [B 同学单测 - Env]
    单独测试库存环境(Env)的成本计算
    """
    print("=== Testing InventoryEnvironment (B) ===")
    batch_size = 3
    
    # 1. 模拟 B 同学求解器得出的决策变量 Q_it
    dummy_Q_it = np.array([6, 11, 3], dtype=np.int32)
    solver_out = SolverOutput(Q_it=dummy_Q_it)
    print(f"Mock Solver Output (Q_it): {dummy_Q_it}")
    
    # 2. 模拟真实需求 D_it (例如发生了需求突增)
    true_demand = np.array([8.0, 9.0, 0.0], dtype=np.float32)
    print(f"Mock True Demand (D_it): {true_demand}")
    
    # 3. 模拟业务参数 (cost_params)
    cost_params_list = [
        SKUCostParams(item_id="item_0", store_id="store_A", c_h=1.0, c_u=5.0, c_f=10.0, v_i=2.0, p_i=20.0),
        SKUCostParams(item_id="item_1", store_id="store_A", c_h=0.5, c_u=2.0, c_f=5.0,  v_i=1.0, p_i=10.0),
        SKUCostParams(item_id="item_2", store_id="store_A", c_h=2.0, c_u=10.0,c_f=20.0, v_i=5.0, p_i=50.0),
    ]
    
    # 4. 初始化并测试环境 (Env)
    env = InventoryEnvironment()
    
    try:
        env_out = env.evaluate_cost(solver_out, true_demand, cost_params_list)
        print(f"Environment Evaluated. True Costs: {env_out.true_costs}")
        print(f"Env Output Type: {type(env_out.true_costs)} (Expected: numpy.ndarray float32)")
        
        assert isinstance(env_out.true_costs, np.ndarray), "Env must return a numpy array!"
        assert env_out.true_costs.shape == (batch_size, ), f"true_costs shape must be ({batch_size}, )!"
        
    except Exception as e:
        print(f"Environment Evaluation Failed: {e}")

if __name__ == "__main__":
    test_environment()
