import torch
import numpy as np
from torch.optim import Adam
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from env.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel, SurrogateAutogradFunction
from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput, SolverOutput, EnvironmentOutput

def train_predict_and_optimize(
    dataloader, 
    predictor: DemandPredictor, 
    solver: ABCASolver, 
    env: InventoryEnvironment, 
    surrogate: SurrogateModel,
    epochs: int = 10
):
    """
    [Step 6] 端到端 Predict-and-Optimize 训练循环 (C同学负责)
    负责串联 A(Predictor)、B(Solver+Env)、C(Surrogate) 的大动脉
    """
    optimizer = Adam(predictor.parameters(), lr=1e-3)
    
    # 历史缓冲池，用于训练 Surrogate Model
    history_y_pred = []
    history_true_cost = []
    
    for epoch in range(epochs):
        for batch_idx, (features, true_demand, cost_params_dict) in enumerate(dataloader):
            
            # ==========================================================
            # [Step 2] 前向预测 (A)
            # ==========================================================
            optimizer.zero_grad()
            y_pred_tensor = predictor(features)  # 预测未来的需求 y_it
            
            # 包装为接口类
            y_pred_np = y_pred_tensor.detach().cpu().numpy().flatten()
            predictor_out = PredictorOutput(y_pred=y_pred_np)
            true_demand_np = true_demand.numpy().flatten()
            
            # 模拟：将 DataLoader 中取出的字典转为强类型的 DataClass
            # 实际场景中 DataLoader 可能直接返回字典，需要在此处转换
            cost_params_list = []
            for i in range(len(y_pred_np)):
                cp = SKUCostParams(
                    item_id=cost_params_dict.get('item_id', [f"item_{i}"])[i] if isinstance(cost_params_dict.get('item_id'), list) else f"item_{i}",
                    store_id=cost_params_dict.get('store_id', [f"store_{i}"])[i] if isinstance(cost_params_dict.get('store_id'), list) else f"store_{i}",
                    c_h=float(cost_params_dict['c_h'].item() if isinstance(cost_params_dict['c_h'], torch.Tensor) else cost_params_dict['c_h']),
                    c_u=float(cost_params_dict['c_u'].item() if isinstance(cost_params_dict['c_u'], torch.Tensor) else cost_params_dict['c_u']),
                    c_f=float(cost_params_dict['c_f'].item() if isinstance(cost_params_dict['c_f'], torch.Tensor) else cost_params_dict['c_f']),
                    v_i=float(cost_params_dict['v_i'].item() if isinstance(cost_params_dict['v_i'], torch.Tensor) else cost_params_dict['v_i']),
                    p_i=float(cost_params_dict['p_i'].item() if isinstance(cost_params_dict['p_i'], torch.Tensor) else cost_params_dict['p_i'])
                )
                cost_params_list.append(cp)
            
            # ==========================================================
            # [Step 3] 求解决策 (B)
            # ==========================================================
            # 定义全局约束
            global_constraints = GlobalConstraints(V_max=10000.0, B_total=50000.0)
            
            # ABCA 求解器给出最优订货量 (返回 SolverOutput 接口类)
            solver_out = solver.solve(predictor_out, cost_params_list, global_constraints)
            
            # ==========================================================
            # [Step 4] 真实成本评估 (B)
            # ==========================================================
            # 在真实的环境中，评估该决策 Q_it 会产生多少实际的 Cost
            env_out = env.evaluate_cost(solver_out, true_demand_np, cost_params_list)
            true_costs_np = env_out.true_costs

            
            # 收集数据以训练代理模型
            history_y_pred.extend(y_pred_np)
            history_true_cost.extend(true_costs_np)
            
            # ==========================================================
            # [Step 5 & 6] 代理模型拟合与反向传播 (C)
            # ==========================================================
            # 每隔一定的 batch 训练/更新一次 Surrogate Model (Burn-in 阶段)
            if len(history_y_pred) >= 128:
                surrogate.train_surrogate(np.array(history_y_pred), np.array(history_true_cost))
                
                # 保留滑动窗口，丢弃太旧的探索数据
                history_y_pred = history_y_pred[-1000:]
                history_true_cost = history_true_cost[-1000:]
                
            if surrogate.is_trained:
                # 关键步骤：使用自定义的 Autograd Function 桥接 PyTorch 和 LightGBM
                # 这样一来，Cost 就能被求导并回传给 y_pred_tensor
                cost_tensor = SurrogateAutogradFunction.apply(y_pred_tensor, surrogate)
                
                # 我们的终极目标是最小化业务成本，而不是 MSE
                loss = cost_tensor.mean()
                
                # 反向传播 (更新 A同学 的神经网络)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Surrogate Loss (True Cost): {loss.item():.2f}")
            else:
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Collecting data to warmup Surrogate...")
