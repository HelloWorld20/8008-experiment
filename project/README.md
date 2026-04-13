# Predict-then-Optimize: End-to-End Inventory Management

这是一个基于端到端决策学习（Predict-and-Optimize）框架的库存管理项目。
本项目旨在将传统的预测（Demand Forecasting）和运筹优化（Operations Research）紧密结合，使得神经网络在训练时能够直接感知下游的真实业务成本，而不是仅仅最小化统计误差（如 MSE）。

---

## 🚀 项目架构与模块划分

项目整体结构按照功能高度解耦，方便三人并行开发：

```text
project/
│
├── data/             # 数据处理与特征工程
│   └── dataset.py    # [Step 1] 数据加载器 (M5Dataset)
│
├── model/            # 预测模型
│   └── lstm.py       # [Step 2] 需求预测网络 (DemandPredictor)
│
├── solver/           # 运筹优化求解器
│   └── abca.py       # [Step 3] 人工蜂群求解器 (ABCASolver)
│
├── env/              # 业务环境
│   └── inventory.py  # [Step 4] 真实库存成本计算 (InventoryEnvironment)
│
├── surrogate/        # 代理模型与伪梯度
│   └── model.py      # [Step 5] 代理模型拟合与 PyTorch Autograd (SurrogateModel)
│
├── train/            # 训练主循环
│   └── loop.py       # [Step 6] 端到端反向传播 (train_predict_and_optimize)
│
└── main.py           # 全局入口文件
```

---

## 💡 接口约定 (Data Contracts)

为确保三人并行开发不冲突，项目中通过严格的 Dataclass 进行接口约束。具体定义参见 `interfaces.py`：

### 1. `SKUCostParams` (单品成本与业务参数)

由 **A 同学** 在 `data/dataset.py` 中解析 M5 数据生成，提供给 B 同学的黑盒使用。

- `item_id (str)`: 商品唯一标识，例如 `"HOBBIES_1_001"`。
- `store_id (str)`: 门店标识，例如 `"CA_1"`。
- `c_h (float)`: 单位持有成本。例如：基于采购价 `p_i` 乘以 20% 年化率分摊到周。
- `c_u (float)`: 单位缺货成本。即销售损失，通常为 `sell_price - p_i` (毛利)。
- `c_f (float)`: 固定订货成本。只要发生补货行为 ($Q>0$) 就固定扣除的费用。
- `v_i (float)`: 单位商品体积。由类别(cat_id)推算，用于评估仓储消耗。
- `p_i (float)`: 单位采购价格。用于评估预算消耗。

### 2. `GlobalConstraints` (全局运筹边界)

由 **C 同学** 在 `train/loop.py` 或 `main.py` 统筹注入，控制 B 同学的联合决策上限。

- `V_max (float)`: 仓库最大可用容积 (Storage Capacity)。
- `B_total (float)`: 单次补货的全局总预算上限 (Budget Constraint)。

### 3. 数据流向与形状约束

- **A -> B (预测 -> 求解)**:
  - 接口类: `PredictorOutput`
  - 核心字段: `y_pred` (预测需求量)
  - 类型与形状: `np.ndarray` (由 PyTorch Tensor `.detach().cpu().numpy()` 转换)，Shape: `(batch_size, )`。
- **B -> Env (运筹 -> 业务环境)**:
  - 接口类: `SolverOutput`
  - 核心字段: `Q_it` (最优订货量)
  - 类型与形状: `np.ndarray` (离散整数 `int32`)，Shape: `(batch_size, )`。
- **B -> C (业务环境 -> 训练代理)**:
  - 接口类: `EnvironmentOutput`
  - 核心字段: `true_costs` (由于订货决策和真实需求产生的非对称总成本)
  - 类型与形状: `np.ndarray` (浮点数 `float32`)，Shape: `(batch_size, )`。

---

## � 三人开发分工建议

本架构严格遵循 Predict-and-Optimize 的技术边界，三位开发者各司其职，无缝对接。

### 👨‍💻 开发者 A: 预测模型工程师 (Predictive Model Engineer)

- **负责模块**: `data/` 和 `model/`
- **核心任务**: 处理特征工程并构建 PyTorch 数据集；实现时间序列预测网络，完成前向推理。

### 👨‍💻 开发者 B: 运筹与环境工程师 (OR Solver & Environment Engineer)

- **负责模块**: `solver/` 和 `env/`
- **核心任务**: 使用 NumPy 实现人工蜂群求解器 (ABCA)；实现真实的库存成本 (Cost Function) 计算环境。

### 👨‍💻 开发者 C: 代理模型与集成工程师 (Surrogate & Integration Engineer)

- **负责模块**: `surrogate/`, `train/` 和 `main.py`
- **核心任务**: 训练代理模型并编写自定义 Autograd 实现伪梯度；串联各个模块，实现端到端的闭环训练与验证。

---

## 🔄 核心 Pipeline (数据流向)

整个端到端的训练流程在 `train/loop.py` 中循环执行，其数据流向如下：

1.  **[Step 1] Data Loading**: `DataLoader` 吐出一个 Batch 的数据 `(features, true_demand, cost_params)`。
2.  **[Step 2] Forward Pass (A)**: `DemandPredictor` 接收 `features`，输出预测需求量 `y_pred` (Tensor)。
3.  **[Step 3] Heuristic Solving (B)**: 将 `y_pred` 转换为 NumPy，送入 `ABCASolver`。求解器在满足全局约束的前提下，输出离散的最优订货量 `Q_it`。
4.  **[Step 4] Environment Evaluation (B)**: 将 `Q_it` 和 `true_demand` 送入 `InventoryEnvironment`，计算出本次决策的真实总成本 `true_costs`。
5.  **[Step 5] Surrogate Fitting (C)**: 收集历史的 `(y_pred, true_costs)` 对，定期训练/更新 LightGBM 代理模型。
6.  **[Step 6] Backpropagation (C -> A)**: 当代理模型 Ready 后，通过自定义的 `SurrogateAutogradFunction`，将 `true_costs` 转化为 Tensor 并计算伪梯度。执行 `loss.backward()`，更新 A 的预测网络权重。
