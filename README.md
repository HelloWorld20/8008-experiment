Raw Data (M5)
   ↓
Feature Engineering + Parameter Generation
   ↓
[Step 1] Demand Clustering（风险分组）
   ↓
[Step 2] Neural Network（预测 demand）
   ↓
[Step 3] ABCA Solver（决策 Q_it）
   ↓
[Step 4] Inventory Environment（计算真实 cost）
   ↓
[Step 5] Surrogate Model（拟合 cost function）
   ↓
[Step 6] Backprop（更新 NN）

构造 Q_it 空间结果: abc_test_data_results.csv

ABCA计算后的结果：abca_final_results.csv
相对于输入，会多一列『ABCA_Best_Q』，代表每个 SKU 最优的 Q_it（进货量）

整体Pipeline的解释：

所以神经网络是在训练的每一次epoch，然后用当前状态的model推理一次，然后再走Step3到Step5，计算loss，然后再反向更新

### 完整的单次训练迭代 (One Training Step)

1. [Step 2] 前向推理 (Forward Pass) ：
   - 当前状态的神经网络（哪怕刚初始化，或者训练了一半）读取一个 Batch 的历史数据。
   - 网络输出这个 Batch 所有 SKU 的 预测需求 [ o bj ec tO bj ec t ] y ^ ​ 。

2. [Step 3] 求解决策 (Solve for Decision) ：
   - 拿着这个新鲜出炉的 [ o bj ec tO bj ec t ] y ^ ​ ，结合物理约束（预算、容量等），构造出搜索空间。
   - 扔给 abca.py （ABCA Solver）跑一圈。
   - ABCA 给出一个在这个 [ o bj ec tO bj ec t ] y ^ ​ 认知下“自认为最好”的订货方案 [ o bj ec tO bj ec t ] Q ∗ 。

3. [Step 4] 计算真实业务损失 (Evaluate True Cost) ：
   - 拿着这个决定好的订货单 [ o bj ec tO bj ec t ] Q ∗ ，去和**真实的未来需求（Ground Truth [ o bj ec tO bj ec t ] y ）**对账。
   - 在虚拟的“库存环境”中计算：卖出去了多少赚了多少钱？积压了多少亏了多少仓储费？缺货了多少扣了多少信誉分？
   - 得出一个真实的、反映业务痛点的 总成本 (True Cost) 。
   - （这就是你整个系统的终极 Loss 目标！）

4. [Step 5] 代理模型拟合伪梯度 (Surrogate Model for Gradients) ：
   - 难点来了 ：ABCA 是一个基于随机变异和贪心选择的黑盒算法，它 不可导 ！你没法用 PyTorch 的 .backward() 把 True Cost 的梯度直接穿过 ABCA 传回给神经网络。
   - 解法 ：我们用另一个简单的、可导的神经网络（代理模型，Surrogate Model）来学习这个映射关系： [ o bj ec tO bj ec t ] y ^ ​ → T r u e C os t 。
   - 代理模型看多了“什么样的预测会导致什么样的成本”之后，它就能告诉你： “如果你的预测 [ o bj ec tO bj ec t ] y ^ ​ 能稍微往上偏一点点，最终的 Cost 就会下降” 。这就产生了梯度（ [ o bj ec tO bj ec t ] ∂ y ^ ​ ∂ C os t ​ ）。

5. [Step 6] 反向传播更新网络 (Backward Pass & Update) ：
   - 把代理模型算出来的伪梯度传回给最初的神经网络（Step 2）。
   - 网络根据这个梯度，使用优化器（如 Adam）更新自己的权重参数。
   - 目的：让下一次（下一个 Epoch）输出的 [ o bj ec tO bj ec t ] y ^ ​ ，能引导 ABCA 做出一个 True Cost 更小的 [ o bj ec tO bj ec t ] Q ∗ 。
