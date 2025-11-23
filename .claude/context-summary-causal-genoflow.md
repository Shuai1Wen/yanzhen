# Causal-GenoFlow 项目上下文摘要

生成时间：2024-11-22

## 1. 项目概述

Causal-GenoFlow 是一个因果-生成式深度学习框架，用于建模单细胞基因表达动力学。它综合了以下关键成分：

- **概率模型**：负二项分布VAE（NB-VAE）处理离散计数数据
- **动力学建模**：条件流匹配（CFM）在潜空间学习向量场
- **物理约束**：GNN+交叉注意力融合GRN拓扑约束
- **因果解耦**：对抗判别器分离身份与动力学

## 2. 核心数学模型

### 2.1 概率生成模型
- **似然函数**：条件负二项分布 $p_\psi(x|z,l,c) = \prod_g NB(x_g; \mu_g, \theta_g)$
- **编码器**：$q_\phi(z|x,c) = \mathcal{N}(z; \mu_\phi(x,c), \sigma_\phi^2(x,c))$
- **解码器**：
  - 缩放因子：$\rho = Softmax(f_\psi^\rho(z,c))$
  - 均值：$\mu_g = l \cdot \rho_g$
  - 离散度：$\theta_g = exp(f_\psi^\theta(z))$

### 2.2 流匹配动力学
- **插值路径**：$\psi_t(z_0, z_1) = (1-t)z_0 + tz_1$
- **目标向量场**：$u_t(z|z_0,z_1) = z_1 - z_0$
- **损失函数**：$\mathcal{L}_{FM} = E_{t,z_0,z_1}[||v_\theta(\psi_t, t, c) - (z_1-z_0)||^2]$

### 2.3 因果解耦
- 对抗判别器从Z反推条件C
- 生成器通过对抗损失最大化Z的条件独立性

## 3. 架构组件清单

### 3.1 核心模块
1. **NBLoss** - 数值稳定的负二项损失（Log-space实现）
2. **CausalEncoder** - 编码器，处理离散数据
3. **NBDecoder** - NB解码器，生成μ和θ
4. **TissueDiscriminator** - 对抗判别器
5. **CorrectedGNNVectorField** - GNN向量场，含交叉注意力
6. **CausalGenoFlow** - 主模型框架
7. **ODEIntegrator** - 推理ODE求解器

### 3.2 关键设计
- GNN向量场使用**可学习基因嵌入 + 交叉注意力**解决Batch维度不匹配
- 两阶段训练：Phase 1冻结Flow训练VAE，Phase 2冻结VAE训练Flow
- 对抗训练：交替更新判别器和编码器

## 4. 实现约束

### 4.1 数值稳定性
- NB Loss必须在Log-space计算（使用lgamma）
- θ必须被限制在[1e-4, 1e4]范围内
- μ预测时使用log1p(x)做输入归一化

### 4.2 图数据处理
- GRN与scRNA基因的交集
- Edge index需要重新索引到[0, n_valid_genes)
- 基因嵌入维度必须与GNN输入匹配

### 4.3 两阶段训练
- Phase 1：100个Epoch训练NB-VAE，冻结Flow
- Phase 2：计算全量Latent Z，执行OT耦合，训练Vector Field

## 5. 代码结构规划

```
causal_genoflow/
├── __init__.py
├── modules.py          # 所有神经网络模块
├── losses.py           # 损失函数
├── data.py             # 数据加载和预处理
├── trainer.py          # 两阶段训练器
├── inference.py        # ODE推理和轨迹生成
└── utils.py            # 辅助函数

项目根目录：
├── main.py             # 完整训练管道
├── requirements.txt    # 依赖列表
└── README.md           # 使用说明
```

## 6. 关键依赖

- **PyTorch** >= 1.9.0（带cuda支持）
- **PyTorch Geometric** >= 2.0（GNN和图操作）
- **torchdiffeq** 最新版（ODE求解）
- **NumPy**, **SciPy**（数值计算）
- **POT** 或 **ot**（最优传输）
- **scikit-learn**（预处理，特别是Z-score标准化）

## 7. 验证策略

### 7.1 单元测试
- 每个模块（损失、编码器、解码器、判别器、向量场）
- 边界条件：极小batch、zero counts、极端θ值

### 7.2 集成测试
- 完整的前向传播（VAE）
- 完整的流匹配前向（Vector Field）
- ODE求解和轨迹生成

### 7.3 数值验证
- NB损失与参考实现对比
- CFM与直接ODE积分结果的一致性

## 8. 已识别的关键细节（从details2.txt）

1. **缺口1**：TissueDiscriminator + 对抗训练逻辑（已规划）
2. **缺口2**：Vector Field需接收condition参数（已规划）
3. **缺口3**：GRN与scRNA基因交集处理（已规划）
4. **缺口4**：torchdiffeq ODE求解（已规划）

所有4个缺口都会在代码实现中严格补全。

## 9. 项目约定

- **语言**：所有代码注释和文档必须使用简体中文
- **命名**：类和函数使用英文snake_case/CamelCase
- **文档**：每个模块必须有docstring说明其与model.txt的对应关系
- **验证**：必须提供可复现的验证脚本

## 10. 风险点和处理策略

| 风险 | 处理策略 |
|------|---------|
| GNN Batch不匹配 | 使用交叉注意力而非直接卷积 |
| NB Loss数值溢出 | Log-space计算，θ范围限制 |
| VAE Latent不收敛 | 调整β权重，两阶段训练 |
| OT计算复杂度 | 仅计算相邻时间点对 |
| 对抗训练不稳定 | 判别器早停，对抗权重衰减 |

