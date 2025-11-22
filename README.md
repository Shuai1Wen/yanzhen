# Causal-GenoFlow: 因果生成式单细胞动力学建模框架

## 1. 项目概述

**Causal-GenoFlow** 是一个基于因果推理和生成式建模的框架，用于建模单细胞基因表达的动力学过程。

### 核心创新

1. **NB-Guided Latent Flow Matching**：使用负二项分布VAE学习平滑的潜流形，然后在该流形上进行流匹配动力学建模
2. **GNN约束的向量场**：将先验的基因调控网络（GRN）直接融合到向量场计算中，确保生成的轨迹符合生物学约束
3. **对抗因果解耦**：通过对抗判别器将细胞的身份特征（如组织类型）与动力学特征分离
4. **反事实推断**：支持虚拟临床试验和"如果...会怎样"的反事实分析

### 科学意义

- **局部vs全身免疫**：解开免疫细胞在不同器官和疾病背景下为何表现不同的悖论
- **物种转化**：将小鼠数据训练的模型推广到人类临床数据，预测患者预后
- **离散→连续**：从稀疏的时间点快照填补连续的动力学过程

## 2. 环境设置

### 依赖安装

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import torch; import torch_geometric; import torchdiffeq; print('✓ 所有依赖已正确安装')"
```

## 3. 使用方法

### 3.1 基本工作流

```python
import torch
import numpy as np
from causal_genoflow import (
    CausalGenoFlow,
    TwoStageTrainer,
    ODEIntegrator,
    GRNPreprocessor,
    DataPreprocessor,
    create_simple_grn
)

# =======================
# Step 1: 准备数据
# =======================

# 生成模拟数据（实际应用请使用真实数据）
n_cells = 1000
n_genes = 500
n_latent = 32
n_cond = 2  # 两种条件（如健康vs疾病）

# 表达矩阵（计数）
X = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)

# 条件向量（假设独热编码）
C = np.eye(n_cond)[np.random.randint(0, n_cond, n_cells)].astype(np.float32)

# 时间标签（0-5表示不同的时间点）
time_labels = np.random.randint(0, 5, n_cells)

# 创建简单的GRN（实际应该使用DoRothEA等真实网络）
grn_edge_index = create_simple_grn(n_genes, density=0.1)

# =======================
# Step 2: 数据预处理
# =======================

# 计算库大小
lib_size = np.sum(X, axis=1)

# 转换为tensor
X_tensor = torch.from_numpy(X)
C_tensor = torch.from_numpy(C)
L_tensor = torch.from_numpy(lib_size).float()
time_labels_tensor = torch.from_numpy(time_labels).long()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =======================
# Step 3: 建立模型
# =======================

model = CausalGenoFlow(
    n_genes=n_genes,
    n_latent=n_latent,
    n_cond=n_cond,
    grn_edge_index=grn_edge_index.to(device),
    beta=1.0,
    lambda_adv=0.1,
    use_adversarial=True
).to(device)

# =======================
# Step 4: Phase 1 - VAE预热
# =======================

trainer = TwoStageTrainer(
    model=model,
    device=device,
    learning_rate=1e-3,
    beta=1.0,
    lambda_adv=0.1,
    lambda_fm=1.0
)

# 训练NB-VAE（冻结Flow）
Z_all = trainer.phase1_train(
    X=X_tensor.to(device),
    C=C_tensor.to(device),
    L=L_tensor.to(device),
    num_epochs=100,
    batch_size=32,
    verbose=True
)

# =======================
# Step 5: Phase 2 - 动力学学习
# =======================

trainer.phase2_train(
    X=X_tensor.to(device),
    C=C_tensor.to(device),
    L=L_tensor.to(device),
    Z_all=Z_all.to(device),
    time_labels=time_labels_tensor.to(device),
    num_epochs=200,
    batch_size=32,
    verbose=True
)

# =======================
# Step 6: 推理和轨迹生成
# =======================

# 创建ODE积分器
integrator = ODEIntegrator(model, solver='dopri5', device=device)

# 选择一个初始细胞状态
z_start = Z_all[0:1]

# 指定条件（例如疾病条件）
condition = C_tensor[0:1].to(device)

# 生成轨迹（时间从0到1）
traj_z, traj_mean, info = integrator.simulate_trajectory(
    z_initial=z_start,
    condition=condition,
    t_span=torch.linspace(0, 1, 100, device=device),
    library_size=L_tensor[0:1].to(device)
)

print(f"轨迹形状: {traj_z.shape}")  # (100, 1, 32) - 时间×批次×潜维度
print(f"基因表达轨迹: {traj_mean.shape}")  # (100, 1, 500) - 时间×批次×基因

# =======================
# Step 7: 反事实分析
# =======================

condition_healthy = C_tensor[0:1].to(device)  # 健康条件
condition_disease = torch.tensor([[0, 1]], dtype=torch.float32).to(device)  # 疾病条件

# 对比两种条件下的轨迹
traj_z_h, traj_mean_h, _, traj_mean_d = integrator.counterfactual_simulation(
    z_initial=z_start,
    condition_original=condition_healthy,
    condition_counterfactual=condition_disease
)

print("反事实分析完成，可用于虚拟临床试验")
```

### 3.2 参数详解

#### CausalGenoFlow模型

```python
model = CausalGenoFlow(
    n_genes=500,              # 基因数量
    n_latent=32,              # 潜空间维度
    n_cond=2,                 # 条件向量维度（独热编码）
    grn_edge_index=None,      # GRN边列表 (2, n_edges)
    beta=1.0,                 # KL散度权重
    lambda_adv=0.1,           # 对抗损失权重
    use_adversarial=True      # 是否启用对抗解耦
)
```

#### TwoStageTrainer

```python
trainer = TwoStageTrainer(
    model=model,
    device='cuda',            # 计算设备
    learning_rate=1e-3,       # 学习率
    beta=1.0,                 # VAE的KL权重
    lambda_adv=0.1,           # 对抗权重
    lambda_fm=1.0             # 流匹配权重
)

# Phase 1: NB-VAE预热
Z_all = trainer.phase1_train(
    X=X_tensor,
    C=C_tensor,
    L=L_tensor,
    num_epochs=100,           # 通常100-200个epoch
    batch_size=32,
    verbose=True
)

# Phase 2: 动力学学习
trainer.phase2_train(
    X=X_tensor,
    C=C_tensor,
    L=L_tensor,
    Z_all=Z_all,
    time_labels=time_labels_tensor,
    num_epochs=200,           # 通常200-500个epoch
    batch_size=32,
    verbose=True
)
```

#### ODEIntegrator

```python
integrator = ODEIntegrator(
    model=model,
    solver='dopri5',          # 'dopri5'、'adams'、'euler'
    device='cuda'
)

# 基本轨迹生成
traj_z, traj_mean, info = integrator.simulate_trajectory(
    z_initial=z_start,              # (batch_size, n_latent)
    condition=condition,            # (batch_size, n_cond)
    t_span=torch.linspace(0, 1, 100),  # 时间网格
    library_size=lib_sizes,         # (batch_size,)
    rtol=1e-5,                      # ODE相对容差
    atol=1e-6                       # ODE绝对容差
)

# 反事实模拟
traj_z_orig, traj_mean_orig, traj_z_cf, traj_mean_cf = integrator.counterfactual_simulation(
    z_initial=z_start,
    condition_original=condition_1,
    condition_counterfactual=condition_2
)

# 敏感性分析
results = integrator.sensitivity_analysis(
    z_initial=z_start,
    condition=condition,
    perturbation_dim=0,                    # 扰动第0维
    perturbation_values=np.linspace(-1, 1, 10),
    t_span=torch.linspace(0, 1, 100)
)
```

## 4. 实现细节

### 4.1 模型架构

```
Causal-GenoFlow 模型结构
│
├─ CausalEncoder
│  ├─ 输入: log1p(X), C
│  └─ 输出: μ_z, log σ²_z
│
├─ NBDecoder
│  ├─ 输入: z, C, l
│  ├─ 输出: μ = l·softmax(...), θ
│  └─ 分布: NB(μ, θ)
│
├─ TissueDiscriminator (可选)
│  ├─ 输入: z
│  └─ 输出: 条件预测logits
│
├─ CorrectedGNNVectorField
│  ├─ 基因嵌入: E_gene ∈ ℝ^(n_genes × d)
│  ├─ GAT层: 处理GRN拓扑
│  ├─ 交叉注意力: cells⟵genes
│  ├─ 时间嵌入: t⟶t_emb
│  ├─ 条件嵌入: c⟶c_emb
│  └─ 输出: v = dz/dt
│
└─ 损失函数
   ├─ NB Loss: -E[log NB(x|μ,θ)]
   ├─ KL Loss: D_KL(q||p)
   ├─ Flow Matching Loss: ||v_pred - (z1-z0)||²
   └─ Adversarial Loss: H(discriminator(z))
```

### 4.2 两阶段训练

**Phase 1: NB-VAE预热（冻结Flow）**
- 目标：学习光滑的潜流形，z ≈ N(0, I)
- 训练：Encoder + Decoder + 可选的Discriminator
- 损失：L = Recon + β·KL + λ·Adv
- 持续时间：通常100个epoch

**Phase 2: 动力学学习（冻结VAE）**
- 目标：训练向量场拟合Flow Matching损失
- 步骤：
  1. 冻结Encoder/Decoder，计算全量Z
  2. 为相邻时间点对计算OT耦合
  3. 构造训练对(z0, z1)及随机时间t
  4. 最小化||v_θ(ψ_t, t, c) - (z1-z0)||²
- 损失：L_FM = ||v_pred - u_target||²
- 持续时间：通常200-500个epoch

### 4.3 关键数学公式

#### 负二项分布（NB Loss）

$$p(x|\mu,\theta) = \frac{\Gamma(x+\theta)}{\Gamma(x+1)\Gamma(\theta)} \left(\frac{\theta}{\theta+\mu}\right)^\theta \left(\frac{\mu}{\theta+\mu}\right)^x$$

Log-space实现（数值稳定）：
```
log p = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
      + θ(log θ - log(θ+μ))
      + x(log μ - log(θ+μ))
```

#### 条件流匹配

线性插值路径：$\psi_t(z_0, z_1) = (1-t)z_0 + tz_1$

目标向量场：$u_t = z_1 - z_0$

损失函数：$L_{FM} = E_{t, (z_0,z_1)~\pi^*}[||v_\theta(\psi_t, t, c) - (z_1-z_0)||^2]$

#### GNN向量场（交叉注意力）

1. 基因特征：$H_{gene} = \text{GAT}(E_{gene}, A_{GRN})$
2. 交叉注意力：$\text{Attn}(z) = \text{softmax}(zH_{gene}^T)H_{gene}$
3. 最终向量：$v = \text{MLP}([z, \text{Attn}(z), t_{emb}, c_{emb}])$

## 5. 验证和测试

### 运行验证脚本

```bash
# 完整的端到端验证
python verify_implementation.py

# 或逐步验证各组件
python -c "from causal_genoflow import *; print('✓ 模块导入成功')"
```

### 单元测试

```python
import torch
from causal_genoflow import NBLoss

# 测试NB Loss的数值稳定性
loss_fn = NBLoss()
x = torch.randn(32, 100).abs()  # 计数数据
mean = torch.rand(32, 100) * 10
theta = torch.ones(100) * 5.0

loss = loss_fn(x, mean, theta)
print(f"NB Loss: {loss.item():.4f}")  # 应该是一个合理的标量
```

## 6. 高级功能

### 6.1 自定义GRN

```python
from causal_genoflow import GRNPreprocessor

preprocessor = GRNPreprocessor()

# 加载真实GRN（例如DoRothEA）
# grn_edges 应该是 (n_edges, 2) 的基因名列表
valid_genes, edge_index = preprocessor.align_genes(
    scRNA_genes=gene_names_from_data,
    GRN_edges=grn_edges_from_database,
    GRN_gene_names=all_genes_in_grn
)

# 子集表达数据
X_subset = preprocessor.subset_expression(X, gene_names_from_data)
```

### 6.2 条件编码

```python
from causal_genoflow import ConditionDataLoader

# 支持多种条件格式
conditions_onehot = ConditionDataLoader.one_hot_encode(
    conditions=np.array([0, 1, 0, 1, ...]),  # 条件索引
    n_conditions=2
)
```

### 6.3 模型保存和加载

```python
# 保存检查点
trainer.save_checkpoint('model_checkpoint.pt')

# 加载检查点
trainer.load_checkpoint('model_checkpoint.pt')

# 推理模式
model.eval()
with torch.no_grad():
    z, _, _ = model.encode(X, C)
    mean, theta = model.decode(z, C, library_size)
```

## 7. 常见问题

### Q: 如何处理缺失值？
A: 在预处理阶段将缺失值设为0（视为未检测）。NB分布能够妥善处理零计数。

### Q: 是否需要数据标准化？
A: 在编码器输入前使用log1p归一化。对于OT计算，需要对潜变量Z进行Z-score标准化（已在trainer中自动处理）。

### Q: GRN有多稀疏/密集的上限？
A: 模型支持任何密度。对于非常稀疏的GRN，某些基因可能未被约束；对于密集网络，交叉注意力会自动学习关键连接。

### Q: Phase 1需要多少个epoch？
A: 通常100-200个。监测验证集的重建损失和KL散度，当KL损失趋于稳定时可停止。

### Q: 如何调整λ_adv？
A: 从0.01开始。如果z仍包含条件信息，增加λ_adv；如果损失震荡，降低λ_adv。

## 8. 推荐配置

### 小规模数据（<10k cells, <2k genes）
```
n_latent = 16
batch_size = 64
learning_rate = 1e-3
Phase 1 epochs = 100
Phase 2 epochs = 200
```

### 中等规模（10k-100k cells）
```
n_latent = 32
batch_size = 128
learning_rate = 5e-4
Phase 1 epochs = 100-150
Phase 2 epochs = 200-300
```

### 大规模（>100k cells）
```
n_latent = 64
batch_size = 256
learning_rate = 1e-4
Phase 1 epochs = 50-100
Phase 2 epochs = 100-200
```

## 9. 引用信息

**请在使用本框架的论文中引用以下文献：**

```bibtex
@article{causal-genoflow,
  title={Causal-GenoFlow: 因果生成式单细胞动力学建模},
  year={2024}
}

@article{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Y. et al.},
  journal={arXiv preprint arXiv:2210.02747},
  year={2023}
}

@article{scvi,
  title={Probabilistic modeling of single-cell transcriptomics data with scVI},
  author={Lopez, R. et al.},
  journal={Nature Methods},
  year={2018}
}
```

## 10. 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 11. 贡献指南

欢迎提交Issue和Pull Request！请遵循以下指南：
- 新功能应包含相应的测试
- 所有中文注释应使用简体中文
- 遵循项目的代码风格（见.style.txt）

---

**最后更新**：2024年11月
**维护者**：AI Research Team
