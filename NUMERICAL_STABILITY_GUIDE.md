# 数值稳定性和梯度流指南

**用途**: 帮助用户诊断和解决训练过程中的数值问题
**版本**: 1.0
**最后更新**: 2024-11-22

---

## 1. 关键的数值问题和解决方案

### 1.1 NB Loss中的NAN问题

#### 问题来源

负二项分布损失包含以下项：
```
log p = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
      + θ(log θ - log(θ+μ))
      + x(log μ - log(θ+μ))
```

**风险点**：
1. 当x很大时，lgamma(x+θ)会溢出
2. 当θ或μ为0时，log会产生-inf
3. 当μ非常大时，log(θ+μ)可能溢出

#### 代码中的保护机制

```python
# losses.py 中的防护
log_theta_mu_eps = torch.log(theta + mean + eps)  # eps=1e-8
log_theta_eps = torch.log(theta + eps)
log_mu_eps = torch.log(mean + eps)

# 数值检查
if torch.any(torch.isnan(log_likelihood)) or torch.any(torch.isinf(log_likelihood)):
    # 替换异常值
    log_likelihood = torch.where(
        torch.isfinite(log_likelihood), 
        log_likelihood, 
        torch.zeros_like(log_likelihood)
    )
```

#### 用户排查清单

- [ ] 检查输入计数x的范围：不应超过1e6
- [ ] 检查μ的范围：应该接近数据的平均值
- [ ] 检查θ的范围：应该在[1e-4, 1e4]之间
- [ ] 查看控制台输出中的[警告]消息
- [ ] 如果持续出现NAN，尝试：
  ```python
  # 降低计数值
  X_normalized = X / X.max() * 1000  # 缩放到合理范围
  ```

---

### 1.2 KL散度中的爆炸问题

#### 问题来源

```
KL = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
```

**风险点**：
1. 当logvar > 20时，exp(logvar) → ∞
2. 当μ的绝对值很大时，μ² → ∞
3. 两项都可能导致数值溢出

#### 代码中的保护机制

```python
# losses.py 中的防护
logvar_safe = torch.clamp(logvar, max=20.0)  # 限制上界
mu_safe = torch.clamp(mu, min=-100.0, max=100.0)  # 限制范围

# 计算时使用安全版本
kl = -0.5 * torch.sum(
    1 + logvar_safe - mu_safe.pow(2) - logvar_safe.exp(), 
    dim=1
)
```

#### 用户排查清单

- [ ] 监控encoder输出的均值和方差
- [ ] 检查logvar是否经常超过15
- [ ] 如果KL损失异常大（>1000），说明编码器输出不稳定
- [ ] 尝试增加β（KL权重）的预热：
  ```python
  beta_annealed = min(epoch / 10, 1.0) * beta  # 逐步增加β
  ```

---

### 1.3 OT计算中的NAN问题

#### 问题来源

最优传输需要对潜变量Z进行标准化：

```python
Z_norm = (Z - mean) / std
```

**风险点**：
1. 当std=0时（某维度方差为0），出现除以0
2. 当Z包含极端异常值时，标准化失败
3. 当Z为空或太小时，统计计算不稳定

#### 代码中的保护机制

```python
# trainer.py 中的防护
source_std = torch.clamp(source_std, min=1e-8)  # 防止除以0
target_std = torch.clamp(target_std, min=1e-8)

Z_source_norm = (Z_source - source_mean) / source_std
Z_target_norm = (Z_target - target_mean) / target_std

# 检查并修复NAN
if torch.any(torch.isnan(Z_source_norm)):
    Z_source_norm = torch.where(
        torch.isnan(Z_source_norm), 
        torch.zeros_like(Z_source_norm), 
        Z_source_norm
    )
```

#### 用户排查清单

- [ ] 确保每个时间点有足够的细胞（>10）
- [ ] 检查Z的分布：不应有极端值
- [ ] 监控OT计算的日志输出
- [ ] 如果某个维度的std为0，说明此维度退化，尝试：
  ```python
  # 增加潜空间维度
  n_latent = 64  # 改为更大的值
  ```

---

## 2. 梯度流问题诊断

### 2.1 梯度消失（Vanishing Gradient）

#### 症状

- 损失不下降
- 模型参数没有更新
- 梯度范数非常小（< 1e-7）

#### 诊断代码

```python
# 在训练循环中添加
grad_norms, has_nan, has_zero = TwoStageTrainer.check_gradients(
    model, 
    phase_name=f"Phase 1, Epoch {epoch}"
)

# 如果有梯度为零的参数
if has_zero_grad:
    print("发现梯度为零的参数，可能表示：")
    print("1. 该参数未连接到损失")
    print("2. 使用了detach()断裂了梯度流")
    print("3. 该部分网络被冻结了")
```

#### 常见原因和修复

| 原因 | 症状 | 修复 |
|------|------|------|
| 初始化方差太小 | 所有梯度都很小 | 使用Xavier初始化（已实现） |
| 网络过深 | 低层梯度消失 | 减少网络深度或使用残差连接 |
| 学习率太小 | 梯度存在但损失不变 | 增加学习率到1e-3-1e-2 |
| 参数被冻结 | 特定层梯度为零 | 检查requires_grad设置 |
| 错误的detach() | 意外的梯度断裂 | 检查是否正确使用detach |

### 2.2 梯度爆炸（Exploding Gradient）

#### 症状

- 损失变成NAN
- 梯度范数非常大（> 1e2）
- 参数更新过大导致损失恶化

#### 诊断代码

```python
# 检查梯度范数
grad_norms, _, _ = TwoStageTrainer.check_gradients(model, "Phase 1")

# 计算平均梯度范数
avg_grad_norm = np.mean(list(grad_norms.values()))
print(f"平均梯度范数: {avg_grad_norm:.4e}")

# 如果太大，使用梯度裁剪
if avg_grad_norm > 10:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 常见原因和修复

| 原因 | 症状 | 修复 |
|------|------|------|
| 学习率太高 | 梯度爆炸或损失变NAN | 降低学习率到1e-4-1e-3 |
| 某层初始化不当 | 特定层梯度爆炸 | 检查该层的初始化 |
| Loss scale不合理 | 所有梯度都很大 | 使用梯度裁剪或调整权重 |
| 批大小太小 | 梯度震荡 | 增加批大小到≥32 |

#### 梯度裁剪示例

```python
# 在优化器step前添加
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0,  # 限制梯度范数
    norm_type=2.0
)
optimizer.step()
```

### 2.3 梯度断裂（Broken Gradient Flow）

#### 症状

- 某些参数的梯度始终为None
- 模型某部分不学习
- 梯度检查显示某参数梯度为零

#### 排查步骤

```python
# 1. 检查所有参数是否有梯度
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is None:
        print(f"[断裂] {name} 没有梯度")

# 2. 检查是否正确使用了detach()
# 应该这样：
z = encoder(x, c)  # 有梯度

# 不应该这样（除非刻意）：
z_detached = z.detach()  # 梯度断裂
features = gnn(z_detached)  # features无梯度

# 3. 检查是否有参数未连接到损失
# 使用钩子监控梯度流
def register_grad_hook(module):
    def hook(grad):
        if grad is None:
            print(f"[未连接] {module} 无梯度")
    return hook

for module in model.modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(register_grad_hook(module))
```

#### 对抗训练中的梯度管理

在Phase 1的对抗训练中要特别注意梯度流：

```python
# ✓ 正确的梯度管理
z = encoder(x, c)  # z有梯度

# 判别器的梯度更新
z_detached = z.detach()  # 正确：断裂梯度给encoder
disc_logits = discriminator(z_detached)
loss_disc = ...
loss_disc.backward()

# 编码器的对抗梯度更新
optimizer_enc.zero_grad()
z_new = encoder(x, c)  # 重新计算，有梯度
disc_logits_new = discriminator(z_new)  # 没有detach
loss_adv = ...  # 对抗损失
loss_adv.backward()
```

---

## 3. 性能监控清单

### 3.1 Phase 1 训练监控

在每个epoch后检查：

```python
# 损失检查
if recon_loss > 1000 or kl_loss > 100:
    print("[警告] 损失异常大")

# 梯度检查
grad_norms, has_nan, has_zero = TwoStageTrainer.check_gradients(model)
if has_nan or has_zero:
    print("[警告] 梯度异常")

# 参数检查
for name, param in model.named_parameters():
    if torch.any(torch.isnan(param.data)):
        print(f"[错误] {name} 包含NAN")
        break

# 输出统计
print(f"平均梯度范数: {np.mean(list(grad_norms.values())):.4e}")
```

### 3.2 Phase 2 训练监控

```python
# OT计算检查
if torch.any(torch.isnan(ot_matrix)):
    print("[错误] OT矩阵包含NAN")

# 流匹配损失检查
if fm_loss > 100 or torch.isnan(fm_loss):
    print("[警告] FM损失异常")

# 向量场检查
v_norms = torch.norm(v_pred, dim=1).mean().item()
target_norms = torch.norm(u_target, dim=1).mean().item()
print(f"预测速度范数: {v_norms:.4e}, 目标速度范数: {target_norms:.4e}")

# 如果比值太大表示规模不匹配
if v_norms > 10 * target_norms or target_norms > 10 * v_norms:
    print("[警告] 速度向量规模不匹配")
```

---

## 4. 调试技巧

### 4.1 启用PyTorch数值检查

```python
# 在脚本开始处添加
torch.autograd.set_detect_anomaly(True)  # 检测异常梯度
torch.set_printoptions(precision=6, edgeitems=10)  # 更好的打印格式
```

### 4.2 保存中间结果用于分析

```python
# 在训练中保存关键的中间值
debug_data = {
    'z_mean': mu_z.detach().cpu().numpy(),
    'z_logvar': logvar_z.detach().cpu().numpy(),
    'recon_mean': mean_recon.detach().cpu().numpy(),
    'recon_theta': theta_recon.detach().cpu().numpy(),
    'loss_recon': recon_loss.item(),
    'loss_kl': kl_loss.item(),
}

# 分析
import numpy as np
print(f"Z均值范围: [{debug_data['z_mean'].min()}, {debug_data['z_mean'].max()}]")
print(f"Z方差范围: [{np.exp(debug_data['z_logvar']).min()}, {np.exp(debug_data['z_logvar']).max()}]")
```

### 4.3 使用TensorBoard可视化

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/debug')

# 在训练中记录
writer.add_scalar('Loss/recon', recon_loss, global_step)
writer.add_scalar('Loss/kl', kl_loss, global_step)
writer.add_histogram('Gradients/encoder', 
                     torch.cat([p.grad.flatten() for p in encoder.parameters()]))
writer.add_scalar('Metrics/grad_norm', avg_grad_norm, global_step)

# 可视化
# tensorboard --logdir runs
```

---

## 5. 最佳实践总结

### 5.1 数据准备

- ✓ 计数数据应该在[0, 1e6]范围内
- ✓ 使用log1p(X)作为编码器输入
- ✓ 库大小应该在[1e2, 1e6]范围内
- ✗ 不要对计数数据进行z-score标准化

### 5.2 超参数设置

- ✓ 学习率：1e-3 - 1e-4（从1e-3开始）
- ✓ β（KL权重）：0.1 - 1.0
- ✓ λ_adv（对抗权重）：0.01 - 0.1
- ✓ 批大小：32 - 256（越大越稳定）
- ✗ 不要一下子设置很大的学习率

### 5.3 训练监控

- ✓ 每10个epoch检查一次梯度
- ✓ 保存最佳模型（按验证损失）
- ✓ 使用梯度裁剪防止爆炸
- ✓ 监控所有损失项的范围
- ✗ 不要忽视控制台的[警告]消息

### 5.4 故障排除流程

```
是否出现NAN？
├─ 是 → 检查数据范围 → 检查loss是否平衡 → 降低学习率
└─ 否 ↓

是否收敛很慢？
├─ 是 → 检查梯度是否消失 → 检查初始化 → 增加学习率
└─ 否 ↓

是否梯度爆炸？
├─ 是 → 使用梯度裁剪 → 降低学习率 → 增加批大小
└─ 否 ↓

正常训练！
```

---

## 参考文献

- PyTorch 数值稳定性: https://pytorch.org/docs/stable/notes/numerical_accuracy.html
- 梯度问题诊断: https://d2l.ai/chapter_optimization/optimization-algos.html
- 对抗训练稳定性: https://arxiv.org/abs/1606.01549

