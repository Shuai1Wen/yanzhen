"""
Causal-GenoFlow 完整工作流示例

这个脚本展示了从数据加载到轨迹生成的完整工作流。
使用模拟数据，可直接运行。
"""

import torch
import numpy as np
import sys

# 确保可以导入本地包
sys.path.insert(0, '/home/engine/project')

from causal_genoflow import (
    CausalGenoFlow,
    TwoStageTrainer,
    ODEIntegrator,
    create_simple_grn,
    DataPreprocessor
)


def print_header(text):
    """打印格式化的标题"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def main():
    """主程序"""
    
    print_header("Causal-GenoFlow 完整工作流示例")
    
    # ================================================================
    # Step 1: 准备合成数据
    # ================================================================
    print_header("Step 1: 准备数据")
    
    # 模拟参数
    n_cells = 500
    n_genes = 200
    n_latent = 16
    n_cond = 2  # 两种条件（例如：健康vs疾病）
    n_timepoints = 3  # 三个时间点
    
    print(f"生成合成数据...")
    print(f"  - 细胞数: {n_cells}")
    print(f"  - 基因数: {n_genes}")
    print(f"  - 潜维数: {n_latent}")
    print(f"  - 条件数: {n_cond}")
    print(f"  - 时间点: {n_timepoints}")
    
    # 生成表达数据（泊松分布计数）
    # 注意：实际应用应使用真实的scRNA-seq数据
    X = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    
    # 生成条件向量（独热编码）
    condition_indices = np.random.randint(0, n_cond, n_cells)
    C = np.eye(n_cond)[condition_indices].astype(np.float32)
    
    # 生成时间标签
    time_labels = np.random.randint(0, n_timepoints, n_cells)
    
    print(f"✓ 数据生成完成")
    print(f"  - X形状: {X.shape}")
    print(f"  - C形状: {C.shape}")
    print(f"  - 时间点分布: {np.bincount(time_labels)}")
    
    # ================================================================
    # Step 2: 数据预处理
    # ================================================================
    print_header("Step 2: 数据预处理")
    
    preprocessor = DataPreprocessor()
    
    # 计算库大小
    library_size = preprocessor.compute_library_sizes(X)
    print(f"✓ 计算库大小")
    print(f"  - 库大小范围: [{library_size.min():.0f}, {library_size.max():.0f}]")
    
    # 转换为PyTorch张量
    X_tensor = torch.from_numpy(X)
    C_tensor = torch.from_numpy(C)
    L_tensor = torch.from_numpy(library_size).float()
    time_labels_tensor = torch.from_numpy(time_labels).long()
    
    print(f"✓ 转换为张量")
    
    # 选择计算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # ================================================================
    # Step 3: 创建GRN
    # ================================================================
    print_header("Step 3: 创建基因调控网络 (GRN)")
    
    # 创建简单的GRN（实际应用应使用DoRothEA等）
    grn_edge_index = create_simple_grn(n_genes, density=0.05)
    print(f"✓ GRN创建完成")
    print(f"  - 基因数: {n_genes}")
    print(f"  - 边数: {grn_edge_index.shape[1]}")
    
    # ================================================================
    # Step 4: 构建模型
    # ================================================================
    print_header("Step 4: 构建Causal-GenoFlow模型")
    
    model = CausalGenoFlow(
        n_genes=n_genes,
        n_latent=n_latent,
        n_cond=n_cond,
        grn_edge_index=grn_edge_index.to(device),
        beta=1.0,           # KL散度权重
        lambda_adv=0.1,     # 对抗损失权重
        use_adversarial=True # 启用对抗解耦
    )
    
    model = model.to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型构建完成")
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # ================================================================
    # Step 5: Phase 1 - NB-VAE预热
    # ================================================================
    print_header("Step 5: Phase 1 - NB-VAE预热")
    
    trainer = TwoStageTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        beta=1.0,
        lambda_adv=0.1
    )
    
    print("开始Phase 1训练 (冻结Flow，训练VAE)...")
    print("  配置:")
    print("    - 轮数: 50 (演示用，实际应使用100-200)")
    print("    - 批大小: 32")
    print("    - 学习率: 1e-3")
    
    Z_all = trainer.phase1_train(
        X=X_tensor.to(device),
        C=C_tensor.to(device),
        L=L_tensor.to(device),
        num_epochs=50,
        batch_size=32,
        verbose=True
    )
    
    print(f"\n✓ Phase 1 完成")
    print(f"  - 潜变量Z形状: {Z_all.shape}")
    print(f"  - 训练历史: {len(trainer.history['phase1'])} epochs")
    
    # ================================================================
    # Step 6: Phase 2 - 动力学学习
    # ================================================================
    print_header("Step 6: Phase 2 - Flow Matching动力学学习")
    
    print("开始Phase 2训练 (冻结VAE，训练Vector Field)...")
    print("  配置:")
    print("    - 轮数: 50 (演示用，实际应使用200-500)")
    print("    - 批大小: 32")
    print("    - OT耦合: Sinkhorn")
    
    trainer.phase2_train(
        X=X_tensor.to(device),
        C=C_tensor.to(device),
        L=L_tensor.to(device),
        Z_all=Z_all.to(device),
        time_labels=time_labels_tensor.to(device),
        num_epochs=50,
        batch_size=32,
        verbose=True
    )
    
    print(f"\n✓ Phase 2 完成")
    print(f"  - 训练历史: {len(trainer.history['phase2'])} epochs")
    
    # ================================================================
    # Step 7: 推理 - 生成轨迹
    # ================================================================
    print_header("Step 7: ODE推理 - 生成轨迹")
    
    integrator = ODEIntegrator(
        model=model,
        solver='dopri5',
        device=device
    )
    
    # 选择几个初始细胞状态
    n_demo = 3
    z_starts = Z_all[:n_demo]
    
    print(f"为 {n_demo} 个细胞生成轨迹...")
    
    # 为不同条件生成轨迹
    for demo_idx in range(n_demo):
        z_init = z_starts[demo_idx:demo_idx+1]
        
        # 为条件0生成轨迹
        condition = C_tensor[0:1].to(device)  # 条件0
        lib_size = L_tensor[0:1].to(device)
        
        traj_z, traj_mean, info = integrator.simulate_trajectory(
            z_initial=z_init.to(device),
            condition=condition,
            t_span=torch.linspace(0, 1, 50, device=device),
            library_size=lib_size
        )
        
        print(f"\n  细胞{demo_idx}:")
        print(f"    - 潜空间轨迹: {traj_z.shape}")
        print(f"    - 基因表达轨迹: {traj_mean.shape}")
        print(f"    - 初始表达均值: {traj_mean[0, 0].mean().item():.4f}")
        print(f"    - 末端表达均值: {traj_mean[-1, 0].mean().item():.4f}")
    
    print(f"\n✓ 轨迹生成完成")
    
    # ================================================================
    # Step 8: 反事实分析
    # ================================================================
    print_header("Step 8: 反事实模拟")
    
    print("进行反事实分析：对比两种条件下的演化...")
    
    # 固定一个初始细胞
    z_start = Z_all[0:1].to(device)
    
    # 两种条件
    c_condition0 = torch.tensor([[1.0, 0.0]]).to(device)  # 条件0
    c_condition1 = torch.tensor([[0.0, 1.0]]).to(device)  # 条件1
    
    # 生成反事实轨迹
    traj_z_0, traj_mean_0, traj_z_1, traj_mean_1 = integrator.counterfactual_simulation(
        z_initial=z_start,
        condition_original=c_condition0,
        condition_counterfactual=c_condition1,
        t_span=torch.linspace(0, 1, 30).to(device),
        library_size=L_tensor[0:1].to(device)
    )
    
    # 计算两条轨迹之间的差异
    diff = torch.abs(traj_mean_0 - traj_mean_1).mean(dim=(1, 2))
    
    print(f"✓ 反事实分析完成")
    print(f"  - 初始差异: {diff[0].item():.6f}")
    print(f"  - 末端差异: {diff[-1].item():.6f}")
    print(f"  - 最大差异: {diff.max().item():.6f}")
    
    # ================================================================
    # Step 9: 模型保存
    # ================================================================
    print_header("Step 9: 保存训练结果")
    
    checkpoint_path = '/tmp/causal_genoflow_demo.pt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"✓ 模型已保存到: {checkpoint_path}")
    
    # ================================================================
    # 完成
    # ================================================================
    print_header("工作流完成!")
    
    print("""
总结：
✓ 数据加载和预处理
✓ 模型构建（6个神经网络模块）
✓ Phase 1：NB-VAE预热训练
✓ Phase 2：Flow Matching动力学学习
✓ ODE推理和轨迹生成
✓ 反事实分析（虚拟临床试验）
✓ 模型保存

关键指标：
- 总模型参数: {total_params:,}
- 潜空间维度: {n_latent}
- 细胞数: {n_cells}
- 基因数: {n_genes}
- 使用设备: {device}

下一步建议：
1. 使用真实的scRNA数据替换合成数据
2. 集成真实的GRN（DoRothEA）
3. 调整超参数以适应你的数据
4. 进行更深入的分析和可视化
5. 参考README.md获取详细文档

更多信息请查看：README.md
""".format(
        total_params=total_params,
        n_latent=n_latent,
        n_cells=n_cells,
        n_genes=n_genes,
        device=device
    ))


if __name__ == '__main__':
    main()
