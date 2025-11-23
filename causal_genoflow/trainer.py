"""
训练模块

实现details.txt第二部分的两阶段训练流程：
- Phase 1：训练NB-VAE学习潜流形
- Phase 2：计算OT耦合，训练Flow Matching

详细流程见：details.txt第54-61行和第74-112行的算法1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
import tqdm


class TwoStageTrainer:
    """
    两阶段训练器
    
    严格按照details.txt第54-61行的两阶段训练策略：
    
    Stage 1: Warm-up (100 Epochs)
    - 冻结Flow网络
    - 训练NB-VAE (Encoder + Decoder)
    - 同时训练TissueDiscriminator（对抗解耦）
    
    Stage 2: Dynamics Learning
    - 冻结Encoder/Decoder
    - 计算全量Latent Z
    - 计算OT耦合对
    - 训练Vector Field (Flow Matching)
    """
    
    @staticmethod
    def check_gradients(model, phase_name: str = ""):
        """
        检查模型梯度的健康状况
        
        诊断梯度消失、梯度爆炸或梯度断裂的问题
        
        Args:
            model: PyTorch模型
            phase_name: 调试阶段名称
        """
        grad_norms = {}
        has_nan_grad = False
        has_zero_grad = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norms[name] = grad_norm
                
                if torch.any(torch.isnan(param.grad)):
                    print(f"[梯度错误] {name} 包含NAN梯度 ({phase_name})")
                    has_nan_grad = True
                
                if grad_norm == 0:
                    print(f"[梯度警告] {name} 梯度为零 ({phase_name})")
                    has_zero_grad = True
                
                elif grad_norm < 1e-7:
                    print(f"[梯度警告] {name} 梯度过小: {grad_norm:.2e} ({phase_name})")
                
                elif grad_norm > 1e2:
                    print(f"[梯度警告] {name} 梯度过大: {grad_norm:.2e} ({phase_name})")
        
        return grad_norms, has_nan_grad, has_zero_grad
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_fm: float = 1.0
    ):
        """
        初始化训练器
        
        Args:
            model: CausalGenoFlow模型实例
            device: 'cpu' 或 'cuda'
            learning_rate: 学习率
            beta: KL项权重
            lambda_adv: 对抗项权重
            lambda_fm: 流匹配项权重
        """
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.beta = beta
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        
        # 训练历史
        self.history = {
            'phase1': [],
            'phase2': []
        }
    
    def phase1_train(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        L: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Phase 1: NB-VAE预热
        
        details.txt第85-95行的Phase 1实现。
        
        目标：学习一个良好的潜流形，使得latent space光滑并接近高斯。
        
        Args:
            X: 观测计数 (n_cells, n_genes)
            C: 条件向量 (n_cells, n_cond)
            L: 库大小 (n_cells,)
            num_epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印日志
        
        Returns:
            Z_all: 全量潜变量 (n_cells, n_latent)
        """
        print("[Phase 1] 开始NB-VAE预热训练...")
        
        # 冻结Flow相关参数（向量场）
        for param in self.model.vector_field.parameters():
            param.requires_grad = False
        
        # 解冻VAE和判别器
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        if self.model.use_adversarial:
            for param in self.model.discriminator.parameters():
                param.requires_grad = True
        
        # 优化器
        vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        optimizer_vae = torch.optim.Adam(vae_params, lr=self.lr)
        
        if self.model.use_adversarial:
            optimizer_disc = torch.optim.Adam(
                self.model.discriminator.parameters(),
                lr=self.lr * 0.5
            )
        
        # 数据加载
        dataset = TensorDataset(X, C, L)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            pbar = tqdm.tqdm(dataloader, disable=not verbose)
            for batch_x, batch_c, batch_l in pbar:
                batch_x = batch_x.to(self.device)
                batch_c = batch_c.to(self.device)
                batch_l = batch_l.to(self.device)
                
                # ==================
                # 1. 训练VAE
                # ==================
                optimizer_vae.zero_grad()
                
                # VAE前向传播
                loss_vae, z, mean_recon, theta_recon, mu_z, logvar_z = \
                    self.model.vae_forward(batch_x, batch_c, batch_l)
                
                # 不包含对抗项的VAE损失（先单独算）
                vae_base_loss = loss_vae
                
                # 对抗损失（如果启用）
                if self.model.use_adversarial:
                    # ==================
                    # 2. 训练判别器
                    # ==================
                    optimizer_disc.zero_grad()
                    
                    # 判别器看真实条件
                    z_detached = z.detach()
                    disc_logits = self.model.discriminator(z_detached)
                    
                    # 条件标签转换
                    # 必须确定C的格式：独热编码或标签索引
                    if batch_c.ndim > 1:
                        # 假设是独热编码 (batch, n_cond)
                        if batch_c.shape[1] > 1:
                            # 多个1表示不是严格独热编码，尝试转换
                            c_labels = torch.argmax(batch_c, dim=1)
                        else:
                            # 只有一列，直接作为标签
                            c_labels = batch_c.squeeze(1).long()
                    else:
                        # 已经是标签
                        c_labels = batch_c.long()
                    
                    # 安全检查：标签范围应该在[0, n_cond)
                    if torch.any(c_labels < 0) or torch.any(c_labels >= batch_c.shape[-1]):
                        print(f"[警告] 条件标签超出范围: min={c_labels.min()}, max={c_labels.max()}, expected=[0, {batch_c.shape[-1]})")
                        # 跳过此批的判别器训练
                        loss_disc = torch.tensor(0.0, device=batch_c.device)
                    else:
                        loss_disc = F.cross_entropy(disc_logits, c_labels)
                    
                    loss_disc.backward()
                    optimizer_disc.step()
                    
                    # ==================
                    # 3. 对抗训练编码器
                    # ==================
                    # 编码器试图欺骗判别器
                    optimizer_vae.zero_grad()
                    disc_logits_gen = self.model.discriminator(z)
                    
                    # 对抗损失：最大化判别器的熵（让其无法确定条件）
                    probs = F.softmax(disc_logits_gen, dim=1)
                    entropy = -torch.sum(probs * F.log_softmax(disc_logits_gen, dim=1), dim=1)
                    loss_adv = -torch.mean(entropy)
                    
                    # 总损失
                    total_loss = vae_base_loss + self.lambda_adv * loss_adv
                else:
                    total_loss = vae_base_loss
                
                total_loss.backward()
                optimizer_vae.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
                
                if verbose:
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs} Loss: {total_loss.item():.4f}")
            
            avg_loss = epoch_loss / n_batches
            self.history['phase1'].append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}] 平均损失: {avg_loss:.4f}")
        
        print("[Phase 1] NB-VAE预热完成！")
        
        # 冻结VAE，计算全量Z
        self.model.eval()
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        
        # 计算全量潜变量
        dataset_all = TensorDataset(X, C, L)
        dataloader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=False)
        
        Z_list = []
        with torch.no_grad():
            for batch_x, batch_c, batch_l in dataloader_all:
                batch_x = batch_x.to(self.device)
                batch_c = batch_c.to(self.device)
                
                x_log = torch.log1p(batch_x)
                mu_z, _ = self.model.encoder(x_log, batch_c)
                Z_list.append(mu_z.cpu())
        
        Z_all = torch.cat(Z_list, dim=0)
        
        print(f"[Phase 1] 计算的潜变量形状: {Z_all.shape}")
        
        return Z_all
    
    def compute_ot_coupling(
        self,
        Z_source: torch.Tensor,
        Z_target: torch.Tensor,
        method: str = 'sinkhorn',
        epsilon: float = 0.01
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        计算最优传输耦合
        
        details.txt第98-102行的OT计算。
        
        使用Sinkhorn算法求解Kantorovich问题：
        π* = argmin_π Σ_ij π_ij ||z_i - z_j||²
        
        Args:
            Z_source: 源分布的潜变量 (n_source, n_latent)
            Z_target: 目标分布的潜变量 (n_target, n_latent)
            method: 'sinkhorn' 或 'emd'
            epsilon: Sinkhorn正则化参数
        
        Returns:
            (ot_matrix, paired_indices)
        """
        try:
            import ot  # POT库
        except ImportError:
            raise ImportError("请安装POT库用于OT计算: pip install POT")
        
        print(f"[OT] 计算最优传输耦合...")
        print(f"[OT] 源分布大小: {Z_source.shape[0]}, 目标分布大小: {Z_target.shape[0]}")
        
        # Z-score标准化（details.txt第36行要求）
        # 数值稳定性处理：
        # 1. 检查是否为空
        # 2. 处理std=0的情况
        # 3. 防止NAN传播
        
        if Z_source.shape[0] == 0 or Z_target.shape[0] == 0:
            raise ValueError(f"空的潜变量集合: Z_source={Z_source.shape}, Z_target={Z_target.shape}")
        
        # 计算统计量
        source_mean = Z_source.mean(dim=0, keepdim=True)
        source_std = Z_source.std(dim=0, keepdim=True)
        target_mean = Z_target.mean(dim=0, keepdim=True)
        target_std = Z_target.std(dim=0, keepdim=True)
        
        # 处理std=0的情况（某个维度方差为0）
        # 用小值替换以避免除以0
        source_std = torch.clamp(source_std, min=1e-8)
        target_std = torch.clamp(target_std, min=1e-8)
        
        Z_source_norm = (Z_source - source_mean) / source_std
        Z_target_norm = (Z_target - target_mean) / target_std
        
        # 检查是否产生了NAN
        if torch.any(torch.isnan(Z_source_norm)) or torch.any(torch.isnan(Z_target_norm)):
            print(f"[警告] OT标准化后包含NAN")
            print(f"  Z_source_norm NAN数量: {torch.sum(torch.isnan(Z_source_norm)).item()}")
            print(f"  Z_target_norm NAN数量: {torch.sum(torch.isnan(Z_target_norm)).item()}")
            # 用0替换NAN（该维度被忽略）
            Z_source_norm = torch.where(torch.isnan(Z_source_norm), torch.zeros_like(Z_source_norm), Z_source_norm)
            Z_target_norm = torch.where(torch.isnan(Z_target_norm), torch.zeros_like(Z_target_norm), Z_target_norm)
        
        # 计算成本矩阵（欧氏距离平方）
        Z_source_np = Z_source_norm.numpy()
        Z_target_np = Z_target_norm.numpy()
        
        cost_matrix = ot.dist(Z_source_np, Z_target_np, metric='euclidean')
        
        # 计算OT矩阵
        if method == 'sinkhorn':
            ot_matrix = ot.sinkhorn(
                np.ones(Z_source.shape[0]) / Z_source.shape[0],
                np.ones(Z_target.shape[0]) / Z_target.shape[0],
                cost_matrix,
                epsilon
            )
        elif method == 'emd':
            ot_matrix = ot.emd(
                np.ones(Z_source.shape[0]) / Z_source.shape[0],
                np.ones(Z_target.shape[0]) / Z_target.shape[0],
                cost_matrix
            )
        else:
            raise ValueError(f"Unknown OT method: {method}")
        
        # 从OT矩阵提取配对
        # 选择概率最高的配对
        paired_indices = []
        for i in range(Z_source.shape[0]):
            j = np.argmax(ot_matrix[i])
            paired_indices.append((i, j))
        
        print(f"[OT] 生成了 {len(paired_indices)} 对样本")
        
        return ot_matrix, paired_indices
    
    def phase2_train(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        L: torch.Tensor,
        Z_all: torch.Tensor,
        time_labels: torch.Tensor,
        num_epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True
    ) -> None:
        """
        Phase 2: 动力学学习
        
        details.txt第103-112行的Phase 2实现。
        
        目标：训练向量场拟合Flow Matching损失。
        
        Args:
            X: 观测计数
            C: 条件向量
            L: 库大小
            Z_all: Phase 1计算的全量潜变量
            time_labels: 时间标签 (n_cells,)
            num_epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印日志
        """
        print("[Phase 2] 开始动力学学习...")
        
        # 冻结VAE
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        
        # 解冻向量场
        for param in self.model.vector_field.parameters():
            param.requires_grad = True
        
        # 优化器
        optimizer_flow = torch.optim.Adam(
            self.model.vector_field.parameters(),
            lr=self.lr
        )
        
        # 获取时间点
        unique_times = sorted(torch.unique(time_labels).tolist())
        print(f"[Phase 2] 发现的时间点: {unique_times}")
        
        # 为每对相邻时间点计算OT耦合
        all_pairs = []
        all_t_values = []
        
        for t_idx in range(len(unique_times) - 1):
            t_A = unique_times[t_idx]
            t_B = unique_times[t_idx + 1]
            
            # 获取这两个时间点的细胞
            mask_A = (time_labels == t_A)
            mask_B = (time_labels == t_B)
            
            Z_A = Z_all[mask_A]
            Z_B = Z_all[mask_B]
            C_A = C[mask_A]
            
            print(f"[Phase 2] 时间点 {t_A} -> {t_B}: {Z_A.shape[0]} -> {Z_B.shape[0]} 细胞")
            
            # 计算OT耦合
            ot_matrix, paired_indices = self.compute_ot_coupling(Z_A, Z_B)
            
            # 构建训练对
            for i, j in paired_indices:
                z0 = Z_A[i:i+1]
                z1 = Z_B[j:j+1]
                c = C_A[i:i+1]
                t = torch.tensor([(t_A + t_B) / 2.0])  # 使用中点作为时间
                
                all_pairs.append((z0, z1, c, t))
                all_t_values.append(t.item())
        
        print(f"[Phase 2] 生成了 {len(all_pairs)} 个训练对")
        
        # 训练Loop
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # 随机打乱对
            indices = np.random.permutation(len(all_pairs))
            
            pbar = tqdm.tqdm(range(0, len(indices), batch_size), disable=not verbose)
            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]
                
                # 收集批数据
                batch_z0_list = []
                batch_z1_list = []
                batch_c_list = []
                batch_t_list = []
                
                for idx in batch_indices:
                    z0, z1, c, t = all_pairs[idx]
                    batch_z0_list.append(z0)
                    batch_z1_list.append(z1)
                    batch_c_list.append(c)
                    batch_t_list.append(t)
                
                batch_z0 = torch.cat(batch_z0_list, dim=0).to(self.device)
                batch_z1 = torch.cat(batch_z1_list, dim=0).to(self.device)
                batch_c = torch.cat(batch_c_list, dim=0).to(self.device)
                batch_t = torch.cat(batch_t_list, dim=0).to(self.device)
                
                # 随机时间采样 (details.txt第106行)
                t_random = torch.rand(batch_z0.shape[0], device=self.device)
                
                # 线性插值 (model.txt第70行)
                batch_z_t = (1 - t_random.unsqueeze(1)) * batch_z0 + \
                            t_random.unsqueeze(1) * batch_z1
                
                # 目标向量 (model.txt第72行)
                batch_u_target = batch_z1 - batch_z0
                
                # 前向传播
                optimizer_flow.zero_grad()
                batch_v_pred = self.model.vector_field(t_random, batch_z_t, batch_c)
                
                # 流匹配损失 (model.txt第75行)
                loss_fm = self.model.fm_loss(batch_v_pred, batch_u_target)
                loss_fm.backward()
                optimizer_flow.step()
                
                epoch_loss += loss_fm.item()
                n_batches += 1
                
                if verbose:
                    pbar.set_description(
                        f"Epoch {epoch+1}/{num_epochs} FM Loss: {loss_fm.item():.4f}"
                    )
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.history['phase2'].append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"[Epoch {epoch+1}] 平均FM损失: {avg_loss:.4f}")
        
        print("[Phase 2] 动力学学习完成！")
    
    def save_checkpoint(self, path: str) -> None:
        """
        保存模型检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            'model_state': self.model.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"[Save] 模型已保存到: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载模型检查点
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.history = checkpoint['history']
        print(f"[Load] 模型已从以下位置加载: {path}")
