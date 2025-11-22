"""
损失函数模块

实现model.txt第4-5章节定义的所有损失函数，包括：
- NBLoss：数值稳定的负二项损失
- KL散度损失
- 流匹配损失
- 对抗损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NBLoss(nn.Module):
    """
    数值稳定的负二项（Negative Binomial）损失函数
    
    model.txt第4章节的严格实现。
    
    负二项PMF：
    P(x|μ,θ) = Γ(x+θ)/(Γ(x+1)Γ(θ)) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x
    
    对数似然（Log-space，数值稳定）：
    log P = log Γ(x+θ) - log Γ(θ) - log Γ(x+1)
          + θ(log θ - log(θ+μ))
          + x(log μ - log(θ+μ))
    
    模型约束：
    - θ 必须在 [1e-4, 1e4] 范围内
    - μ 不能为0（使用eps防护）
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        初始化NB损失函数
        
        Args:
            eps: 数值稳定性边界值，防止log(0)
        """
        super().__init__()
        self.eps = eps
    
    def forward(
        self, 
        x: torch.Tensor, 
        mean: torch.Tensor, 
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NB损失
        
        Args:
            x: 观测计数 (batch_size, n_genes)，类型为float
            mean: 预测均值 μ_g = library_size * scale (batch_size, n_genes)
            theta: 离散度参数 θ_g (batch_size, n_genes) 或 (n_genes,)
        
        Returns:
            标量损失值（负对数似然的平均值）
        
        实现对应关系：
        - model.txt公式(3)：条件负二项分布
        - model.txt第85-99行：NB Loss的梯度稳定性推导
        """
        # 广播theta到正确的维度（如果是基因特异性）
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # 从 (n_genes,) -> (1, n_genes)
        
        # 数值稳定的log计算，防止log(0)或log负数
        log_theta_mu_eps = torch.log(theta + mean + self.eps)
        log_theta_eps = torch.log(theta + self.eps)
        log_mu_eps = torch.log(mean + self.eps)
        
        # 第一项：lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
        # 注：lgamma(x+1) = loggamma的常数部分，对梯度无影响，但保留以确保数值一致性
        term1 = (
            torch.lgamma(x + theta + self.eps) 
            - torch.lgamma(theta + self.eps) 
            - torch.lgamma(x + 1 + self.eps)
        )
        
        # 第二项：θ * (log θ - log(θ+μ))
        term2 = theta * (log_theta_eps - log_theta_mu_eps)
        
        # 第三项：x * (log μ - log(θ+μ))
        term3 = x * (log_mu_eps - log_theta_mu_eps)
        
        # 组合所有项得到对数似然
        log_likelihood = term1 + term2 + term3
        
        # 返回负对数似然的平均值（最小化）
        return -torch.mean(log_likelihood)


class KLDivergenceLoss(nn.Module):
    """
    高斯KL散度损失
    
    model.txt第2章节的VAE正则化项。
    
    KL(q(z|x)||p(z)) = KL(N(μ,σ²)||N(0,I))
                     = -0.5 * Σ(1 + log σ² - μ² - σ²)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化KL散度损失
        
        Args:
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度
        
        Args:
            mu: 后验均值 (batch_size, n_latent)
            logvar: 后验对数方差 (batch_size, n_latent)
        
        Returns:
            KL散度（标量或求和）
        """
        # KL = -0.5 * Σ(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(kl)
        elif self.reduction == 'sum':
            return torch.sum(kl)
        else:
            return kl


class FlowMatchingLoss(nn.Module):
    """
    条件流匹配（Conditional Flow Matching）损失
    
    model.txt第3章节的FM损失函数实现。
    
    目标：训练向量场 v_θ(z_t, t, c) 拟合直线插值路径
    L_FM = E_{t,z0,z1~π*}[||v_θ(ψ_t(z0,z1), t, c) - (z1-z0)||²]
    
    其中π*是OT耦合矩阵，确保z0和z1在同一轨迹上。
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化流匹配损失
        
        Args:
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        v_pred: torch.Tensor, 
        u_target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算流匹配损失（MSE）
        
        Args:
            v_pred: 预测的向量场 (batch_size, n_latent)
            u_target: 目标向量 z1 - z0 (batch_size, n_latent)
        
        Returns:
            MSE损失
        
        实现对应关系：
        - model.txt公式：L_FM = E[||v_θ(...) - u_t||²]
        - 直线插值的目标速度始终是常数 u_t = z1 - z0
        """
        loss = F.mse_loss(v_pred, u_target, reduction=self.reduction)
        return loss


class AdversarialLoss(nn.Module):
    """
    对抗损失函数
    
    details2.txt缺口一的实现：使用对抗判别器分离Z_tissue和Z_dynamics。
    
    两部分：
    1. 判别器损失：尽可能准确地从z预测条件c
    2. 生成器（编码器）对抗损失：最大化判别器的困惑度
    """
    
    def __init__(self, loss_type: str = 'entropy'):
        """
        初始化对抗损失
        
        Args:
            loss_type: 'minimax' 或 'entropy'
                - 'minimax'：L = -log(1 - p)，直接最大化判别器误分类
                - 'entropy'：L = H(softmax(logits))，最大化预测熵（推荐）
        """
        super().__init__()
        self.loss_type = loss_type
    
    def discriminator_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算判别器损失
        
        Args:
            logits: 判别器输出的logits (batch_size, n_conditions)
            labels: 真实条件标签 (batch_size,)
        
        Returns:
            交叉熵损失
        """
        return F.cross_entropy(logits, labels)
    
    def generator_loss(
        self, 
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算生成器（编码器）的对抗损失
        
        目标：让判别器猜不准条件（最大化熵或最小化置信度）
        
        Args:
            logits: 判别器输出的logits (batch_size, n_conditions)
        
        Returns:
            对抗损失
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        if self.loss_type == 'entropy':
            # 最大化熵：H = -Σ p*log p
            # 损失 = -H （我们要最小化这个，所以最大化H）
            entropy = -torch.sum(probs * log_probs, dim=1)
            return -torch.mean(entropy)  # 负号使得最小化等于最大化熵
        
        elif self.loss_type == 'minimax':
            # Minimax：-log(1 - p_true)，目标是最小化p_true（让判别器误分类）
            # 等价于：CE，但取负（最小化错误分类）
            return -F.cross_entropy(logits, logits.argmax(dim=1))
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class CombinedVAELoss(nn.Module):
    """
    联合VAE损失
    
    model.txt第2章节的完整ELBO：
    L = -E[log p(x|z,l,c)] + β·KL(q(z|x,c)||p(z))
    
    可选的对抗项用于因果解耦：
    L = L_recon + β·L_KL + λ·L_adv
    """
    
    def __init__(
        self, 
        beta: float = 1.0, 
        lambda_adv: float = 0.1, 
        use_adversarial: bool = True
    ):
        """
        初始化联合VAE损失
        
        Args:
            beta: KL项的权重
            lambda_adv: 对抗项的权重（如果使用）
            use_adversarial: 是否使用对抗损失
        """
        super().__init__()
        self.beta = beta
        self.lambda_adv = lambda_adv
        self.use_adversarial = use_adversarial
        
        self.nb_loss = NBLoss()
        self.kl_loss = KLDivergenceLoss(reduction='mean')
        if use_adversarial:
            self.adv_loss = AdversarialLoss(loss_type='entropy')
    
    def forward(
        self, 
        x: torch.Tensor,
        mu_recon: torch.Tensor,
        theta_recon: torch.Tensor,
        mu_z: torch.Tensor,
        logvar_z: torch.Tensor,
        adv_logits: torch.Tensor = None,
        condition_labels: torch.Tensor = None
    ) -> tuple:
        """
        计算完整的VAE损失
        
        Args:
            x: 观测计数 (batch_size, n_genes)
            mu_recon: 重建均值
            theta_recon: 重建离散度
            mu_z: 后验均值
            logvar_z: 后验对数方差
            adv_logits: 判别器输出（可选）
            condition_labels: 条件标签（可选，用于对抗损失）
        
        Returns:
            (total_loss, loss_dict) 其中loss_dict包含各项损失的详细信息
        """
        # 重建损失（NB）
        recon_loss = self.nb_loss(x, mu_recon, theta_recon)
        
        # KL损失
        kl_loss = self.kl_loss(mu_z, logvar_z)
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss
        
        # 对抗损失（可选）
        adv_loss = torch.tensor(0.0, device=x.device)
        if self.use_adversarial and adv_logits is not None:
            adv_loss = self.adv_loss.generator_loss(adv_logits)
            total_loss = total_loss + self.lambda_adv * adv_loss
        
        loss_dict = {
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
