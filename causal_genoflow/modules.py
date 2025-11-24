"""
神经网络模块定义

实现model.txt第5章节和details.txt修正部分定义的所有模块：
- CausalEncoder：编码器，处理条件和离散数据
- NBDecoder：负二项解码器
- TissueDiscriminator：对抗判别器（details2.txt缺口一）
- CorrectedGNNVectorField：GNN向量场，含交叉注意力（details.txt修正）
- CausalGenoFlow：主模型框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CausalEncoder(nn.Module):
    """
    因果编码器
    
    model.txt第2章节，表示分布的编码器：
    q_φ(z|x,c) = N(z; μ_φ(x,c), diag(σ²_φ(x,c)))
    
    设计要点：
    - 输入：log1p(x)归一化的表达计数 + 条件向量c
    - 输出：潜变量的均值和对数方差
    - 架构：FC -> BatchNorm -> LeakyReLU的堆叠
    
    实现对应关系：
    - model.txt代码第148-167行
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_latent: int, 
        n_cond: int,
        hidden_dim: int = 512
    ):
        """
        初始化编码器
        
        Args:
            n_input: 输入基因数量（来自x的维度）
            n_latent: 潜空间维度（z的维度）
            n_cond: 条件向量维度（c的维度）
            hidden_dim: 隐层维度
        """
        super().__init__()
        
        # 主体网络：处理拼接的[x, c]输入
        self.fc = nn.Sequential(
            nn.Linear(n_input + n_cond, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # 输出头：生成均值μ和对数方差log σ²
        self.mu_head = nn.Linear(hidden_dim // 2, n_latent)
        self.logvar_head = nn.Linear(hidden_dim // 2, n_latent)
    
    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入表达（通常是log1p归一化） (batch_size, n_input)
            c: 条件向量 (batch_size, n_cond)
        
        Returns:
            (mu, logvar) 后验参数
        """
        # 拼接输入和条件
        h = self.fc(torch.cat([x, c], dim=1))
        
        # 生成均值和对数方差
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        return mu, logvar
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        重参数化技巧
        
        z = μ + σ⊙ε，其中ε~N(0,I)
        
        Args:
            mu: 均值 (batch_size, n_latent)
            logvar: 对数方差 (batch_size, n_latent)
        
        Returns:
            采样的z (batch_size, n_latent)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class NBDecoder(nn.Module):
    """
    负二项解码器
    
    model.txt第2章节A小节的解码器设计：
    p_ψ(x|z,l,c) = ∏_g NB(x_g; μ_g, θ_g)
    
    关键设计（model.txt第41-44行）：
    1. 预测归一化表达率：ρ = Softmax(f_ψ^ρ(z,c))
    2. 通过库大小缩放：μ_g = l · ρ_g
    3. 基因特异性离散度：θ_g = exp(f_ψ^θ(z))
    
    实现对应关系：
    - model.txt代码第177-207行
    """
    
    def __init__(
        self, 
        n_latent: int, 
        n_output: int, 
        n_cond: int,
        hidden_dim: int = 512
    ):
        """
        初始化NB解码器
        
        Args:
            n_latent: 输入潜空间维度
            n_output: 输出基因数量
            n_cond: 条件向量维度
            hidden_dim: 隐层维度
        """
        super().__init__()
        
        # 主体网络：处理拼接的[z, c]输入
        self.fc = nn.Sequential(
            nn.Linear(n_latent + n_cond, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 输出头1：缩放因子（表达比例）
        # Softmax保证Σρ_g ≈ 1（相对丰度）
        self.scale_head = nn.Linear(hidden_dim, n_output)
        
        # 输出头2：基因特异性离散度（只依赖于基因，不依赖于z或c）
        # 这是为了数值稳定性（scVI推荐做法）
        self.register_parameter(
            'px_r', 
            nn.Parameter(torch.randn(n_output))
        )
    
    def forward(
        self, 
        z: torch.Tensor, 
        c: torch.Tensor, 
        library_size: torch.Tensor
    ) -> tuple:
        """
        前向传播
        
        Args:
            z: 潜变量 (batch_size, n_latent)
            c: 条件向量 (batch_size, n_cond)
            library_size: 库大小 (batch_size, 1) 或标量
        
        Returns:
            (mean, theta) 其中：
            - mean: 预测的均值 μ_g = l · ρ_g (batch_size, n_output)
            - theta: 离散度参数 (batch_size, n_output) 或 (n_output,)
        """
        # 通过FC网络生成隐特征
        h = self.fc(torch.cat([z, c], dim=1))
        
        # 缩放因子：使用Softmax确保相对丰度
        scale = torch.softmax(self.scale_head(h), dim=1)
        
        # 均值：缩放因子乘以库大小（model.txt关键约束）
        # 库大小可能是标量或(batch_size, 1)，需要广播
        if library_size.ndim == 1:
            library_size = library_size.unsqueeze(1)
        mean = library_size * scale
        
        # 离散度：通过exp确保θ > 0
        # px_r是基因特异性参数，使用exp(px_r)得到θ
        theta = torch.exp(self.px_r)
        
        return mean, theta


class TissueDiscriminator(nn.Module):
    """
    组织判别器（对抗组件）
    
    details2.txt缺口一的实现：
    从潜变量z推断条件c（如组织类型、疾病状态）。
    
    目标：
    1. 判别器D：最大化从z正确预测c的概率
    2. 生成器G（编码器）：最小化D的准确率（通过对抗损失）
    
    结果：z被迫学习独立于c的表示，实现解耦。
    
    实现对应关系：
    - details2.txt第24-37行
    """
    
    def __init__(
        self, 
        n_latent: int, 
        n_conditions: int,
        hidden_dim: int = 128
    ):
        """
        初始化判别器
        
        Args:
            n_latent: 输入潜空间维度
            n_conditions: 条件类别数量（如组织类型数）
            hidden_dim: 隐层维度
        """
        super().__init__()
        
        # 网络：尝试从z反推条件c
        self.net = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_conditions)  # 输出各类别的logits
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜变量 (batch_size, n_latent)
        
        Returns:
            logits: 条件预测的logits (batch_size, n_conditions)
        """
        return self.net(z)


class CorrectedGNNVectorField(nn.Module):
    """
    修正的GNN向量场
    
    model.txt第3章节C小节和details.txt修正部分的完整实现。
    
    解决关键问题（details.txt第44-53行）：
    - PyG GNN在节点（基因）上卷积，但z来自细胞
    - 解决方案：使用可学习的基因嵌入 + 交叉注意力
    
    架构（details.txt第124-149行）：
    1. 基因嵌入：E_gene ∈ R^(n_genes × d)
    2. GNN处理：graph_feat = GAT(E_gene, A_grn)
    3. 交叉注意力：cells attend to genes
    4. 条件嵌入：c -> c_emb
    5. 最终MLP：combine [z, attention_context, t_emb, c_emb]
    
    实现对应关系：
    - model.txt第212-254行
    - details.txt第122-189行
    """
    
    def __init__(
        self, 
        n_latent: int, 
        n_genes: int,
        n_cond: int,
        edge_index: torch.Tensor = None,
        n_hidden: int = 64
    ):
        """
        初始化GNN向量场
        
        Args:
            n_latent: 潜空间维度
            n_genes: 基因数量
            n_cond: 条件向量维度
            edge_index: GRN的邻接矩阵 (2, n_edges)，PyG格式
            n_hidden: GNN隐层维度
        """
        super().__init__()
        
        # Step 1: 可学习的基因嵌入矩阵
        # 形状：(n_genes, n_hidden)
        # 使用Xavier初始化保证稳定的梯度流
        self.gene_embeddings = nn.Parameter(
            torch.empty(n_genes, n_hidden)
        )
        nn.init.xavier_uniform_(self.gene_embeddings, gain=1.0)
        
        # Step 2: 图注意力网络
        # 输入：(n_genes, n_hidden) -> 输出：(n_genes, n_hidden*2)
        self.gat = GATv2Conv(
            in_channels=n_hidden,
            out_channels=n_hidden,
            heads=2,
            concat=True,  # 连接多头输出，维度变成n_hidden*2
            dropout=0.1
        )
        
        # Step 3: 交叉注意力的投影层
        # 将细胞状态z投影到与基因特征相同的维度
        self.query_proj = nn.Linear(n_latent, n_hidden * 2)
        # Key和Value就是图特征，不需要额外投影
        self.key_proj = nn.Identity()
        self.val_proj = nn.Identity()
        
        # Step 4: 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16)
        )
        
        # Step 5: 条件嵌入（缺口二：向量场需接收条件）
        self.cond_mlp = nn.Linear(n_cond, 16)
        
        # Step 6: 最终速度预测MLP
        # 输入维度：n_latent(z) + n_hidden*2(attention) + 16(time) + 16(cond)
        self.final_mlp = nn.Sequential(
            nn.Linear(n_latent + (n_hidden * 2) + 16 + 16, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_latent)
        )
        
        # 保存edge_index用于GNN计算
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.edge_index = None
    
    def forward(
        self, 
        t: torch.Tensor, 
        z: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：计算向量场v_θ(z_t, t, c)
        
        Args:
            t: 时间 (batch_size,) 或 (batch_size, 1)
            z: 细胞状态 (batch_size, n_latent)
            c: 条件向量 (batch_size, n_cond)
        
        Returns:
            v: 预测的速度向量 (batch_size, n_latent)
        
        实现流程对应details.txt第153-188行
        """
        # Step A: 处理基因图（独立于批大小）
        # graph_feat: (n_genes, n_hidden*2)
        # 当未提供GRN或边为空时，跳过GAT以避免维度错误
        if self.edge_index is None or self.edge_index.numel() == 0:
            graph_feat = self.gene_embeddings
        else:
            graph_feat = self.gat(self.gene_embeddings, self.edge_index)
        
        # Step B: 交叉注意力（细胞查看基因）
        # Query: 细胞状态 (batch_size, n_hidden*2)
        Q = self.query_proj(z)
        # Key: 基因特征 (n_genes, n_hidden*2)
        K = graph_feat
        # Value: 基因特征 (n_genes, n_hidden*2)
        V = graph_feat
        
        # 注意力分数：[batch_size, n_genes]
        # 代表：这个细胞状态对哪些基因更"感兴趣"
        # 
        # 维度分析：
        # Q: (batch, n_hidden*2)
        # K.t(): (n_hidden*2, n_genes) 
        # 结果: (batch, n_genes) ✓
        #
        # 缩放：应该按照d_k的维度缩放，即sqrt(n_hidden*2)
        d_k = Q.shape[-1]  # n_hidden*2
        attn_scores = torch.matmul(Q, K.t()) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 聚合基因背景信息：[batch_size, n_hidden*2]
        context = torch.matmul(attn_weights, V)
        
        # Step C: 时间嵌入
        # 确保t的形状是(batch_size, 1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t_emb = self.time_mlp(t)
        
        # Step D: 条件嵌入
        c_emb = self.cond_mlp(c)
        
        # Step E: 预测速度
        # 组合：细胞状态 + 基因背景 + 时间 + 条件
        v_input = torch.cat([z, context, t_emb, c_emb], dim=1)
        v = self.final_mlp(v_input)
        
        return v


class CausalGenoFlow(nn.Module):
    """
    完整的Causal-GenoFlow模型
    
    model.txt第5章节的主模型框架。
    
    结合VAE（潜流形学习）和Flow Matching（动力学建模）的完整框架。
    
    两阶段训练（details.txt第54-61行）：
    Phase 1: 冻结Flow，训练NB-VAE学习潜流形
    Phase 2: 冻结VAE，计算OT，训练Vector Field
    
    实现对应关系：
    - model.txt代码第259-299行
    """
    
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        n_cond: int,
        grn_edge_index: torch.Tensor = None,
        beta: float = 1.0,
        lambda_adv: float = 0.1,
        use_adversarial: bool = True
    ):
        """
        初始化完整模型
        
        Args:
            n_genes: 基因数量
            n_latent: 潜空间维度
            n_cond: 条件向量维度
            grn_edge_index: GRN邻接表 (2, n_edges)
            beta: KL权重
            lambda_adv: 对抗损失权重
            use_adversarial: 是否使用对抗训练
        """
        super().__init__()
        
        # 编码器
        self.encoder = CausalEncoder(
            n_input=n_genes,
            n_latent=n_latent,
            n_cond=n_cond
        )
        
        # 解码器
        self.decoder = NBDecoder(
            n_latent=n_latent,
            n_output=n_genes,
            n_cond=n_cond
        )
        
        # 向量场（GNN约束）
        self.vector_field = CorrectedGNNVectorField(
            n_latent=n_latent,
            n_genes=n_genes,
            n_cond=n_cond,
            edge_index=grn_edge_index
        )
        
        # 可选：对抗判别器
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.discriminator = TissueDiscriminator(
                n_latent=n_latent,
                n_conditions=n_cond  # 假设n_cond个不同的条件
            )
        
        # 损失函数
        from .losses import CombinedVAELoss, FlowMatchingLoss
        self.vae_loss = CombinedVAELoss(
            beta=beta,
            lambda_adv=lambda_adv,
            use_adversarial=use_adversarial
        )
        self.fm_loss = FlowMatchingLoss()
        
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_cond = n_cond
    
    def encode(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor
    ) -> tuple:
        """
        编码：x, c -> z
        
        Args:
            x: 原始表达计数 (batch_size, n_genes)
            c: 条件向量 (batch_size, n_cond)
        
        Returns:
            (z, mu_z, logvar_z) 潜变量及其分布参数
        """
        mu_z, logvar_z = self.encoder(x, c)
        z = self.encoder.reparameterize(mu_z, logvar_z)
        return z, mu_z, logvar_z
    
    def decode(
        self, 
        z: torch.Tensor, 
        c: torch.Tensor, 
        library_size: torch.Tensor
    ) -> tuple:
        """
        解码：z, c, l -> x
        
        Args:
            z: 潜变量 (batch_size, n_latent)
            c: 条件向量 (batch_size, n_cond)
            library_size: 库大小 (batch_size,) 或 (batch_size, 1)
        
        Returns:
            (mean, theta) NB分布的参数
        """
        return self.decoder(z, c, library_size)
    
    def vae_forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        library_size: torch.Tensor,
        detach_adv: bool = False
    ) -> tuple:
        """
        VAE前向传播（用于Phase 1训练）
        
        Args:
            x: 观测计数 (batch_size, n_genes)
            c: 条件向量 (batch_size, n_cond)
            library_size: 库大小 (batch_size,)
        
        Returns:
            (loss, z, mean_recon, theta_recon, mu_z, logvar_z)

        detach_adv:
            - False: 允许对抗损失的梯度回传到编码器，驱动解耦
            - True: 仅用于推理或判别器训练，避免对抗项影响编码器
        """
        # 1. 编码：x, c -> z
        x_log = torch.log1p(x)  # Log归一化输入
        mu_z, logvar_z = self.encoder(x_log, c)
        z = self.encoder.reparameterize(mu_z, logvar_z)
        
        # 2. 解码：z, c, l -> 重建参数
        mean_recon, theta_recon = self.decoder(z, c, library_size)
        
        # 3. 计算VAE损失
        if self.use_adversarial:
            adv_input = z.detach() if detach_adv else z
            adv_logits = self.discriminator(adv_input)
        else:
            adv_logits = None
        
        loss, loss_dict = self.vae_loss(
            x, mean_recon, theta_recon,
            mu_z, logvar_z,
            adv_logits=adv_logits
        )
        
        return loss, z, mean_recon, theta_recon, mu_z, logvar_z
    
    def flow_forward(
        self,
        z_t: torch.Tensor,
        z_target: torch.Tensor,
        t_batch: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        流匹配前向传播（用于Phase 2训练）
        
        Args:
            z_t: 插值后的潜变量 ψ_t(z_0, z_1) (batch_size, n_latent)
            z_target: 目标潜变量 z_1 (batch_size, n_latent)
            t_batch: 时间 (batch_size,)
            c: 条件向量 (batch_size, n_cond)
        
        Returns:
            flow_loss: 流匹配损失
        """
        # 预测向量场
        v_pred = self.vector_field(t_batch, z_t, c)
        
        # 目标向量
        u_target = z_target  # 已经是z_1 - z_0（在外部计算）
        
        # 计算损失
        flow_loss = self.fm_loss(v_pred, u_target)
        
        return flow_loss
    
    def sample_trajectory(
        self,
        z_start: torch.Tensor,
        c: torch.Tensor,
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        采样轨迹（推理阶段）
        
        使用ODE求解器从z_start沿着t_span生成轨迹。
        这由inference.py的ODEIntegrator处理。
        
        Args:
            z_start: 初始潜变量 (batch_size, n_latent)
            c: 条件向量 (batch_size, n_cond)
            t_span: 时间轴 (n_timepoints,)
        
        Returns:
            traj_z: 轨迹 (n_timepoints, batch_size, n_latent)
        """
        # 这个函数的具体实现在inference.py
        # 这里仅作为接口留下
        raise NotImplementedError("Use ODEIntegrator from inference.py")
