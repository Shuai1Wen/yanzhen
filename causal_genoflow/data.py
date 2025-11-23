"""
数据处理和预处理模块

实现details2.txt缺口三的完整GRN对齐和图数据处理：
- GRN与scRNA基因的交集
- Edge index重索引
- 条件标签编码
- 库大小计算
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional


class GRNPreprocessor:
    """
    基因调控网络预处理器
    
    details2.txt缺口三：处理GRN与scRNA数据的基因对齐问题。
    
    关键步骤（details2.txt第120-128行）：
    1. 取交集：Valid_Genes = Genes_scRNA & Genes_GRN
    2. 子图提取：只保留有效基因之间的边
    3. 重索引：将基因名映射到0~n_valid_genes的整数
    """
    
    def __init__(self):
        """初始化GRN预处理器"""
        self.gene_to_idx = {}
        self.idx_to_gene = {}
        self.valid_genes = None
        self.edge_index = None
        self.n_valid_genes = 0
    
    def align_genes(
        self,
        scRNA_genes: List[str],
        GRN_edges: np.ndarray,
        GRN_gene_names: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        对齐scRNA数据和GRN的基因集合
        
        Args:
            scRNA_genes: scRNA-seq中的基因名列表 (n_genes_scRNA,)
            GRN_edges: GRN的边列表 (n_edges, 2)，包含基因名或索引
            GRN_gene_names: GRN的所有基因名 (n_genes_GRN,)
        
        Returns:
            (valid_genes, edge_index_tensor)
            - valid_genes: 交集基因列表
            - edge_index_tensor: 重索引后的PyG edge_index (2, n_valid_edges)
        
        实现对应关系：
        - details2.txt第119-128行：缺口三解决方案
        """
        # Step 1: 计算交集
        scRNA_set = set(scRNA_genes)
        GRN_set = set(GRN_gene_names)
        self.valid_genes = sorted(list(scRNA_set & GRN_set))
        self.n_valid_genes = len(self.valid_genes)
        
        # 打印统计信息
        print(f"[GRN Preprocessor] scRNA基因数: {len(scRNA_genes)}")
        print(f"[GRN Preprocessor] GRN基因数: {len(GRN_gene_names)}")
        print(f"[GRN Preprocessor] 交集基因数: {self.n_valid_genes}")
        
        if self.n_valid_genes == 0:
            raise ValueError("scRNA和GRN没有公共基因！")
        
        # Step 2: 创建基因名到索引的映射
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.valid_genes)}
        self.idx_to_gene = {idx: gene for idx, gene in enumerate(self.valid_genes)}
        
        # Step 3: 子图提取 + 重索引
        valid_edges = []
        n_original_edges = len(GRN_edges)
        
        for edge in GRN_edges:
            # 假设edge格式为[source, target]，可能是基因名或原始索引
            # 如果GRN_edges已经是基因名，直接使用
            if isinstance(edge[0], str):
                source_gene = edge[0]
                target_gene = edge[1]
            else:
                # 如果是索引，则从GRN_gene_names转换
                source_gene = GRN_gene_names[edge[0]]
                target_gene = GRN_gene_names[edge[1]]
            
            # 只保留两个基因都在有效集合中的边
            if source_gene in self.gene_to_idx and target_gene in self.gene_to_idx:
                source_idx = self.gene_to_idx[source_gene]
                target_idx = self.gene_to_idx[target_gene]
                valid_edges.append([source_idx, target_idx])
        
        # 转换为tensor
        if len(valid_edges) > 0:
            self.edge_index = torch.tensor(
                valid_edges, 
                dtype=torch.long
            ).t().contiguous()  # 转置到(2, n_edges)格式
        else:
            # 如果没有有效边，创建空tensor
            self.edge_index = torch.tensor(
                ([], []), 
                dtype=torch.long
            )
        
        print(f"[GRN Preprocessor] 原始边数: {n_original_edges}")
        print(f"[GRN Preprocessor] 有效边数: {self.edge_index.shape[1]}")
        
        return self.valid_genes, self.edge_index
    
    def subset_expression(
        self, 
        X: np.ndarray, 
        gene_names: List[str]
    ) -> np.ndarray:
        """
        从表达矩阵中提取有效基因子集
        
        Args:
            X: 表达矩阵 (n_cells, n_genes)
            gene_names: X对应的基因名列表
        
        Returns:
            X_subset: 子集矩阵 (n_cells, n_valid_genes)
        """
        # 创建临时映射
        gene_to_col = {gene: idx for idx, gene in enumerate(gene_names)}
        
        # 提取列
        col_indices = [gene_to_col[gene] for gene in self.valid_genes]
        X_subset = X[:, col_indices]
        
        return X_subset


class DataPreprocessor:
    """
    数据预处理器
    
    处理scRNA-seq表达矩阵的标准预处理步骤：
    - 库大小计算
    - Log1p归一化
    - Z-score标准化（用于OT计算）
    - 条件标签编码
    """
    
    def __init__(self):
        """初始化数据预处理器"""
        self.library_sizes = None
        self.condition_encoder = {}
        self.means = None
        self.stds = None
    
    def compute_library_sizes(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        计算库大小（每个细胞的总计数）
        
        model.txt中文库大小定义：l_i = Σ_j x_ij
        
        Args:
            X: 计数矩阵 (n_cells, n_genes)
        
        Returns:
            library_sizes: 向量(n_cells,)
        """
        self.library_sizes = np.sum(X, axis=1)
        return self.library_sizes
    
    def encode_conditions(
        self,
        conditions: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        将条件向量编码为独热向量
        
        Args:
            conditions: 条件标签数组 (n_cells,)，可以是字符串或整数
        
        Returns:
            (condition_onehot, encoder_dict)
            - condition_onehot: 独热编码 (n_cells, n_conditions)
            - encoder_dict: 标签到索引的映射
        """
        unique_conditions = np.unique(conditions)
        self.condition_encoder = {
            cond: idx for idx, cond in enumerate(unique_conditions)
        }
        
        # 转换为索引
        condition_indices = np.array([
            self.condition_encoder[cond] for cond in conditions
        ])
        
        # 独热编码
        n_cond = len(unique_conditions)
        condition_onehot = np.eye(n_cond)[condition_indices]
        
        return condition_onehot, self.condition_encoder
    
    def normalize_log1p(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Log1p归一化
        
        X_norm = log(1 + X)
        
        Args:
            X: 计数矩阵
        
        Returns:
            X_normalized
        """
        return np.log1p(X)
    
    def standardize_z_score(
        self,
        Z: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Z-score标准化（用于OT计算）
        
        details.txt第35-36行：在计算OT前，必须对Latent Z进行Z-score标准化
        确保不同维度的权重一致。
        
        Args:
            Z: 潜变量矩阵 (n_cells, n_latent)
            fit: 是否使用这个数据来拟合均值和标准差
        
        Returns:
            Z_standardized
        """
        if fit:
            self.means = np.mean(Z, axis=0)
            self.stds = np.std(Z, axis=0) + 1e-8  # 防止除以0
        
        if self.means is None or self.stds is None:
            raise ValueError("必须先fit数据或提供预计算的means和stds")
        
        Z_standardized = (Z - self.means) / self.stds
        return Z_standardized
    
    def inverse_standardize(
        self,
        Z_std: np.ndarray
    ) -> np.ndarray:
        """
        反向Z-score标准化
        
        Args:
            Z_std: 标准化后的潜变量
        
        Returns:
            Z_original: 原始潜变量
        """
        if self.means is None or self.stds is None:
            raise ValueError("没有保存的标准化参数")
        
        return Z_std * self.stds + self.means


class ConditionDataLoader:
    """
    条件数据加载器
    
    处理条件向量的编码和批处理。
    """
    
    @staticmethod
    def one_hot_encode(
        conditions: np.ndarray,
        n_conditions: int = None
    ) -> torch.Tensor:
        """
        独热编码条件向量
        
        Args:
            conditions: 条件索引 (batch_size,)，范围[0, n_conditions)
            n_conditions: 条件总数
        
        Returns:
            condition_onehot: (batch_size, n_conditions)
        """
        if n_conditions is None:
            n_conditions = int(np.max(conditions)) + 1
        
        condition_onehot = np.eye(n_conditions)[conditions]
        return torch.from_numpy(condition_onehot).float()
    
    @staticmethod
    def to_tensor(
        data: np.ndarray,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        将NumPy数组转换为PyTorch张量
        
        Args:
            data: NumPy数组
            dtype: 目标数据类型
        
        Returns:
            PyTorch张量
        """
        return torch.from_numpy(data).to(dtype)


def create_simple_grn(n_genes: int, density: float = 0.1) -> torch.Tensor:
    """
    创建用于测试的简单GRN
    
    生成一个随机稀疏图，用于单元测试和演示。
    
    Args:
        n_genes: 基因数量
        density: 边的密度（0-1）
    
    Returns:
        edge_index: (2, n_edges)的张量
    """
    n_edges = int(n_genes * n_genes * density)
    edge_list = []
    
    for _ in range(n_edges):
        source = np.random.randint(0, n_genes)
        target = np.random.randint(0, n_genes)
        if source != target:  # 避免自环
            edge_list.append([source, target])
    
    if len(edge_list) == 0:
        # 至少创建一些边
        edge_list = [[i, (i+1) % n_genes] for i in range(min(10, n_genes))]
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index
