"""
推理模块

实现details2.txt缺口四：使用torchdiffeq进行ODE求解和轨迹生成。

推理流程：
1. 从初始z_start和条件c开始
2. 使用训练好的向量场v_θ进行ODE积分
3. 解码潜变量回基因表达空间
4. 返回轨迹和预测的基因表达动力学
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Callable
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    ODE函数包装器
    
    适配torchdiffeq的func(t, y)接口。
    """
    
    def __init__(
        self,
        vector_field: nn.Module,
        condition: torch.Tensor
    ):
        """
        初始化ODE函数
        
        Args:
            vector_field: 训练好的向量场模型
            condition: 固定的条件向量 (1, n_cond)
        """
        super().__init__()
        self.vector_field = vector_field
        self.condition = condition
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        ODE函数：dz/dt = v_θ(z, t, c)
        
        Args:
            t: 时间标量
            z: 当前潜变量 (batch_size, n_latent)
        
        Returns:
            dz/dt: 速度向量 (batch_size, n_latent)
        """
        # torchdiffeq传入的t是标量，需要广播到batch size
        t_batch = t.repeat(z.shape[0])
        
        # 调用向量场
        v = self.vector_field(t_batch, z, self.condition)
        
        return v


class ODEIntegrator:
    """
    ODE积分器
    
    details2.txt缺口四的完整实现（第132-181行）。
    
    使用torchdiffeq.odeint进行ODE求解，生成细胞轨迹。
    """
    
    def __init__(
        self,
        model: nn.Module,
        solver: str = 'dopri5',
        device: str = 'cpu'
    ):
        """
        初始化ODE积分器
        
        Args:
            model: CausalGenoFlow模型
            solver: ODE求解器类型 ('dopri5', 'adams', 'euler', 等)
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model
        self.solver = solver
        self.device = device
    
    def simulate_trajectory(
        self,
        z_initial: torch.Tensor,
        condition: torch.Tensor,
        t_span: torch.Tensor = None,
        library_size: torch.Tensor = None,
        rtol: float = 1e-5,
        atol: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模拟单细胞轨迹
        
        使用torchdiffeq对向量场进行积分，生成从z_initial开始的轨迹。
        
        Args:
            z_initial: 初始潜变量 (batch_size, n_latent)
            condition: 条件向量 (batch_size, n_cond) 或 (1, n_cond)
            t_span: 时间轴 (n_timepoints,)，默认为linspace(0, 1, 100)
            library_size: 库大小 (batch_size,)，默认为1.0
            rtol: ODE求解相对容差
            atol: ODE求解绝对容差
        
        Returns:
            (traj_z, traj_mean, trajectory_info)
            
            其中：
            - traj_z: 潜空间轨迹 (n_timepoints, batch_size, n_latent)
            - traj_mean: 基因表达预测 (n_timepoints, batch_size, n_genes)
            - trajectory_info: 包含元数据的字典
        
        实现对应关系：
        - details2.txt第145-181行：完整代码
        - model.txt第3章节：流匹配和ODE概念
        """
        # 确保张量在正确的设备上
        z_initial = z_initial.to(self.device)
        condition = condition.to(self.device)
        
        # 默认时间轴
        if t_span is None:
            t_span = torch.linspace(0, 1, steps=100, device=self.device)
        else:
            t_span = t_span.to(self.device)
        
        # 默认库大小
        if library_size is None:
            library_size = torch.ones(z_initial.shape[0], device=self.device) * 10000.0
        else:
            library_size = library_size.to(self.device)
        
        # 广播条件向量到正确的batch size
        if condition.shape[0] == 1 and z_initial.shape[0] > 1:
            condition = condition.repeat(z_initial.shape[0], 1)
        
        # 创建ODE函数
        ode_func = ODEFunc(self.model.vector_field, condition)
        ode_func = ode_func.to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # ODE求解
        with torch.no_grad():
            traj_z = odeint(
                ode_func,
                z_initial,
                t_span,
                method=self.solver,
                rtol=rtol,
                atol=atol
            )
        
        # traj_z: (n_timepoints, batch_size, n_latent)
        
        # 解码回基因表达空间
        # 形状变换用于batch处理
        n_timepoints = traj_z.shape[0]
        batch_size = traj_z.shape[1]
        
        # 展平：(n_timepoints*batch_size, n_latent)
        traj_z_flat = traj_z.reshape(-1, traj_z.shape[-1])
        
        # 广播条件和库大小
        condition_rep = condition.repeat(n_timepoints, 1)
        library_size_rep = library_size.repeat(n_timepoints)
        
        # 解码
        with torch.no_grad():
            mean_flat, theta_flat = self.model.decode(
                traj_z_flat,
                condition_rep,
                library_size_rep
            )
        
        # 重塑回轨迹形式
        traj_mean = mean_flat.reshape(n_timepoints, batch_size, -1)
        
        # 准备返回信息
        trajectory_info = {
            't_span': t_span.cpu().numpy(),
            'z_initial': z_initial.cpu().numpy(),
            'condition': condition.cpu().numpy(),
            'library_size': library_size.cpu().numpy(),
            'solver': self.solver,
            'rtol': rtol,
            'atol': atol
        }
        
        return traj_z, traj_mean, trajectory_info
    
    def batch_simulate_trajectories(
        self,
        z_initials: torch.Tensor,
        conditions: torch.Tensor,
        t_span: torch.Tensor = None,
        library_sizes: torch.Tensor = None,
        batch_size: int = 10
    ) -> Tuple[list, list, list]:
        """
        批量模拟多条轨迹
        
        Args:
            z_initials: 多个初始点 (n_trajectories, n_latent)
            conditions: 对应的条件向量 (n_trajectories, n_cond)
            t_span: 时间轴
            library_sizes: 库大小
            batch_size: 每批处理的轨迹数
        
        Returns:
            (all_traj_z, all_traj_mean, all_info_list)
        """
        n_trajectories = z_initials.shape[0]
        all_traj_z = []
        all_traj_mean = []
        all_info_list = []
        
        for i in range(0, n_trajectories, batch_size):
            end_idx = min(i + batch_size, n_trajectories)
            
            z_batch = z_initials[i:end_idx]
            c_batch = conditions[i:end_idx]
            
            if library_sizes is not None:
                l_batch = library_sizes[i:end_idx]
            else:
                l_batch = None
            
            traj_z, traj_mean, info = self.simulate_trajectory(
                z_batch, c_batch, t_span, l_batch
            )
            
            all_traj_z.append(traj_z)
            all_traj_mean.append(traj_mean)
            all_info_list.append(info)
        
        return all_traj_z, all_traj_mean, all_info_list
    
    def counterfactual_simulation(
        self,
        z_initial: torch.Tensor,
        condition_original: torch.Tensor,
        condition_counterfactual: torch.Tensor,
        t_span: torch.Tensor = None,
        library_size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        反事实模拟
        
        说明.txt中提到的"虚拟临床试验"功能：
        对比同一个细胞在不同条件下的演化轨迹。
        
        Args:
            z_initial: 初始潜变量
            condition_original: 原始条件
            condition_counterfactual: 反事实条件
            t_span: 时间轴
            library_size: 库大小
        
        Returns:
            (traj_z_orig, traj_mean_orig, traj_z_cf, traj_mean_cf)
            对应原始轨迹和反事实轨迹
        """
        # 原始条件下的轨迹
        traj_z_orig, traj_mean_orig, _ = self.simulate_trajectory(
            z_initial, condition_original, t_span, library_size
        )
        
        # 反事实条件下的轨迹
        traj_z_cf, traj_mean_cf, _ = self.simulate_trajectory(
            z_initial, condition_counterfactual, t_span, library_size
        )
        
        return traj_z_orig, traj_mean_orig, traj_z_cf, traj_mean_cf
    
    def sensitivity_analysis(
        self,
        z_initial: torch.Tensor,
        condition: torch.Tensor,
        perturbation_dim: int,
        perturbation_values: np.ndarray,
        t_span: torch.Tensor = None,
        library_size: torch.Tensor = None
    ) -> list:
        """
        敏感性分析
        
        研究向量场对特定维度的敏感性。
        
        Args:
            z_initial: 初始潜变量
            condition: 条件向量
            perturbation_dim: 要扰动的潜维度索引
            perturbation_values: 扰动值的数组
            t_span: 时间轴
            library_size: 库大小
        
        Returns:
            results: 列表，每个元素是扰动下的轨迹元组
        """
        results = []
        
        for perturb_value in perturbation_values:
            # 创建扰动的初始状态
            z_perturbed = z_initial.clone()
            z_perturbed[0, perturbation_dim] += perturb_value
            
            traj_z, traj_mean, info = self.simulate_trajectory(
                z_perturbed, condition, t_span, library_size
            )
            
            results.append({
                'perturbation': perturb_value,
                'traj_z': traj_z,
                'traj_mean': traj_mean,
                'info': info
            })
        
        return results
