"""
Causal-GenoFlow 包

用于单细胞基因表达动力学的因果生成式建模框架。

核心组件：
- modules: 神经网络模块
- losses: 损失函数
- data: 数据预处理
- trainer: 两阶段训练
- inference: ODE推理和轨迹生成

使用流程：
1. 数据预处理：GRNPreprocessor.align_genes() 和 DataPreprocessor
2. 构建模型：CausalGenoFlow()
3. 两阶段训练：TwoStageTrainer.phase1_train() 和 phase2_train()
4. 推理和分析：ODEIntegrator.simulate_trajectory()
"""

from .modules import (
    NBDecoder,
    CausalEncoder,
    TissueDiscriminator,
    CorrectedGNNVectorField,
    CausalGenoFlow
)

from .losses import (
    NBLoss,
    KLDivergenceLoss,
    FlowMatchingLoss,
    AdversarialLoss,
    CombinedVAELoss
)

from .data import (
    GRNPreprocessor,
    DataPreprocessor,
    ConditionDataLoader,
    create_simple_grn
)

from .trainer import TwoStageTrainer

from .inference import (
    ODEFunc,
    ODEIntegrator
)

__version__ = "1.0.0"

__all__ = [
    # Modules
    'NBDecoder',
    'CausalEncoder',
    'TissueDiscriminator',
    'CorrectedGNNVectorField',
    'CausalGenoFlow',
    # Losses
    'NBLoss',
    'KLDivergenceLoss',
    'FlowMatchingLoss',
    'AdversarialLoss',
    'CombinedVAELoss',
    # Data
    'GRNPreprocessor',
    'DataPreprocessor',
    'ConditionDataLoader',
    'create_simple_grn',
    # Training
    'TwoStageTrainer',
    # Inference
    'ODEFunc',
    'ODEIntegrator'
]
