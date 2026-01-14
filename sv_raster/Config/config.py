# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """模型配置"""
    n_samp_per_vox: int = 1              # 每个访问的voxel采样的点数
    sh_degree: int = 1                    # 使用 3 * (k+1)^2 个参数表示视角相关的颜色
    ss: float = 1.5                       # 超采样率，用于抗锯齿
    white_background: bool = True         # 假设白色背景
    black_background: bool = False        # 假设黑色背景


@dataclass
class BoundingConfig:
    """场景边界配置"""
    # 定义主要（内部）区域的边界框
    # 默认使用数据集提供的建议边界
    # 否则，自动从 forward 或 camera_median 模式中选择
    # 详见 src/utils/bounding_utils.py

    # default | camera_median | camera_max | forward | pcd
    bound_mode: str = "default"
    bound_scale: float = 1.0              # 边界的缩放因子
    forward_dist_scale: float = 1.0       # forward模式的距离缩放
    pcd_density_rate: float = 0.1         # pcd模式的密度率

    # 主要前景区域外的Octree层级数
    outside_level: int = 5


@dataclass
class OptimizerConfig:
    """优化器配置"""
    geo_lr: float = 0.025                 # 几何学习率
    sh0_lr: float = 0.010                 # SH0学习率
    shs_lr: float = 0.00025               # 高阶SH学习率

    optim_beta1: float = 0.1              # Adam beta1
    optim_beta2: float = 0.99             # Adam beta2
    optim_eps: float = 1e-15              # Adam epsilon

    lr_decay_ckpt: List[int] = field(default_factory=lambda: [19000])
    lr_decay_mult: float = 0.1


@dataclass
class RegularizerConfig:
    """正则化配置"""
    # 主要光度损失
    lambda_photo: float = 1.0
    use_l1: bool = False
    use_huber: bool = False
    huber_thres: float = 0.03

    # SSIM损失
    lambda_ssim: float = 0.02

    # 稀疏深度损失
    lambda_sparse_depth: float = 0.0
    sparse_depth_until: int = 10_000

    # Mask损失
    lambda_mask: float = 0.0

    # DepthAnything损失
    lambda_depthanythingv2: float = 0.0
    depthanythingv2_from: int = 3000
    depthanythingv2_end: int = 20000
    depthanythingv2_end_mult: float = 0.1

    # Mast3r度量深度损失
    lambda_mast3r_metric_depth: float = 0.0
    mast3r_repo_path: str = ''
    mast3r_metric_depth_from: int = 0
    mast3r_metric_depth_end: int = 20000
    mast3r_metric_depth_end_mult: float = 0.01

    # 最终透射率应集中到0或1
    lambda_T_concen: float = 0.0

    # 最终透射率应为0
    lambda_T_inside: float = 0.0

    # 每点RGB损失
    lambda_R_concen: float = 0.01

    # 几何正则化
    lambda_ascending: float = 0.0
    ascending_from: int = 0

    # 分布损失（鼓励分布在射线上集中）
    lambda_dist: float = 0.1
    dist_from: int = 10000

    # 渲染法线与期望深度导出法线的一致性损失
    lambda_normal_dmean: float = 0.0
    n_dmean_from: int = 10_000
    n_dmean_end: int = 20_000
    n_dmean_ks: int = 3
    n_dmean_tol_deg: float = 90.0

    # 渲染法线与中值深度导出法线的一致性损失
    lambda_normal_dmed: float = 0.0
    n_dmed_from: int = 3000
    n_dmed_end: int = 20_000

    # 密度网格的总变分损失
    lambda_tv_density: float = 1e-10
    tv_from: int = 0
    tv_until: int = 10000

    # 数据增强
    ss_aug_max: float = 1.5
    rand_bg: bool = False


@dataclass
class InitConfig:
    """初始化配置"""
    # Voxel属性初始化
    geo_init: float = -10.0               # 预激活密度初始值
    sh0_init: float = 0.5                 # Voxel颜色初始值，范围0~1
    shs_init: float = 0.0                 # 高阶SH系数初始值

    sh_degree_init: int = 3               # 初始激活的SH阶数

    # 通过密集voxel初始化主要内部区域
    init_n_level: int = 6                 # (2^6)^3 个voxels

    # 外部（背景区域）的voxel比例
    init_out_ratio: float = 2.0


@dataclass
class ProcedureConfig:
    """训练流程配置"""
    # 调度
    n_iter: int = 20_000                  # 总迭代次数
    sche_mult: float = 1.0                # 调度乘数
    seed: int = 3721                      # 随机种子

    # 重置SH
    reset_sh_ckpt: List[int] = field(default_factory=lambda: [-1])

    # 自适应通用设置
    adapt_from: int = 1000
    adapt_every: int = 1000

    # 自适应voxel剪枝
    prune_until: int = 18000
    prune_thres_init: float = 0.0001
    prune_thres_final: float = 0.05

    # 自适应voxel细分
    subdivide_until: int = 15000
    subdivide_all_until: int = 0
    subdivide_samp_thres: float = 1.0     # voxel最大采样率应大于此值
    subdivide_prop: float = 0.05
    subdivide_max_num: int = 10_000_000


@dataclass
class AutoExposureConfig:
    """自动曝光配置"""
    enable: bool = False
    auto_exposure_upd_ckpt: List[int] = field(default_factory=lambda: [5000, 10000, 15000])


@dataclass
class OutputConfig:
    """输出配置"""
    model_path: str = "./output"          # 模型保存路径
    save_quantized: bool = False          # 是否保存量化模型
    save_optimizer: bool = False          # 是否保存优化器状态

    # 测试和检查点
    test_iterations: List[int] = field(default_factory=lambda: [-1])
    checkpoint_iterations: List[int] = field(default_factory=list)
    pg_view_every: int = 200              # 进度视图保存频率


@dataclass
class TrainerConfig:
    """Trainer 完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    bounding: BoundingConfig = field(default_factory=BoundingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    init: InitConfig = field(default_factory=InitConfig)
    procedure: ProcedureConfig = field(default_factory=ProcedureConfig)
    auto_exposure: AutoExposureConfig = field(default_factory=AutoExposureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # 设备配置
    device: str = "cuda:0"

    def apply_schedule_multiplier(self) -> None:
        """应用调度乘数到相关配置"""
        if self.procedure.sche_mult == 1.0:
            return

        sche_mult = self.procedure.sche_mult

        # 调整学习率
        self.optimizer.geo_lr /= sche_mult
        self.optimizer.sh0_lr /= sche_mult
        self.optimizer.shs_lr /= sche_mult

        # 调整学习率衰减检查点
        self.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in self.optimizer.lr_decay_ckpt
        ]

        # 调整正则化相关迭代
        for key in ['dist_from', 'tv_from', 'tv_until',
                    'n_dmean_from', 'n_dmean_end',
                    'n_dmed_from', 'n_dmed_end',
                    'depthanythingv2_from', 'depthanythingv2_end',
                    'mast3r_metric_depth_from', 'mast3r_metric_depth_end']:
            setattr(self.regularizer, key, round(getattr(self.regularizer, key) * sche_mult))

        # 调整流程相关迭代
        for key in ['n_iter', 'adapt_from', 'adapt_every',
                    'prune_until', 'subdivide_until', 'subdivide_all_until']:
            setattr(self.procedure, key, round(getattr(self.procedure, key) * sche_mult))

        # 调整重置SH检查点
        self.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in self.procedure.reset_sh_ckpt
        ]

    def resolve_negative_iterations(self) -> None:
        """解析负数迭代次数（相对于总迭代次数）"""
        n_iter = self.procedure.n_iter

        self.output.test_iterations = [
            v + n_iter + 1 if v < 0 else v
            for v in self.output.test_iterations
        ]

        self.output.checkpoint_iterations = [
            v + n_iter + 1 if v < 0 else v
            for v in self.output.checkpoint_iterations
        ]

    def finalize(self) -> None:
        """完成配置（应用所有调整）"""
        self.apply_schedule_multiplier()
        self.resolve_negative_iterations()


# 创建默认全局配置实例
cfg = TrainerConfig()
