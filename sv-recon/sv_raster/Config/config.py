from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置"""
    vox_geo_mode: str = "triinterp1"      # Voxel几何模式
    density_mode: str = "exp_linear_11"  # 密度表示模式
    sh_degree: int = 3                    # 使用 3 * (k+1)^2 个参数表示视角相关的颜色
    ss: float = 1.5                       # 超采样率，用于抗锯齿
    outside_level: int = 5                # 主要3D区域外的Octree层级数
    model_path: str = ""                  # 模型保存路径
    white_background: bool = False        # 假设白色背景
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
    log_s_lr: float = 0.001               # log_s学习率

    optim_beta1: float = 0.1              # Adam beta1
    optim_beta2: float = 0.99             # Adam beta2
    optim_eps: float = 1e-15              # Adam epsilon

    n_warmup: int = 100                   # 学习率warmup迭代次数

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
    mast3r_metric_depth_from: int = 3000
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
    ascending_until: int = 2000

    # 分布损失（鼓励分布在射线上集中）
    lambda_dist: float = 0.1
    dist_from: int = 10000

    # 渲染法线与期望深度导出法线的一致性损失
    lambda_normal_dmean: float = 0.0
    n_dmean_from: int = 2_000
    n_dmean_end: int = 20_000
    n_dmean_ks: int = 3
    n_dmean_tol_deg: float = 90.0

    # 渲染法线与中值深度导出法线的一致性损失
    lambda_normal_dmed: float = 0.0
    n_dmed_from: int = 2000
    n_dmed_end: int = 20_000

    # Pi3法线损失
    lambda_pi3_normal: float = 0.1
    pi3_normal_from: int = 0
    pi3_normal_end: int = 4000
    pi3_normal_decay_every: int = 2000
    pi3_normal_decay_mult: float = 1.0

    # 密度网格的总变分损失
    lambda_tv_density: float = 1e-8
    tv_from: int = 0
    tv_until: int = 10000
    tv_decay_every: int = 1000
    tv_decay_mult: float = 0.8
    tv_sparse: bool = False

    # Voxel梯度损失
    lambda_vg_density: float = 1e-11
    vg_from: int = 6000
    vg_until: int = 8000
    vg_decay_every: int = 2000
    vg_decay_mult: float = 0.25
    vg_sparse: bool = False
    vg_drop_ratio: float = 0.5

    # 网格Eikonal损失
    lambda_ge_density: float = 2e-8
    ge_from: int = 0
    ge_until: int = 6000
    ge_decay_every: int = 2000
    ge_decay_mult: float = 0.25
    ge_sparse: bool = False
    ge_drop_ratio: float = 0.0

    # Laplacian平滑损失
    lambda_ls_density: float = 1e-10
    ls_from: int = 0
    ls_until: int = 8000
    ls_decay_every: int = 2000
    ls_decay_mult: float = 0.25
    ls_sparse: bool = False
    ls_drop_ratio: float = 0.0

    # 点损失
    lambda_points_density: float = 0.0
    points_loss_from: int = 0
    points_loss_until: int = 4000
    points_loss_decay_every: int = 2000
    points_loss_decay_mult: float = 1.0
    points_loss_sparse: bool = False
    points_sample_rate: float = 0.05

    # 数据增强
    ss_aug_max: float = 1
    rand_bg: bool = False


@dataclass
class InitConfig:
    """初始化配置"""
    # Voxel属性初始化
    geo_init: float = -10.0               # 预激活密度初始值
    sh0_init: float = 0.5                 # Voxel颜色初始值，范围0~1
    shs_init: float = 0.0                 # 高阶SH系数初始值
    log_s_init: float = 0.3               # log_s初始值

    sh_degree_init: int = 3               # 初始激活的SH阶数

    # 通过密集voxel初始化主要内部区域
    init_n_level: int = 6                 # (2^6)^3 个voxels

    # 外部（背景区域）的初始化策略
    # none: 外部区域无voxels
    # uniform[N]: 每个shell层级细分N次，例如uniform1, uniform2, uniform3
    # heuristic: 基于init_out_ratio初始化固定数量的voxels
    outside_mode: str = "heuristic"
    init_out_ratio: float = 2.0

    # 如果给定则应用aabb裁剪
    aabb_crop: bool = False

    # 是否初始化稀疏点
    init_sparse_points: bool = True


@dataclass
class ProcedureConfig:
    """训练流程配置"""
    # 调度
    n_iter: int = 10_000                  # 总迭代次数
    sche_mult: float = 1.0                # 调度乘数
    seed: int = 3721                      # 随机种子

    # 重置SH
    reset_sh_ckpt: List[int] = field(default_factory=lambda: [-1])

    # 自适应voxel剪枝
    prune_from: int = 1000
    prune_every: int = 1000
    prune_until: int = 18000
    prune_thres_init: float = 0.0001
    prune_thres_final: float = 0.03

    # 自适应voxel细分
    subdivide_from: int = 1000
    subdivide_every: int = 1000
    subdivide_until: int = 9000
    subdivide_samp_thres: float = 1.0     # voxel最大采样率应大于此值
    subdivide_target_scale: float = 90.0   # 细分目标缩放
    subdivide_max_num: int = 10_000_000
    subdivide_all_until: int = 0          # 在此迭代之前细分所有有效voxels
    subdivide_save_gpu: bool = False      # 细分时是否节省GPU内存


@dataclass
class AutoExposureConfig:
    """自动曝光配置"""
    enable: bool = False
    auto_exposure_upd_ckpt: List[int] = field(default_factory=lambda: [5000, 10000, 15000])


@dataclass
class DataConfig:
    """数据配置"""
    source_path: str = ""
    images: str = "images"
    res_downscale: float = 0.0
    res_width: int = 0
    extension: str = ".png"
    blend_mask: bool = True
    depth_paths: str = ""
    normal_paths: str = ""
    depth_scale: float = 1.0
    data_device: str = "cpu"
    eval: bool = False
    test_every: int = 8


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
    data: DataConfig = field(default_factory=DataConfig)
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
        for key in ['n_iter', 'prune_from', 'prune_every', 'prune_until',
                    'subdivide_from', 'subdivide_every', 'subdivide_until', 'subdivide_all_until']:
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
