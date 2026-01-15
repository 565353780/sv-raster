import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Union

from camera_control.Method.data import toTensor
from camera_control.Module.rgbd_camera import RGBDCamera

from octree_shape.Module.octree_builder import OctreeBuilder

import svraster_cuda
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils import octree_utils
from src.utils import activation_utils
from src.utils.marching_cubes_utils import torch_marching_cubes_grid

from sv_raster.Config.config import TrainerConfig, cfg
from sv_raster.Data.colmap_camera import ColmapCamera
from sv_raster.Method.io import loadMeshFile
from sv_raster.Method.path import createFileFolder
from sv_raster.Model.sparse_voxel import SparseVoxelModel


class Trainer:
    """
    SV-Raster 训练器类

    支持从mesh或点云初始化voxel，并使用RGBDCamera进行训练
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        """
        初始化训练器

        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        self.config = config if config is not None else deepcopy(cfg)
        self.config.finalize()

        # 设置设备
        torch.cuda.set_device(torch.device(self.config.device))

        # 设置随机种子
        self._seed_everything(self.config.procedure.seed)

        # 初始化模型
        self.voxel_model: Optional[SparseVoxelModel] = None

        # 训练相机列表
        self.train_cameras: List[ColmapCamera] = []
        self.test_cameras: List[ColmapCamera] = []

        # 场景边界
        self.bounding: Optional[torch.Tensor] = None

        # 初始化点云（用于初始化voxel）
        self.init_points: Optional[torch.Tensor] = None
        self.init_colors: Optional[torch.Tensor] = None

        # 训练状态
        self.current_iteration: int = 0
        self.optimizer = None
        self.scheduler = None

    def _seed_everything(self, seed: int) -> None:
        """设置所有随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def loadMeshFile(
        self,
        mesh_file_path: str,
        depth_max: int = 9,
    ) -> bool:
        """
        加载mesh并将其初始化为sparse voxel

        Args:
            mesh_file_path: mesh文件路径
            vox_level: voxel的octree层级

        Returns:
            是否成功
        """
        mesh = loadMeshFile(mesh_file_path)
        if mesh is None:
            return False

        octree_builder = OctreeBuilder(
            mesh=mesh,
            depth_max=depth_max,
            focus_center=[0, 0, 0],
            focus_length=1.0,
            normalize_scale=None,
            output_info=True,
        )

        voxel_centers = octree_builder.getDepthCenters(depth_max)

        self.init_points = toTensor(voxel_centers, torch.float32, 'cuda')
        self.init_colors = None

        # 计算边界
        self.bounding = toTensor([
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
        ], torch.float32, 'cuda')

        print(f"[INFO] Loaded mesh with {self.init_points.shape[0]} sampled points")
        return True

    def addCamera(
        self,
        camera: RGBDCamera,
    ) -> bool:
        idx = len(self.train_cameras)
        image_name = f"train_{idx:06d}"

        training_cam = ColmapCamera(camera, image_name=image_name)

        self.train_cameras.append(training_cam)
        return True

    def addCameras(
        self,
        cameras: List[RGBDCamera],
    ) -> bool:
        for cam in cameras:
            self.addCamera(cam)
        return True

    def _init_voxel_model(self, sfm_init_data=None) -> bool:
        """初始化voxel模型"""
        assert self.bounding is not None

        cfg = self.config

        # 创建voxel模型（使用cfg.model配置对象）
        from types import SimpleNamespace
        cfg_model = SimpleNamespace(
            vox_geo_mode=cfg.model.vox_geo_mode,
            density_mode=cfg.model.density_mode,
            sh_degree=cfg.model.sh_degree,
            ss=cfg.model.ss,
            outside_level=cfg.bounding.outside_level,
            model_path=cfg.output.model_path,
            white_background=cfg.model.white_background,
            black_background=cfg.model.black_background,
        )
        self.voxel_model = SparseVoxelModel(cfg_model)

        # 如果有初始化点，将其转换为sfm_init_data格式
        if self.init_points is not None and sfm_init_data is None:
            # 将init_points转换为SFMInitData格式
            from src.dataloader.reader_scene_info import SFMInitData
            points_xyz = self.init_points.cpu().numpy() if isinstance(self.init_points, torch.Tensor) else self.init_points
            sfm_init_data = SFMInitData(
                points_xyz=points_xyz,
                index_to_point_id=None,
                point_id_to_image_ids=None
            )
            print(f"[INFO] Converted {len(points_xyz)} init_points to SFMInitData format")
        
        # 统一使用model_init（从相机或点云初始化）
        cfg_init = SimpleNamespace(
            geo_init=cfg.init.geo_init,
            sh0_init=cfg.init.sh0_init,
            shs_init=cfg.init.shs_init,
            log_s_init=cfg.init.log_s_init,
            sh_degree_init=cfg.init.sh_degree_init,
            init_n_level=cfg.init.init_n_level,
            outside_mode=cfg.init.outside_mode,
            init_out_ratio=cfg.init.init_out_ratio,
            aabb_crop=cfg.init.aabb_crop,
            init_sparse_points=cfg.init.init_sparse_points,
        )
        self.voxel_model.model_init(
            bounding=self.bounding,
            cfg_init=cfg_init,
            cfg_mode=cfg.model.density_mode,
            cameras=self.train_cameras if len(self.train_cameras) > 0 else None,
            sfm_init=sfm_init_data,
        )

        print(f"[INFO] Initialized voxel model with {self.voxel_model.num_voxels} voxels")
        return True

    def _create_optimizer(self):
        """创建优化器（使用voxel_model的optimizer_init方法）"""
        cfg = self.config
        
        from types import SimpleNamespace
        cfg_optimizer = SimpleNamespace(
            geo_lr=cfg.optimizer.geo_lr,
            sh0_lr=cfg.optimizer.sh0_lr,
            shs_lr=cfg.optimizer.shs_lr,
            log_s_lr=cfg.optimizer.log_s_lr,
            optim_beta1=cfg.optimizer.optim_beta1,
            optim_beta2=cfg.optimizer.optim_beta2,
            optim_eps=cfg.optimizer.optim_eps,
        )
        self.voxel_model.optimizer_init(cfg_optimizer)
        
        return self.voxel_model.optimizer

    def _compute_iter_idx(self, n_cameras: int, n_iters: int) -> np.ndarray:
        """计算每次迭代使用的相机索引（使用compute_iter_idx_sparse）"""
        from src.dataloader.data_pack import compute_iter_idx_sparse
        return compute_iter_idx_sparse(n_cameras, n_iters, 1)

    def train(
        self,
        save_data_folder_path: str,
        verbose: bool = True,
    ) -> bool:
        """
        执行训练

        Args:
            verbose: 是否显示进度条

        Returns:
            是否成功
        """
        if len(self.train_cameras) == 0:
            print('[ERROR][Trainer::train]')
            print('\t No training cameras. Please add cameras first.')
            return False

        # 初始化模型（如果还没有）
        if self.voxel_model is None:
            # 如果有data_pack，获取sfm_init_data
            sfm_init_data = None
            if hasattr(self, 'data_pack') and self.data_pack is not None:
                sfm_init_data = self.data_pack.sfm_init_data
                # 保存initial_points用于points损失
                if hasattr(self.data_pack.sfm_init_data, 'points_xyz'):
                    self.initial_points = torch.from_numpy(self.data_pack.sfm_init_data.points_xyz).float().to("cuda")
                    print(f"point num = {self.initial_points.shape[0]}")
            
            if not self._init_voxel_model(sfm_init_data=sfm_init_data):
                return False

        cfg = self.config
        n_iter = cfg.procedure.n_iter

        # 初始化自动曝光
        if cfg.auto_exposure.enable:
            for cam in self.train_cameras:
                cam.auto_exposure_init()

        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率warmup
        first_iter = self.current_iteration + 1
        if first_iter <= cfg.optimizer.n_warmup:
            rate = max(first_iter - 1, 0) / cfg.optimizer.n_warmup
            for param_group in self.voxel_model.optimizer.param_groups:
                param_group["base_lr"] = param_group["lr"]
                param_group["lr"] = rate * param_group["base_lr"]

        # 初始化subdiv参数
        remain_subdiv_times = sum(
            (i >= first_iter)
            for i in range(
                cfg.procedure.subdivide_from, cfg.procedure.subdivide_until+1,
                cfg.procedure.subdivide_every
            )
        )
        subdivide_scale = cfg.procedure.subdivide_target_scale ** (1 / remain_subdiv_times) if remain_subdiv_times > 0 else 1.0
        subdivide_prop = max(0, (subdivide_scale - 1) / 7)
        print(f"Subdiv: times={remain_subdiv_times:2d} scale-each-time={subdivide_scale*100:.1f}% prop={subdivide_prop*100:.1f}%")

        # 计算相机索引
        tr_cam_indices = self._compute_iter_idx(len(self.train_cameras), n_iter)

        # 训练选项
        tr_render_opt = {
            'track_max_w': False,
            'lambda_R_concen': cfg.regularizer.lambda_R_concen,
            'output_T': False,
            'output_depth': False,
            'ss': 1.0,
            'rand_bg': cfg.regularizer.rand_bg,
            'use_auto_exposure': cfg.auto_exposure.enable,
        }

        # 初始化损失函数
        from src.utils import loss_utils
        sparse_depth_loss = loss_utils.SparseDepthLoss(
            iter_end=cfg.regularizer.sparse_depth_until)
        depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
            iter_from=cfg.regularizer.depthanythingv2_from,
            iter_end=cfg.regularizer.depthanythingv2_end,
            end_mult=cfg.regularizer.depthanythingv2_end_mult)
        mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
            iter_from=cfg.regularizer.mast3r_metric_depth_from,
            iter_end=cfg.regularizer.mast3r_metric_depth_end,
            end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
        nd_loss = loss_utils.NormalDepthConsistencyLoss(
            iter_from=cfg.regularizer.n_dmean_from,
            iter_end=cfg.regularizer.n_dmean_end,
            ks=cfg.regularizer.n_dmean_ks,
            tol_deg=cfg.regularizer.n_dmean_tol_deg)
        nmed_loss = loss_utils.NormalMedianConsistencyLoss(
            iter_from=cfg.regularizer.n_dmed_from,
            iter_end=cfg.regularizer.n_dmed_end)
        pi3_normal_loss = loss_utils.Pi3NormalLoss(
            iter_from=cfg.regularizer.pi3_normal_from,
            iter_end=cfg.regularizer.pi3_normal_end
        )

        # 初始化grid相关参数（用于正则化）
        max_voxel_level = self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level
        grid_voxel_coord = ((self.voxel_model.vox_center - self.voxel_model.vox_size * 0.5) - 
                          (self.voxel_model.scene_center - self.voxel_model.inside_extent * 0.5)) / \
                          self.voxel_model.inside_extent * (2**max_voxel_level)
        grid_voxel_size = (self.voxel_model.vox_size / self.voxel_model.inside_extent) * (2**max_voxel_level)
        
        # 保存initial_points（如果有）用于points损失
        self.initial_points = None  # 将在需要时从data_pack获取
        
        # 初始化log_s（如果使用SDF模式）
        if hasattr(self.voxel_model, '_log_s') and cfg.model.density_mode == 'sdf':
            device = self.voxel_model._log_s.device
            dtype = self.voxel_model._log_s.dtype
            learning_thickness = 2.0
            vox_size_min_inv = 1.0 / self.voxel_model.vox_size.min().item()
            vsmi = torch.as_tensor(vox_size_min_inv, device=device, dtype=dtype)
            init = 0.1 * torch.log(
                torch.log(torch.tensor(99.0, device=device, dtype=dtype)) * vsmi / learning_thickness / 2
            )
            with torch.no_grad():
                self.voxel_model._log_s.copy_(init)
            print(f"log_s init = {self.voxel_model._log_s.item():.9f}")

        # 训练循环
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        elapsed = 0

        ema_loss_for_log = 0.0
        ema_psnr_for_log = 0.0

        iter_rng = range(first_iter, n_iter + 1)

        if verbose:
            progress_bar = tqdm(iter_rng, desc="Training")

        for iteration in iter_rng:
            self.current_iteration = iteration

            iter_start.record()

            # 增加SH阶数
            if iteration % 1000 == 0:
                self.voxel_model.sh_degree_add1()

            # 重置SH
            if iteration in cfg.procedure.reset_sh_ckpt:
                print("Reset sh0 from cameras.")
                print("Reset shs to zero.")
                self.voxel_model.reset_sh_from_cameras(self.train_cameras)
                torch.cuda.empty_cache()

            # 超采样增强
            if iteration > 1000:
                if cfg.regularizer.ss_aug_max > 1:
                    tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
                elif 'ss' in tr_render_opt:
                    tr_render_opt.pop('ss')

            # 保存正则化lambda的最终值
            if not hasattr(cfg.regularizer, "_ge_final"):
                cfg.regularizer._ge_final = float(cfg.regularizer.lambda_ge_density)
            if not hasattr(cfg.regularizer, "_ls_final"):
                cfg.regularizer._ls_final = float(cfg.regularizer.lambda_ls_density)
            
            # 更新log_s学习率（在特定迭代）
            if iteration == 10000:
                for param_group in self.voxel_model.optimizer.param_groups:
                    if param_group.get('name') == 'log_s':
                        target_lr = cfg.optimizer.log_s_lr
                        param_group['lr'] = target_lr
                        print(f"\n[INFO] Iteration {iteration}: `log_s` learning rate changed to {target_lr}\n")
                        break
            
            # 增加log_s值（std_increase）
            first_prune = cfg.procedure.prune_from
            prune_every = cfg.procedure.prune_every
            std_increase_rate = 0.07 / (prune_every * 2)
            if 1 <= iteration < 10000:
                with torch.no_grad():
                    if hasattr(self.voxel_model, '_log_s'):
                        self.voxel_model._log_s.add_(std_increase_rate)
            
            if iteration % 100 == 0 and hasattr(self.voxel_model, '_log_s'):
                print(f"iteration {iteration} log_s = {self.voxel_model._log_s.item():.9f}")

            # 确定需要的输出
            need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
            need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
            need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
            need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
            need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
            need_pi3_normal_loss = cfg.regularizer.lambda_pi3_normal > 0 and pi3_normal_loss.is_active(iteration)
            
            tr_render_opt['output_T'] = (
                cfg.regularizer.lambda_T_concen > 0 or
                cfg.regularizer.lambda_T_inside > 0 or
                cfg.regularizer.lambda_mask > 0 or
                need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
            )
            tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss or need_pi3_normal_loss
            tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth

            if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
                tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

            if iteration >= cfg.regularizer.ascending_from and iteration <= cfg.regularizer.ascending_until and cfg.regularizer.lambda_ascending:
                tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

            # 更新自动曝光
            if cfg.auto_exposure.enable and iteration in cfg.procedure.auto_exposure_upd_ckpt:
                for cam in self.train_cameras:
                    with torch.no_grad():
                        ref = self.voxel_model.render(cam, ss=1.0)['color']
                    cam.auto_exposure_update(ref, cam.image.cuda())

            # 选择相机
            cam = self.train_cameras[tr_cam_indices[iteration - 1]]

            # 获取GT图像
            gt_image = cam.image.cuda()
            if cfg.regularizer.lambda_R_concen > 0:
                tr_render_opt['gt_color'] = gt_image

            # 渲染
            render_pkg = self.voxel_model.render(cam, **tr_render_opt)
            render_image = render_pkg['color']

            # 计算损失
            gt_image_modified = gt_image
            mse = loss_utils.l2_loss(render_image, gt_image_modified)

            if cfg.regularizer.use_l1:
                photo_loss = loss_utils.l1_loss(render_image, gt_image_modified)
            elif cfg.regularizer.use_huber:
                photo_loss = loss_utils.huber_loss(render_image, gt_image_modified, cfg.regularizer.huber_thres)
            else:
                photo_loss = mse

            loss_photo = cfg.regularizer.lambda_photo * photo_loss
            loss = loss_photo
            loss_dict = {"photo": loss_photo}

            # depthanythingv2 loss
            if need_depthanythingv2:
                loss_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)
                loss += loss_depthanythingv2
                loss_dict["depthanythingv2"] = loss_depthanythingv2

            if cfg.regularizer.lambda_ssim:
                loss_ssim = cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
                loss += loss_ssim
                loss_dict["ssim"] = loss_ssim

            if cfg.regularizer.lambda_T_concen:
                loss_T_concen = cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg['raw_T'])
                loss += loss_T_concen
                loss_dict["T_concen"] = loss_T_concen

            if cfg.regularizer.lambda_T_inside:
                loss_T_inside = cfg.regularizer.lambda_T_inside * render_pkg['raw_T'].square().mean()
                loss += loss_T_inside
                loss_dict["T_inside"] = loss_T_inside

            if need_nd_loss:
                loss_nd_loss = cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
                loss += loss_nd_loss
                loss_dict["nd_loss"] = loss_nd_loss

            if need_nmed_loss:
                loss_nmed_loss = cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)
                loss += loss_nmed_loss
                loss_dict["nmed_loss"] = loss_nmed_loss

            if need_pi3_normal_loss:
                lambda_pi3_mult = cfg.regularizer.pi3_normal_decay_mult ** (iteration // cfg.regularizer.pi3_normal_decay_every)
                loss_pi3_normal = cfg.regularizer.lambda_pi3_normal * lambda_pi3_mult * pi3_normal_loss(cam, render_pkg, iteration)
                loss += loss_pi3_normal
                loss_dict["pi3_normal_loss"] = loss_pi3_normal

            # 打印损失分解
            if iteration % 100 == 0:
                print(f"[iter {iteration}] loss breakdown:")
                for name, val in loss_dict.items():
                    v = val.item()
                    isn = torch.isnan(val)
                    print(f"   {name:15s}: {v:.6e}{'  <-- NaN !!' if isn else ''}")

            # 检查NaN
            if torch.isnan(loss):
                print(f"[iter {iteration}] 警告: NaN detected in TOTAL loss!")
                for name, val in loss_dict.items():
                    if torch.isnan(val):
                        print(f"   -> NaN found in {name}_loss")

            # 反向传播
            self.voxel_model.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 打印voxel统计信息
            if iteration % 100 == 0:
                with torch.no_grad():
                    nonleaf_mask = (~self.voxel_model.is_leaf.view(-1).bool())
                    levels = self.voxel_model.octlevel.view(-1).to(torch.int64)
                    levels_nonleaf = levels[nonleaf_mask]
                    if levels_nonleaf.numel() == 0:
                        print("  [voxels] non-leaf: 0")
                    else:
                        uniq_lvls, counts = torch.unique(levels_nonleaf, return_counts=True)
                        order = torch.argsort(uniq_lvls)
                        uniq_lvls = uniq_lvls[order].tolist()
                        counts = counts[order].tolist()
                        total_nonleaf = int(sum(counts))
                        print(f"  [voxels] non-leaf total: {total_nonleaf}")
                        for L, C in zip(uniq_lvls, counts):
                            print(f"    - level {L-cfg.bounding.outside_level}: {C}")

            # Grid-level正则化（在backward之后）
            self._apply_grid_regularization(iteration, grid_voxel_coord, grid_voxel_size, max_voxel_level)

            # 优化器步骤
            self.voxel_model.optimizer.step()

            # 学习率warmup
            if iteration <= cfg.optimizer.n_warmup:
                rate = iteration / cfg.optimizer.n_warmup
                for param_group in self.voxel_model.optimizer.param_groups:
                    param_group["lr"] = rate * param_group["base_lr"]

            # 特殊处理geo_lr（前100次迭代为0）
            for pg in self.voxel_model.optimizer.param_groups:
                if pg.get("name") == "_geo_grid_pts":
                    if iteration < 100:
                        pg["lr"] = 0.0
                        pg["base_lr"] = 0.0
                    elif iteration == 100:
                        val = cfg.optimizer.geo_lr
                        pg["lr"] = val
                        pg["base_lr"] = val

            # 学习率衰减
            if iteration in cfg.optimizer.lr_decay_ckpt:
                for param_group in self.voxel_model.optimizer.param_groups:
                    ori_lr = param_group["lr"]
                    param_group["lr"] *= cfg.optimizer.lr_decay_mult
                    print(f'LR decay of {param_group.get("name", "unknown")}: {ori_lr} => {param_group["lr"]}')
                cfg.regularizer.lambda_vg_density *= cfg.optimizer.lr_decay_mult
                cfg.regularizer.lambda_tv_density *= cfg.optimizer.lr_decay_mult
                cfg.regularizer.lambda_ge_density *= cfg.optimizer.lr_decay_mult
                cfg.regularizer.lambda_ls_density *= cfg.optimizer.lr_decay_mult

            # 梯度统计（用于subdivision）
            need_stat = (iteration >= 500 and iteration <= cfg.procedure.subdivide_until)
            if need_stat:
                self.voxel_model.subdiv_meta += self.voxel_model._subdiv_p.grad

            # 自适应剪枝和细分（注意：grid_voxel_coord和grid_voxel_size可能会被更新）
            updated_grid_voxel_coord, updated_grid_voxel_size, updated_max_voxel_level = \
                self._adaptive_voxels(iteration, subdivide_prop, remain_subdiv_times, grid_voxel_coord, grid_voxel_size, max_voxel_level)
            if updated_grid_voxel_coord is not None:
                grid_voxel_coord = updated_grid_voxel_coord
                grid_voxel_size = updated_grid_voxel_size
                max_voxel_level = updated_max_voxel_level

            iter_end.record()
            torch.cuda.synchronize()
            elapsed += iter_start.elapsed_time(iter_end)

            # 日志
            with torch.no_grad():
                loss_val = loss.item()
                psnr = -10 * np.log10(mse.item())

                ema_p = max(0.01, 1 / (iteration - first_iter + 1))
                ema_loss_for_log += ema_p * (loss_val - ema_loss_for_log)
                ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)

                if verbose and iteration % 10 == 0:
                    pb_text = {
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "psnr": f"{ema_psnr_for_log:.2f}",
                    }
                    progress_bar.set_postfix(pb_text)
                    progress_bar.update(10)

                # 保存检查点
                if iteration in cfg.output.checkpoint_iterations or iteration == n_iter:
                    self.save(iteration)

                if iteration % 1000 == 0:
                    render_image = self.render(
                        self.train_cameras[0].rgbd_camera,
                        output_T=True,
                        output_depth=True,
                        output_normal=True,
                    )
                    out_path = save_data_folder_path + f"/render/iter{iteration:06d}_cam0.png"

                    createFileFolder(out_path)
                    cv2.imwrite(out_path, render_image)

        if verbose:
            progress_bar.close()

        print(f"[INFO] Training completed. Final PSNR: {ema_psnr_for_log:.2f}")
        return True

    def _apply_grid_regularization(
        self, 
        iteration: int, 
        grid_voxel_coord: torch.Tensor,
        grid_voxel_size: torch.Tensor,
        max_voxel_level: int
    ) -> None:
        """应用grid级别的正则化（TV, VG, GE, LS, Points）"""
        cfg = self.config
        
        # TV正则化
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=self.voxel_model._geo_grid_pts,
                vox_key=self.voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=self.voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=self.voxel_model._geo_grid_pts.grad)
        
        # Voxel梯度正则化
        voxel_gradient_interval = iteration >= cfg.regularizer.vg_from and iteration <= cfg.regularizer.vg_until
        if cfg.regularizer.lambda_vg_density and voxel_gradient_interval:
            G = self.voxel_model.vox_size_inv.numel()
            K = int(G * (1.0 - float(cfg.regularizer.vg_drop_ratio)))
            active_list = torch.randperm(G, device=self.voxel_model.vox_key.device)[:K].to(torch.int32).contiguous()
            lambda_vg_mult = cfg.regularizer.vg_decay_mult ** (iteration // cfg.regularizer.vg_decay_every) * (G / K)
            svraster_cuda.grid_loss_bw.voxel_gradient(
                grid_pts=self.voxel_model._geo_grid_pts,
                vox_key=self.voxel_model.vox_key,
                vox_size_inv=self.voxel_model.vox_size_inv,
                active_list=active_list,
                weight=cfg.regularizer.lambda_vg_density * lambda_vg_mult,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.vg_sparse,
                grid_pts_grad=self.voxel_model._geo_grid_pts.grad)
        
        # Grid Eikonal正则化
        grid_eikonal_interval = iteration >= cfg.regularizer.ge_from and iteration <= cfg.regularizer.ge_until
        if cfg.regularizer.lambda_ge_density and grid_eikonal_interval:
            lambda_ge_mult = cfg.regularizer.ge_decay_mult ** min(iteration // cfg.regularizer.ge_decay_every, 2)
            G = self.voxel_model.grid_keys.numel()
            K = int(G * (1.0 - float(cfg.regularizer.ls_drop_ratio)))
            active_list = torch.randperm(G, device=self.voxel_model.grid_keys.device)[:K].to(torch.int32).contiguous()
            max_voxel_level = min(self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level, 9)
            vox_size_min_inv = 2**max_voxel_level / self.voxel_model.inside_extent
            svraster_cuda.grid_loss_bw.grid_eikonal(
                grid_pts=self.voxel_model._geo_grid_pts,
                vox_key=self.voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res=2**max_voxel_level,
                grid_mask=self.voxel_model.grid_mask,
                grid_keys=self.voxel_model.grid_keys,
                grid2voxel=self.voxel_model.grid2voxel,
                active_list=active_list,
                weight=cfg.regularizer.lambda_ge_density * lambda_ge_mult * (G / K),
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.ge_sparse,
                grid_pts_grad=self.voxel_model._geo_grid_pts.grad)
        
        # Laplacian平滑正则化
        laplacian_interval = iteration >= cfg.regularizer.ls_from and iteration <= cfg.regularizer.ls_until
        if cfg.regularizer.lambda_ls_density and laplacian_interval:
            lambda_ls_mult = cfg.regularizer.ls_decay_mult ** min(iteration // cfg.regularizer.ls_decay_every, 2)
            G = self.voxel_model.grid_keys.numel()
            K = int(G * (1.0 - float(cfg.regularizer.ls_drop_ratio)))
            active_list = torch.randperm(G, device=self.voxel_model.grid_keys.device)[:K].to(torch.int32).contiguous()
            max_voxel_level = min(self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level, 9)
            vox_size_min_inv = 2**max_voxel_level / self.voxel_model.inside_extent
            svraster_cuda.grid_loss_bw.laplacian_smoothness(
                grid_pts=self.voxel_model._geo_grid_pts,
                vox_key=self.voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res=2**max_voxel_level,
                grid_mask=self.voxel_model.grid_mask,
                grid_keys=self.voxel_model.grid_keys,
                grid2voxel=self.voxel_model.grid2voxel,
                active_list=active_list,
                weight=cfg.regularizer.lambda_ls_density * lambda_ls_mult * (G / K),
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.ls_sparse,
                grid_pts_grad=self.voxel_model._geo_grid_pts.grad)
        
        # Points损失（如果有初始点）
        if hasattr(self, 'initial_points') and self.initial_points is not None:
            points_interval = iteration >= cfg.regularizer.points_loss_from and cfg.regularizer.points_loss_until >= iteration
            if cfg.regularizer.lambda_points_density and points_interval:
                sample_rate = cfg.regularizer.points_sample_rate
                num_points = self.initial_points.shape[0]
                num_sample = max(1, int(num_points * sample_rate))
                idx = torch.randperm(num_points, device=self.initial_points.device)[:num_sample]
                sampled_points = self.initial_points[idx]
                points_in_grid = (sampled_points - (self.voxel_model.scene_center - self.voxel_model.inside_extent*0.5)) / \
                                self.voxel_model.inside_extent * (2**max_voxel_level)
                lambda_points_mult = cfg.regularizer.points_loss_decay_mult ** (iteration // cfg.regularizer.points_loss_decay_every)
                max_voxel_level = min(self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level, 9)
                vox_size_min_inv = 2**max_voxel_level / self.voxel_model.inside_extent
                svraster_cuda.grid_loss_bw.points_loss(
                    points_in_grid=points_in_grid,
                    grid_pts=self.voxel_model._geo_grid_pts,
                    vox_key=self.voxel_model.vox_key,
                    grid_voxel_coord=grid_voxel_coord,
                    grid_voxel_size=grid_voxel_size.view(-1),
                    grid_res=2**max_voxel_level,
                    grid_mask=self.voxel_model.grid_mask,
                    grid_keys=self.voxel_model.grid_keys,
                    grid2voxel=self.voxel_model.grid2voxel,
                    weight=cfg.regularizer.lambda_points_density * lambda_points_mult,
                    vox_size_inv=vox_size_min_inv,
                    no_tv_s=True,
                    tv_sparse=cfg.regularizer.points_loss_sparse,
                    grid_pts_grad=self.voxel_model._geo_grid_pts.grad)

    def _adaptive_voxels(
        self, 
        iteration: int,
        subdivide_prop: float,
        remain_subdiv_times: int,
        grid_voxel_coord: torch.Tensor,
        grid_voxel_size: torch.Tensor,
        max_voxel_level: int
    ) -> tuple:
        """自适应voxel剪枝和细分（匹配train.py的逻辑）"""
        cfg = self.config

        need_pruning = (
            iteration % cfg.procedure.prune_every == 0 and
            iteration >= cfg.procedure.prune_from and
            iteration <= cfg.procedure.prune_until
        )
        if iteration == 1:
            if cfg.procedure.prune_from == 0:
                need_pruning = True
        
        need_subdividing = (
            iteration % cfg.procedure.subdivide_every == 0 and
            iteration >= cfg.procedure.subdivide_from and
            iteration <= cfg.procedure.subdivide_until and
            self.voxel_model.num_voxels < cfg.procedure.subdivide_max_num
        )

        # 最后500次迭代不进行剪枝和细分
        need_pruning &= (iteration <= cfg.procedure.n_iter - 500)
        need_subdividing &= (iteration <= cfg.procedure.n_iter - 500)

        if need_pruning or need_subdividing:
            stat_pkg = self.voxel_model.compute_training_stat(camera_lst=self.train_cameras)
            torch.cuda.empty_cache()

        if need_pruning:
            ori_n = self.voxel_model.num_voxels

            # 计算剪枝阈值
            prune_all_iter = max(1, cfg.procedure.prune_until - cfg.procedure.prune_every)
            prune_now_iter = max(0, iteration - cfg.procedure.prune_every)
            prune_iter_rate = max(0, min(1, prune_now_iter / prune_all_iter))
            thres_inc = max(0, cfg.procedure.prune_thres_final - cfg.procedure.prune_thres_init)
            prune_thres = cfg.procedure.prune_thres_init + thres_inc * prune_iter_rate

            # 剪枝voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)
            
            # SDF模式的特殊剪枝逻辑
            if cfg.model.density_mode == 'sdf' and iteration >= 1000:
                sdf_vals = self.voxel_model._geo_grid_pts[self.voxel_model.vox_key]  # [N, 8, 1]
                signs = (sdf_vals > 0).float()
                has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)
                min_abs_sdf = sdf_vals.abs().min(dim=1).values.view(-1)
                global_vox_size_min = self.voxel_model.vox_size.min().item()
                sdf_thresh = torch.log(torch.tensor(199.0, device=self.voxel_model._log_s.device)) / torch.exp(10 * self.voxel_model._log_s)
                learning_thickness = sdf_thresh / 2 / global_vox_size_min
                print(f"true_learning_thickness = {learning_thickness:.4f}")
                sdf_thresh = max(2*global_vox_size_min, sdf_thresh.item())
                print(f"augmented_learning_thickness = {sdf_thresh/global_vox_size_min/2:.4f}")
                prune_mask = (~has_surface) & (min_abs_sdf > sdf_thresh)
            elif cfg.model.density_mode == 'sdf':
                sdf_vals = self.voxel_model._geo_grid_pts[self.voxel_model.vox_key]
                min_abs_sdf = sdf_vals.abs().min(dim=1).values.view(-1)
                global_vox_size_min = self.voxel_model.vox_size.min().item()
                sdf_thresh = torch.log(torch.tensor(199.0, device=self.voxel_model._log_s.device)) / torch.exp(10 * self.voxel_model._log_s)
                learning_thickness = sdf_thresh / 2 / global_vox_size_min
                print(f"true_learning_thickness = {learning_thickness:.4f}")
                sdf_thresh = max(2*global_vox_size_min, sdf_thresh.item())
                print(f"augmented_learning_thickness = {sdf_thresh/global_vox_size_min/2:.4f}")
                prune_mask = (min_abs_sdf > sdf_thresh)
            
            self.voxel_model.pruning(prune_mask)

            # 更新统计信息（用于后续细分）
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx]

            if hasattr(self.voxel_model, '_log_s'):
                print(f"voxel_model._log_s = {self.voxel_model._log_s}")

            new_n = self.voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')
            torch.cuda.empty_cache()

        if need_subdividing:
            ori_n = self.voxel_model.num_voxels

            # 排除一些voxels
            size_thres = stat_pkg['min_samp_interval'] * cfg.procedure.subdivide_samp_thres
            large_enough_mask = (self.voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest_mask = self.voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            if cfg.model.density_mode == 'sdf':
                non_finest_mask = self.voxel_model.octlevel.squeeze(1) < (svraster_cuda.meta.MAX_NUM_LEVELS - 5 - max(0, 3 - iteration // 2000) + cfg.bounding.outside_level)
                print(f"max octlevel for sdf: {svraster_cuda.meta.MAX_NUM_LEVELS - 2 - max(0, 3 - iteration // 3000)}")
            valid_mask = large_enough_mask & non_finest_mask

            # 获取细分优先级
            priority = self.voxel_model.subdiv_meta.squeeze(1) * valid_mask

            # 计算优先级排名
            rank = torch.zeros_like(priority)
            rank[priority.argsort()] = torch.arange(len(priority), dtype=torch.float32, device="cuda")

            # 确定要细分的voxel数量
            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = rank.quantile(1 - subdivide_prop)

            # 计算细分mask
            subdivide_mask = (rank > thres) & valid_mask
            outside_mask = ~self.voxel_model.inside_mask
            subdivide_mask_for_outside = subdivide_mask & outside_mask
            n_out = int(subdivide_mask_for_outside.sum().item())
            print(f"[outside subdivide] count = {n_out}")

            inside = self.voxel_model.inside_mask
            candidates = valid_mask & inside

            if iteration <= cfg.procedure.subdivide_all_until:
                subdivide_mask = candidates
            else:
                prio = self.voxel_model.subdiv_meta.squeeze(1)
                prio_inside = prio[candidates]
                if prio_inside.numel() == 0:
                    subdivide_mask = torch.zeros_like(candidates)
                else:
                    thres = torch.quantile(prio_inside, 1 - subdivide_prop)
                    subdivide_mask = candidates & (prio >= thres) | subdivide_mask_for_outside

            # SDF模式的特殊细分逻辑
            if cfg.model.density_mode == 'sdf' and iteration < 6000:
                with torch.no_grad():
                    sdf_vals = self.voxel_model._geo_grid_pts[self.voxel_model.vox_key]
                    signs = (sdf_vals > 0).float()
                    has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)
                cur_level = self.voxel_model.octlevel.squeeze(1)
                max_level = 9 + cfg.bounding.outside_level - max(0, 2 - iteration // 2000)
                under_level = cur_level < max_level
                valid_mask = has_surface & under_level
                subdivide_mask = (valid_mask & self.voxel_model.is_leaf.squeeze(1) & self.voxel_model.inside_mask) | (subdivide_mask_for_outside & valid_mask)
                if iteration <= cfg.procedure.subdivide_all_until:
                    subdivide_mask = under_level

            if hasattr(self.voxel_model, '_log_s'):
                print(f"voxel_model._log_s = {self.voxel_model._log_s}")

            # 如果voxel数量超过阈值，限制细分数量
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - self.voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed - 1])

            # 执行细分
            if subdivide_mask.sum() > 0:
                self.voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            
            new_n = self.voxel_model.num_voxels
            in_p = self.voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')
            max_voxel_level = self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level
            print(f'level max : {max_voxel_level}')
            self.voxel_model.subdiv_meta.zero_()  # 重置subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()

        # 更新grid相关参数（如果进行了剪枝或细分）
        if need_pruning or need_subdividing:
            max_voxel_level = min(self.voxel_model.octlevel.max().item() - cfg.bounding.outside_level, 9)
            grid_voxel_coord = ((self.voxel_model.vox_center - self.voxel_model.vox_size * 0.5) - 
                              (self.voxel_model.scene_center - self.voxel_model.inside_extent * 0.5)) / \
                              self.voxel_model.inside_extent * (2**max_voxel_level)
            grid_voxel_coord = torch.round(grid_voxel_coord).float()
            grid_voxel_size = (self.voxel_model.vox_size / self.voxel_model.inside_extent) * (2**max_voxel_level)
            grid_voxel_size = torch.round(grid_voxel_size).float()

            self.voxel_model.grid_mask, self.voxel_model.grid_keys, self.voxel_model.grid2voxel = \
                octree_utils.update_valid_gradient_table(
                    cfg.model.density_mode,
                    self.voxel_model.vox_center,
                    self.voxel_model.vox_size,
                    self.voxel_model.scene_center,
                    self.voxel_model.inside_extent,
                    max_voxel_level,
                    self.voxel_model.is_leaf
                )
            torch.cuda.synchronize()
            return grid_voxel_coord, grid_voxel_size, max_voxel_level
        
        return None, None, None

    def render(
        self,
        camera: Union[RGBDCamera, ColmapCamera],
        **kwargs,
    ) -> dict:
        """
        渲染单个相机视图

        Args:
            camera: 相机（RGBDCamera或ColmapCamera）
            **kwargs: 传递给voxel_model.render的额外参数

        Returns:
            渲染结果字典
        """
        if self.voxel_model is None:
            print('[ERROR][Trainer::render]')
            print('\t Model not initialized. Please train or load a model first.')
            return {}

        if isinstance(camera, RGBDCamera):
            camera = ColmapCamera(camera)

        render_pkg = self.voxel_model.render(camera, **kwargs)

        gt_rgb = camera.rgbd_camera.image_cv
        render_rgb = im_tensor2np(render_pkg['color'].detach().clone())
        render_alpha = im_tensor2np(1-render_pkg['T'].detach().clone())[...,None].repeat(3, axis=-1)
        render_depth_med = viz_tensordepth(render_pkg['depth'][2].detach().clone())
        render_depth = viz_tensordepth(render_pkg['depth'][0].detach().clone(), 1-render_pkg['T'][0].detach().clone())
        render_normal = im_tensor2np(render_pkg['normal'].detach().clone() * 0.5 + 0.5)

        imgs = [
            gt_rgb, render_rgb,
            render_alpha, render_depth_med,
            render_depth, render_normal,
        ]

        # 保证shape一致
        h, w = imgs[0].shape[:2]
        imgs = [cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img for img in imgs]

        # 拼成3x2的大图
        concated_image = np.concatenate(
            [
                np.concatenate(imgs[:2], axis=1),
                np.concatenate(imgs[2:4], axis=1),
                np.concatenate(imgs[4:6], axis=1),
            ], axis=0,
        )
        return concated_image

    def save(
        self,
        iteration: Optional[int] = None,
        path: Optional[str] = None,
    ) -> bool:
        """
        保存模型

        Args:
            iteration: 迭代次数（用于文件名）
            path: 保存路径，如果为None则使用配置中的路径

        Returns:
            是否成功
        """
        if self.voxel_model is None:
            print('[ERROR][Trainer::save]')
            print('\t No model to save.')
            return False

        if path is None:
            path = self.config.output.model_path

        os.makedirs(path, exist_ok=True)

        if iteration is None:
            iteration = self.current_iteration

        self.voxel_model.save_iteration(
            path,
            iteration,
            quantize=self.config.output.save_quantized)

        if self.config.output.save_optimizer and self.optimizer is not None:
            torch.save(
                {'optim': self.optimizer.state_dict(), 'sched': self.scheduler.state_dict()},
                os.path.join(path, "optim.pt"))

        print(f"[SAVE] path={self.voxel_model.latest_save_path}")
        return True

    def load(
        self,
        path: str,
        iteration: int = -1,
    ) -> bool:
        """
        加载模型

        Args:
            path: 模型路径
            iteration: 迭代次数，-1表示最新

        Returns:
            是否成功
        """
        cfg = self.config

        self.voxel_model = SparseVoxelModel(
            n_samp_per_vox=cfg.model.n_samp_per_vox,
            sh_degree=cfg.model.sh_degree,
            ss=cfg.model.ss,
            white_background=cfg.model.white_background,
            black_background=cfg.model.black_background,
        )

        loaded_iter = self.voxel_model.load_iteration(path, iteration)
        self.current_iteration = loaded_iter

        print(f"[LOAD] Loaded model from iteration {loaded_iter}")
        return True

    def evaluate(
        self,
        cameras: Optional[List[Union[RGBDCamera, ColmapCamera]]] = None,
    ) -> dict:
        """
        在测试集上评估模型

        Args:
            cameras: 测试相机列表，如果为None则使用self.test_cameras

        Returns:
            评估指标字典
        """
        if self.voxel_model is None:
            print('[ERROR][Trainer::evaluate]')
            print('\t Model not initialized.')
            return {}

        if cameras is None:
            cameras = self.test_cameras

        if len(cameras) == 0:
            print('[WARNING][Trainer::evaluate]')
            print('\t No test cameras available.')
            return {}

        from src.utils import loss_utils

        self.voxel_model.freeze_vox_geo()

        psnr_list = []
        ssim_list = []

        for camera in tqdm(cameras, desc="Evaluating"):
            if isinstance(camera, RGBDCamera):
                camera = ColmapCamera(camera)

            render_pkg = self.voxel_model.render(camera)
            render_image = render_pkg['color']

            if camera.image is not None:
                gt_image = camera.image.cuda()
                mse = loss_utils.l2_loss(render_image, gt_image).item()
                psnr = -10 * np.log10(mse)
                psnr_list.append(psnr)

                ssim = loss_utils.ssim_score(render_image, gt_image).item()
                ssim_list.append(ssim)

        self.voxel_model.unfreeze_vox_geo()

        results = {}
        if psnr_list:
            results['psnr'] = np.mean(psnr_list)
            results['ssim'] = np.mean(ssim_list)
            print(f"[EVAL] PSNR: {results['psnr']:.2f}, SSIM: {results['ssim']:.4f}")

        return results

    @torch.no_grad()
    def exportMeshFile(
        self,
        save_mesh_file_path: str,
        final_lv: int = 10,
        bbox_scale: float = 1.0,
        crop_bbox: Optional[torch.Tensor] = None,
        use_tsdf: bool = False,
        bandwidth_vox: float = 5.0,
        crop_border: float = 0.01,
        alpha_thres: float = 0.5,
        use_mean: bool = False,
        iso: float = 0.0,
    ) -> bool:
        """
        导出mesh文件

        Args:
            save_mesh_file_path: 保存mesh文件的路径
            final_lv: 用于marching cubes的最终层级
            bbox_scale: 边界框缩放因子
            crop_bbox: 裁剪边界框 [min_xyz, max_xyz]，如果为None则自动计算
            use_tsdf: 是否使用TSDF融合方法（更准确但更慢），False则使用直接marching cubes
            bandwidth_vox: TSDF截断带宽（以voxel大小为单位）
            crop_border: TSDF融合时裁剪边界比例
            alpha_thres: TSDF融合时alpha阈值
            use_mean: 是否使用平均深度（否则使用中值深度）
            iso: marching cubes的iso值

        Returns:
            是否成功
        """
        if self.voxel_model is None:
            print('[ERROR][Trainer::exportMeshFile]')
            print('\t Model not initialized. Please train or load a model first.')
            return False

        if len(self.train_cameras) == 0:
            print('[ERROR][Trainer::exportMeshFile]')
            print('\t No training cameras available.')
            return False

        # 冻结voxel几何
        self.voxel_model.freeze_vox_geo()

        if use_tsdf:
            # 使用TSDF融合方法
            mesh = self._extract_mesh_tsdf(
                final_lv=final_lv,
                bbox_scale=bbox_scale,
                crop_bbox=crop_bbox,
                bandwidth_vox=bandwidth_vox,
                crop_border=crop_border,
                alpha_thres=alpha_thres,
                use_mean=use_mean,
                iso=iso,
            )
        else:
            # 使用直接marching cubes方法
            mesh = self._extract_mesh_direct(
                final_lv=final_lv,
                bbox_scale=bbox_scale,
                crop_bbox=crop_bbox,
                iso=iso,
            )

        # 保存mesh
        createFileFolder(save_mesh_file_path)
        mesh.export(save_mesh_file_path)
        print(f'[INFO] Exported mesh to {save_mesh_file_path}')

        # 解冻voxel几何
        self.voxel_model.unfreeze_vox_geo()

        return True

    @torch.no_grad()
    def _extract_mesh_direct(
        self,
        final_lv: int,
        bbox_scale: float,
        crop_bbox: Optional[torch.Tensor],
        iso: float,
    ) -> trimesh.Trimesh:
        """使用直接marching cubes方法提取mesh"""
        # 过滤背景voxels
        if crop_bbox is None:
            inside_min = self.voxel_model.scene_center - 0.5 * self.voxel_model.inside_extent * bbox_scale
            inside_max = self.voxel_model.scene_center + 0.5 * self.voxel_model.inside_extent * bbox_scale
        else:
            inside_min = crop_bbox[0].cuda() if not crop_bbox[0].is_cuda else crop_bbox[0]
            inside_max = crop_bbox[1].cuda() if not crop_bbox[1].is_cuda else crop_bbox[1]

        inside_mask = ((inside_min <= self.voxel_model.grid_pts_xyz) & 
                      (self.voxel_model.grid_pts_xyz <= inside_max)).all(-1)
        inside_mask = inside_mask[self.voxel_model.vox_key].any(-1)
        inside_idx = torch.where(inside_mask)[0]

        # 推断iso值用于level set
        vox_level = torch.tensor([self.voxel_model.outside_level + final_lv], device="cuda")
        vox_size = octree_utils.level_2_vox_size(self.voxel_model.scene_extent, vox_level).item()
        iso_alpha = torch.tensor(0.5, device="cuda")
        iso_density = activation_utils.alpha2density(iso_alpha, vox_size)
        
        # 尝试获取density_mode，如果不存在则使用默认的softplus
        if hasattr(self.voxel_model, 'density_mode'):
            density_inverse = getattr(activation_utils, f"{self.voxel_model.density_mode}_inverse", 
                                    activation_utils.softplus_inverse)
        else:
            density_inverse = activation_utils.softplus_inverse
        
        iso_value = density_inverse(iso_density)
        sign = -1

        # 如果用户指定了iso值，则使用用户的值，否则使用计算出的iso_value
        final_iso = iso if iso != 0.0 else iso_value

        # 提取mesh
        verts, faces = torch_marching_cubes_grid(
            grid_pts_val=sign * self.voxel_model._geo_grid_pts,
            grid_pts_xyz=self.voxel_model.grid_pts_xyz,
            vox_key=self.voxel_model.vox_key[inside_idx],
            iso=sign * final_iso)

        mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
        return mesh

    @torch.no_grad()
    def _extract_mesh_tsdf(
        self,
        final_lv: int,
        bbox_scale: float,
        crop_bbox: Optional[torch.Tensor],
        bandwidth_vox: float,
        crop_border: float,
        alpha_thres: float,
        use_mean: bool,
        iso: float,
    ) -> trimesh.Trimesh:
        """使用TSDF融合方法提取mesh"""
        from src.utils.fuser_utils import Fuser

        # 渲染所有训练视图的深度和alpha
        depth_lst = []
        alpha_lst = []
        for cam in tqdm(self.train_cameras, desc="Render training views"):
            render_pkg = self.voxel_model.render(cam, output_depth=True, output_T=True)
            if use_mean:
                frame_depth = render_pkg['raw_depth'][[0]]  # 使用平均深度
            else:
                frame_depth = render_pkg['raw_depth'][[2]]  # 使用中值深度
            frame_alpha = 1 - render_pkg['raw_T']
            depth_lst.append(frame_depth)
            alpha_lst.append(frame_alpha)

        # 过滤背景voxels
        if crop_bbox is None:
            inside_min = self.voxel_model.scene_center - 0.5 * self.voxel_model.inside_extent * bbox_scale
            inside_max = self.voxel_model.scene_center + 0.5 * self.voxel_model.inside_extent * bbox_scale
        else:
            inside_min = crop_bbox[0].cuda() if not crop_bbox[0].is_cuda else crop_bbox[0]
            inside_max = crop_bbox[1].cuda() if not crop_bbox[1].is_cuda else crop_bbox[1]

        # 限制层级
        target_lv = self.voxel_model.outside_level + final_lv
        octpath, octlevel = octree_utils.clamp_level(
            self.voxel_model.octpath, self.voxel_model.octlevel, target_lv)

        # 从限制后的自适应稀疏voxels初始化
        vol = SparseVoxelModel(sh_degree=0)
        vol.octpath_init(
            self.voxel_model.scene_center,
            self.voxel_model.scene_extent,
            octpath,
            octlevel,
        )

        # 剪枝外部voxel
        gridpts_outside = ((vol.grid_pts_xyz < inside_min) | (vol.grid_pts_xyz > inside_max)).any(-1)
        corners_outside = gridpts_outside[vol.vox_key]
        prune_mask = corners_outside.all(-1)
        vol.pruning(prune_mask)

        # 确定带宽
        bandwidth = bandwidth_vox * vol.vox_size.min().item()

        # 运行TSDF融合
        print(f"Running TSDF fusion: #voxels={vol.num_voxels:9d} / band={bandwidth}")
        grid_tsdf = self._tsdf_fusion(
            self.train_cameras, depth_lst, alpha_lst,
            vol.grid_pts_xyz, bandwidth, crop_border, alpha_thres)

        # 从grid提取mesh
        verts, faces = torch_marching_cubes_grid(
            grid_pts_val=grid_tsdf,
            grid_pts_xyz=vol.grid_pts_xyz,
            vox_key=vol.vox_key,
            iso=iso)

        mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
        return mesh

    @torch.no_grad()
    def _tsdf_fusion(
        self,
        cam_lst: List[ColmapCamera],
        depth_lst: List[torch.Tensor],
        alpha_lst: List[torch.Tensor],
        grid_pts_xyz: torch.Tensor,
        trunc_dist: float,
        crop_border: float,
        alpha_thres: float,
    ) -> torch.Tensor:
        """执行TSDF融合"""
        from src.utils.fuser_utils import Fuser

        assert len(cam_lst) == len(depth_lst)
        assert len(cam_lst) == len(alpha_lst)

        fuser = Fuser(
            xyz=grid_pts_xyz,
            bandwidth=trunc_dist,
            use_trunc=True,
            fuse_tsdf=True,
            feat_dim=0,
            alpha_thres=alpha_thres,
            crop_border=crop_border,
            normal_weight=False,
            depth_weight=False,
            border_weight=False,
            use_half=False)

        for cam, frame_depth, frame_alpha in zip(tqdm(cam_lst), depth_lst, alpha_lst):
            frame_depth = frame_depth.cuda()
            frame_alpha = frame_alpha.cuda()
            fuser.integrate(cam, frame_depth, alpha=frame_alpha)

        tsdf = fuser.tsdf.squeeze(1).contiguous()
        return tsdf
