# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import time
import uuid
import datetime
import imageio
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dataclasses import replace
from typing import List, Optional, Union

import torch
import trimesh

from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Method.data import toNumpy, toTensor

from sv_raster.Config.config import TrainerConfig, cfg
from sv_raster.Method.io import loadMeshFile
from sv_raster.Model.sparse_voxel import SparseVoxelModel

import svraster_cuda


class TrainingCamera:
    """
    训练用相机包装类，将 RGBDCamera 适配到训练流程所需的接口
    """

    def __init__(
        self,
        rgbd_camera: RGBDCamera,
        image_name: str = "camera",
        near: float = 0.02,
    ) -> None:
        self.rgbd_camera = rgbd_camera
        self.image_name = image_name
        self.near = near

        # 缓存计算结果
        self._w2c = None
        self._c2w = None
        self._fovx = None
        self._fovy = None

        # 初始化图像数据
        self._setup_image_data()

    def _setup_image_data(self) -> None:
        """设置图像数据"""
        if self.rgbd_camera.image is not None:
            # 转换为 CHW 格式，范围 [0, 1]
            image = self.rgbd_camera.image
            if image.dim() == 3 and image.shape[-1] == 3:
                # HWC -> CHW
                image = image.permute(2, 0, 1)
            self.image = image.cpu()
        else:
            self.image = None

        # 深度和mask
        self.depth = self.rgbd_camera.depth.cpu() if self.rgbd_camera.depth is not None else None
        self.mask = None  # 可以根据需要从 valid_depth_mask 生成

        # 稀疏点
        self.sparse_pt = None

    @property
    def image_width(self) -> int:
        return self.rgbd_camera.width

    @property
    def image_height(self) -> int:
        return self.rgbd_camera.height

    @property
    def w2c(self) -> torch.Tensor:
        """世界到相机变换矩阵"""
        if self._w2c is None:
            # RGBDCamera 使用的坐标系：X右，Y上，Z后（相机看向-Z）
            # 需要转换为训练代码的坐标系
            self._w2c = self.rgbd_camera.world2camera.float().cuda()
        return self._w2c

    @property
    def c2w(self) -> torch.Tensor:
        """相机到世界变换矩阵"""
        if self._c2w is None:
            self._c2w = self.w2c.inverse().contiguous()
        return self._c2w

    @property
    def fovx(self) -> float:
        """水平视场角（弧度）"""
        if self._fovx is None:
            self._fovx = 2 * np.arctan(self.image_width / (2 * self.rgbd_camera.fx))
        return self._fovx

    @property
    def fovy(self) -> float:
        """垂直视场角（弧度）"""
        if self._fovy is None:
            self._fovy = 2 * np.arctan(self.image_height / (2 * self.rgbd_camera.fy))
        return self._fovy

    @property
    def tanfovx(self) -> float:
        return np.tan(self.fovx * 0.5)

    @property
    def tanfovy(self) -> float:
        return np.tan(self.fovy * 0.5)

    @property
    def cx_p(self) -> float:
        """归一化的主点x坐标"""
        return self.rgbd_camera.cx / self.image_width

    @property
    def cy_p(self) -> float:
        """归一化的主点y坐标"""
        return self.rgbd_camera.cy / self.image_height

    @property
    def cx(self) -> float:
        return self.rgbd_camera.cx

    @property
    def cy(self) -> float:
        return self.rgbd_camera.cy

    @property
    def lookat(self) -> torch.Tensor:
        return self.c2w[:3, 2]

    @property
    def position(self) -> torch.Tensor:
        return self.c2w[:3, 3]

    @property
    def pix_size(self) -> float:
        return 2 * self.tanfovx / self.image_width

    def to(self, device: str):
        """移动数据到指定设备"""
        if self.image is not None:
            self.image = self.image.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self

    def project(self, pts: torch.Tensor, return_depth: bool = False):
        """将3D点投影到图像平面"""
        cam_pts = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        depth = cam_pts[:, [2]]
        cam_uv = cam_pts[:, :2] / depth
        scale_x = 1 / self.tanfovx
        scale_y = 1 / self.tanfovy
        shift_x = 2 * self.cx_p - 1
        shift_y = 2 * self.cy_p - 1
        cam_uv[:, 0] = cam_uv[:, 0] * scale_x + shift_x
        cam_uv[:, 1] = cam_uv[:, 1] * scale_y + shift_y
        if return_depth:
            return cam_uv, depth
        return cam_uv

    def depth2pts(self, depth: torch.Tensor) -> torch.Tensor:
        """将深度图转换为3D点"""
        device = depth.device
        h, w = depth.shape[-2:]
        rd = self.compute_rd(wh=(w, h), device=device)
        return self.position.view(3, 1, 1).to(device) + rd * depth

    def compute_rd(self, wh=None, cxcy=None, device=None):
        """计算射线方向"""
        if wh is None:
            wh = (self.image_width, self.image_height)
        if cxcy is None:
            cxcy = (self.cx * wh[0] / self.image_width, self.cy * wh[1] / self.image_height)
        rd = svraster_cuda.utils.compute_rd(
            width=wh[0], height=wh[1],
            cx=cxcy[0], cy=cxcy[1],
            tanfovx=self.tanfovx, tanfovy=self.tanfovy,
            c2w_matrix=self.c2w.cuda())
        rd = rd.to(device if device is not None else self.c2w.device)
        return rd

    def depth2normal(self, depth: torch.Tensor, ks: int = 3, tol_cos: float = -1) -> torch.Tensor:
        """从深度图计算法线"""
        assert ks % 2 == 1
        pad = ks // 2
        ks_1 = ks - 1
        pts = self.depth2pts(depth)
        normal_pseudo = torch.zeros_like(pts)
        dx = pts[:, pad:-pad, ks_1:] - pts[:, pad:-pad, :-ks_1]
        dy = pts[:, ks_1:, pad:-pad] - pts[:, :-ks_1, pad:-pad]
        normal_pseudo[:, pad:-pad, pad:-pad] = torch.nn.functional.normalize(
            torch.cross(dx, dy, dim=0), dim=0)

        if tol_cos > 0:
            with torch.no_grad():
                pts_dir = torch.nn.functional.normalize(pts - self.position.view(3, 1, 1), dim=0)
                dot = (normal_pseudo * pts_dir).sum(0)
                mask = (dot > tol_cos)
            normal_pseudo = normal_pseudo * mask

        return normal_pseudo

    def auto_exposure_init(self):
        """初始化自动曝光"""
        self._exposure_A = torch.eye(3, dtype=torch.float32, device="cuda")
        self._exposure_t = torch.zeros([3, 1, 1], dtype=torch.float32, device="cuda")
        self.exposure_updated = False

    def auto_exposure_apply(self, image: torch.Tensor) -> torch.Tensor:
        """应用自动曝光"""
        if self.exposure_updated:
            image = torch.einsum('ij,jhw->ihw', self._exposure_A, image) + self._exposure_t
        return image

    def auto_exposure_update(self, ren: torch.Tensor, ref: torch.Tensor):
        """更新自动曝光参数"""
        self.exposure_updated = True
        self._exposure_A.requires_grad_()
        self._exposure_t.requires_grad_()
        optim = torch.optim.Adam([self._exposure_A, self._exposure_t], lr=1e-3)
        for _ in range(100):
            loss = (self.auto_exposure_apply(ren).clamp(0, 1) - ref).abs().mean()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        self._exposure_A.requires_grad_(False)
        self._exposure_t.requires_grad_(False)


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
        self.train_cameras: List[TrainingCamera] = []
        self.test_cameras: List[TrainingCamera] = []

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
        vox_level: int = 9,
        sample_density: float = 100.0,
    ) -> bool:
        """
        加载mesh并将其初始化为sparse voxel

        Args:
            mesh_file_path: mesh文件路径
            vox_level: voxel的octree层级
            sample_density: 每单位面积的采样点数

        Returns:
            是否成功
        """
        mesh = loadMeshFile(mesh_file_path)
        if mesh is None:
            return False

        # 从mesh表面采样点
        # 计算采样点数
        area = mesh.area
        n_samples = max(10000, int(area * sample_density))
        points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
        points = torch.tensor(points, dtype=torch.float32, device="cuda")

        # 获取颜色（如果有）
        if mesh.visual.vertex_colors is not None:
            # 从顶点颜色插值
            vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0
            face_vertices = mesh.faces[face_indices]
            colors = vertex_colors[face_vertices].mean(axis=1)
            colors = torch.tensor(colors, dtype=torch.float32, device="cuda")
        else:
            colors = torch.ones_like(points) * 0.5

        self.init_points = points
        self.init_colors = colors

        # 计算边界
        self._compute_bounding_from_points(points)

        print(f"[INFO] Loaded mesh with {len(points)} sampled points")
        return True

    def loadPcdFile(
        self,
        pcd_file_path: str,
        vox_level: int = 9,
    ) -> bool:
        """
        加载点云并将其初始化为sparse voxel

        Args:
            pcd_file_path: 点云文件路径
            vox_level: voxel的octree层级

        Returns:
            是否成功
        """
        if not os.path.exists(pcd_file_path):
            print(f'[ERROR][Trainer::loadPcdFile]')
            print(f'\t pcd file not exist!')
            print(f'\t pcd_file_path: {pcd_file_path}')
            return False

        pcd = trimesh.load(pcd_file_path)

        # 获取点坐标
        if hasattr(pcd, 'vertices'):
            points = np.array(pcd.vertices)
        elif hasattr(pcd, 'points'):
            points = np.array(pcd.points)
        else:
            print(f'[ERROR][Trainer::loadPcdFile]')
            print(f'\t Cannot extract points from file')
            return False

        points = torch.tensor(points, dtype=torch.float32, device="cuda")

        # 获取颜色（如果有）
        if hasattr(pcd, 'colors') and pcd.colors is not None:
            colors = np.array(pcd.colors)[:, :3]
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = torch.tensor(colors, dtype=torch.float32, device="cuda")
        elif hasattr(pcd, 'visual') and hasattr(pcd.visual, 'vertex_colors'):
            colors = np.array(pcd.visual.vertex_colors)[:, :3] / 255.0
            colors = torch.tensor(colors, dtype=torch.float32, device="cuda")
        else:
            colors = torch.ones_like(points) * 0.5

        self.init_points = points
        self.init_colors = colors

        # 计算边界
        self._compute_bounding_from_points(points)

        print(f"[INFO] Loaded point cloud with {len(points)} points")
        return True

    def _compute_bounding_from_points(self, points: torch.Tensor) -> None:
        """从点云计算边界"""
        min_bound = points.min(dim=0).values
        max_bound = points.max(dim=0).values

        # 添加一些padding
        extent = max_bound - min_bound
        padding = extent * 0.1
        min_bound = min_bound - padding
        max_bound = max_bound + padding

        self.bounding = torch.stack([min_bound, max_bound])

    def addCamera(
        self,
        camera: RGBDCamera,
        is_test: bool = False,
        image_name: Optional[str] = None,
    ) -> bool:
        """
        添加相机到训练/测试集

        Args:
            camera: RGBDCamera实例
            is_test: 是否为测试相机
            image_name: 图像名称

        Returns:
            是否成功
        """
        if image_name is None:
            idx = len(self.test_cameras if is_test else self.train_cameras)
            image_name = f"{'test' if is_test else 'train'}_{idx:04d}"

        training_cam = TrainingCamera(camera, image_name=image_name)

        if is_test:
            self.test_cameras.append(training_cam)
        else:
            self.train_cameras.append(training_cam)

        return True

    def addCameras(
        self,
        cameras: List[RGBDCamera],
        is_test: bool = False,
    ) -> bool:
        """
        批量添加相机

        Args:
            cameras: RGBDCamera列表
            is_test: 是否为测试相机

        Returns:
            是否成功
        """
        for i, cam in enumerate(cameras):
            self.addCamera(cam, is_test=is_test)
        return True

    def _init_voxel_model(self) -> bool:
        """初始化voxel模型"""
        if self.bounding is None:
            if len(self.train_cameras) > 0:
                # 从相机位置估计边界
                positions = torch.stack([cam.position for cam in self.train_cameras])
                self._compute_bounding_from_points(positions)
            else:
                print('[ERROR][Trainer::_init_voxel_model]')
                print('\t No bounding available. Please load mesh/pcd or add cameras first.')
                return False

        cfg = self.config

        # 创建voxel模型
        self.voxel_model = SparseVoxelModel(
            n_samp_per_vox=cfg.model.n_samp_per_vox,
            sh_degree=cfg.model.sh_degree,
            ss=cfg.model.ss,
            white_background=cfg.model.white_background,
            black_background=cfg.model.black_background,
        )

        # 如果有初始化点，使用points_init
        if self.init_points is not None:
            # 计算voxel大小
            extent = (self.bounding[1] - self.bounding[0]).max()
            vox_size = extent / (2 ** cfg.init.init_n_level)

            self.voxel_model.points_init(
                scene_center=(self.bounding[0] + self.bounding[1]) * 0.5,
                scene_extent=extent * (2 ** cfg.bounding.outside_level),
                xyz=self.init_points,
                expected_vox_size=vox_size,
                rgb=self.init_colors if self.init_colors is not None else cfg.init.sh0_init,
                density=cfg.init.geo_init,
            )
        else:
            # 使用model_init（从相机初始化）
            self.voxel_model.model_init(
                bounding=self.bounding,
                outside_level=cfg.bounding.outside_level,
                init_n_level=cfg.init.init_n_level,
                init_out_ratio=cfg.init.init_out_ratio,
                sh_degree_init=cfg.init.sh_degree_init,
                geo_init=cfg.init.geo_init,
                sh0_init=cfg.init.sh0_init,
                shs_init=cfg.init.shs_init,
                cameras=self.train_cameras if len(self.train_cameras) > 0 else None,
            )

        print(f"[INFO] Initialized voxel model with {self.voxel_model.num_voxels} voxels")
        return True

    def _create_optimizer(self):
        """创建优化器和调度器"""
        cfg = self.config

        optimizer = svraster_cuda.sparse_adam.SparseAdam(
            [
                {'params': [self.voxel_model._geo_grid_pts], 'lr': cfg.optimizer.geo_lr},
                {'params': [self.voxel_model._sh0], 'lr': cfg.optimizer.sh0_lr},
                {'params': [self.voxel_model._shs], 'lr': cfg.optimizer.shs_lr},
            ],
            betas=(cfg.optimizer.optim_beta1, cfg.optimizer.optim_beta2),
            eps=cfg.optimizer.optim_eps)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optimizer.lr_decay_ckpt,
            gamma=cfg.optimizer.lr_decay_mult)

        return optimizer, scheduler

    def _compute_iter_idx(self, n_cameras: int, n_iters: int) -> np.ndarray:
        """计算每次迭代使用的相机索引"""
        return np.random.randint(0, n_cameras, size=n_iters)

    def train(
        self,
        n_iter: Optional[int] = None,
        verbose: bool = True,
    ) -> bool:
        """
        执行训练

        Args:
            n_iter: 迭代次数，如果为None则使用配置中的值
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
            if not self._init_voxel_model():
                return False

        cfg = self.config
        if n_iter is None:
            n_iter = cfg.procedure.n_iter

        # 初始化自动曝光
        if cfg.auto_exposure.enable:
            for cam in self.train_cameras:
                cam.auto_exposure_init()

        # 创建优化器
        self.optimizer, self.scheduler = self._create_optimizer()

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
        nd_loss = loss_utils.NormalDepthConsistencyLoss(
            iter_from=cfg.regularizer.n_dmean_from,
            iter_end=cfg.regularizer.n_dmean_end,
            ks=cfg.regularizer.n_dmean_ks,
            tol_deg=cfg.regularizer.n_dmean_tol_deg)
        nmed_loss = loss_utils.NormalMedianConsistencyLoss(
            iter_from=cfg.regularizer.n_dmed_from,
            iter_end=cfg.regularizer.n_dmed_end)

        # 训练循环
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        elapsed = 0

        ema_loss_for_log = 0.0
        ema_psnr_for_log = 0.0

        first_iter = self.current_iteration + 1
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

            # 确定需要的输出
            need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
            need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
            need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)

            tr_render_opt['output_T'] = (
                cfg.regularizer.lambda_T_concen > 0 or
                cfg.regularizer.lambda_T_inside > 0 or
                cfg.regularizer.lambda_mask > 0 or
                need_sparse_depth or need_nd_loss
            )
            tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
            tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss

            if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
                tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

            if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
                tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

            # 更新自动曝光
            if cfg.auto_exposure.enable and iteration in cfg.auto_exposure.auto_exposure_upd_ckpt:
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
            mse = loss_utils.l2_loss(render_image, gt_image)

            if cfg.regularizer.use_l1:
                photo_loss = loss_utils.l1_loss(render_image, gt_image)
            elif cfg.regularizer.use_huber:
                photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
            else:
                photo_loss = mse

            loss = cfg.regularizer.lambda_photo * photo_loss

            if need_sparse_depth:
                loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

            if cfg.regularizer.lambda_mask and cam.mask is not None:
                gt_T = 1 - cam.mask.cuda()
                loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

            if cfg.regularizer.lambda_ssim:
                loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)

            if cfg.regularizer.lambda_T_concen:
                loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg['raw_T'])

            if cfg.regularizer.lambda_T_inside:
                loss += cfg.regularizer.lambda_T_inside * render_pkg['raw_T'].square().mean()

            if need_nd_loss:
                loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)

            if need_nmed_loss:
                loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # TV正则化
            if (cfg.regularizer.lambda_tv_density and
                iteration >= cfg.regularizer.tv_from and
                iteration <= cfg.regularizer.tv_until):
                self.voxel_model.apply_tv_on_density_field(cfg.regularizer.lambda_tv_density)

            # 优化器步骤
            self.optimizer.step()

            # 自适应剪枝和细分
            self._adaptive_voxels(iteration)

            # 更新学习率
            self.scheduler.step()

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

        if verbose:
            progress_bar.close()

        print(f"[INFO] Training completed. Final PSNR: {ema_psnr_for_log:.2f}")
        return True

    def _adaptive_voxels(self, iteration: int) -> None:
        """自适应voxel剪枝和细分"""
        cfg = self.config

        meet_adapt_period = (
            iteration % cfg.procedure.adapt_every == 0 and
            iteration >= cfg.procedure.adapt_from and
            iteration <= cfg.procedure.n_iter - 500
        )
        need_pruning = (
            meet_adapt_period and
            iteration <= cfg.procedure.prune_until
        )
        need_subdividing = (
            meet_adapt_period and
            iteration <= cfg.procedure.subdivide_until and
            self.voxel_model.num_voxels < cfg.procedure.subdivide_max_num
        )

        if not (need_pruning or need_subdividing):
            return

        # 计算voxel统计
        stat_pkg = self.voxel_model.compute_training_stat(camera_lst=self.train_cameras)
        scheduler_state = self.scheduler.state_dict()

        if need_pruning:
            ori_n = self.voxel_model.num_voxels

            prune_thres = np.interp(
                iteration,
                xp=[cfg.procedure.adapt_from, cfg.procedure.prune_until],
                fp=[cfg.procedure.prune_thres_init, cfg.procedure.prune_thres_final])

            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)
            self.voxel_model.pruning(prune_mask)

            new_n = self.voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')

        if need_subdividing:
            ori_n = self.voxel_model.num_voxels

            min_samp_interval = stat_pkg['min_samp_interval']
            if need_pruning:
                min_samp_interval = min_samp_interval[~prune_mask]

            size_thres = min_samp_interval * cfg.procedure.subdivide_samp_thres
            large_enough = (self.voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest = self.voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough & non_finest

            priority = self.voxel_model.subdivision_priority.squeeze(1) * valid_mask

            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = priority.quantile(1 - cfg.procedure.subdivide_prop)

            subdivide_mask = (priority > thres) & valid_mask

            max_n_subdiv = round((cfg.procedure.subdivide_max_num - self.voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (priority > priority[subdivide_mask].sort().values[n_removed - 1])

            self.voxel_model.subdividing(subdivide_mask)

            new_n = self.voxel_model.num_voxels
            in_p = self.voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            self.voxel_model.reset_subdivision_priority()

        # 重新创建优化器
        self.optimizer, self.scheduler = self._create_optimizer()
        self.scheduler.load_state_dict(scheduler_state)
        torch.cuda.empty_cache()

    def render(
        self,
        camera: Union[RGBDCamera, TrainingCamera],
        **kwargs,
    ) -> dict:
        """
        渲染单个相机视图

        Args:
            camera: 相机（RGBDCamera或TrainingCamera）
            **kwargs: 传递给voxel_model.render的额外参数

        Returns:
            渲染结果字典
        """
        if self.voxel_model is None:
            print('[ERROR][Trainer::render]')
            print('\t Model not initialized. Please train or load a model first.')
            return {}

        if isinstance(camera, RGBDCamera):
            camera = TrainingCamera(camera)

        return self.voxel_model.render(camera, **kwargs)

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
        cameras: Optional[List[Union[RGBDCamera, TrainingCamera]]] = None,
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
                camera = TrainingCamera(camera)

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
