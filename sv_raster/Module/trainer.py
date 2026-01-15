import os
import cv2
import torch
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

import trimesh


def demo():
    octree_builder = OctreeBuilder(
        mesh_file_path,
        depth_max=8,
        focus_center=[0, 0, 0],
        focus_length=1.0,
        normalize_scale=0.99,
        output_info=True,
    )

    leaf_num = octree_builder.leafNum
    shape_code = octree_builder.getShapeCode()

    print("shape leaf num:", leaf_num)
    print("shape code size:", len(shape_code))

    octree_builder.loadShapeCode(shape_code)

    leaf_num = octree_builder.leafNum
    shape_code = octree_builder.getShapeCode()

    print("shape leaf num:", leaf_num)
    print("shape code size:", len(shape_code))

    octree_builder.renderLeaf()

    for depth in range(1, depth_max + 1):
        octree_builder.renderDepth(depth)

    occ = octree_builder.getDepthOcc(8)
    print("occ shape:", occ.shape)
    octree_builder.renderDepthOcc(8)
    return True


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

    def _init_voxel_model(self) -> bool:
        """初始化voxel模型"""
        assert self.bounding is not None

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
            if not self._init_voxel_model():
                return False

        cfg = self.config
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

            # 检查min_samp_interval是否为空或大小不匹配
            if min_samp_interval.numel() == 0:
                print(f'[WARNING] min_samp_interval is empty after pruning, skipping subdivision')
            elif min_samp_interval.shape[0] != self.voxel_model.num_voxels:
                print(f'[WARNING] min_samp_interval size ({min_samp_interval.shape[0]}) does not match voxel count ({self.voxel_model.num_voxels}), skipping subdivision')
            else:
                # 只有当min_samp_interval有效时才执行细分
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
