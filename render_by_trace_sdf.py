#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio

import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.fuser_utils import Fuser


@torch.no_grad()
def render_set(name, iteration, suffix, args, views, voxel_model):

    render_path = os.path.join(args.model_path, name, f"ours_{iteration}{suffix}_trace_by_sdf", "renders")
    makedirs(render_path, exist_ok=True)
    print(f'render_path: {render_path}')
    print(f'ss            =: {voxel_model.ss}')
    print(f'vox_geo_mode  =: {voxel_model.vox_geo_mode}')
    print(f'density_mode  =: {voxel_model.density_mode}')

    if args.eval_fps:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if args.eval_fps:
        # Warmup
        voxel_model.render_trace_sdf(views[0])

    eps_time = time.perf_counter()
    psnr_lst = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        hit_depth, hit_vox_id = voxel_model.render_trace_sdf(view)
        if not args.eval_fps:
            rendering = voxel_model._sh0[hit_vox_id].moveaxis(-1, 0)
            rendering *= (hit_vox_id != -1)
            gt = view.image.cuda()
            mse = (rendering.clip(0,1) - gt.clip(0,1)).square().mean()
            psnr = -10 * torch.log10(mse)
            psnr_lst.append(psnr.item())
            fname = view.image_name

            # RGB
            imageio.imwrite(
                os.path.join(render_path, fname + (".jpg" if args.use_jpg else ".png")),
                im_tensor2np(rendering)
            )
    torch.cuda.synchronize()
    eps_time = time.perf_counter() - eps_time
    peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
    if args.eval_fps:
        print(f'Eps time: {eps_time:.3f} sec')
        print(f"Peak mem: {peak_mem:.2f} GB")
        print(f'FPS     : {len(views)/eps_time:.0f}')
        outtxt = os.path.join(args.model_path, name, "ours_{}{}.txt".format(iteration, suffix))
        with open(outtxt, 'w') as f:
            f.write(f"n={len(views):.6f}\n")
            f.write(f"eps={eps_time:.6f}\n")
            f.write(f"peak_mem={peak_mem:.2f}\n")
            f.write(f"fps={len(views)/eps_time:.6f}\n")
    else:
        print('PSNR:', np.mean(psnr_lst))


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--eval_fps", action="store_true")
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--use_jpg", action="store_true")
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background, camera_params_only=False)

    # Load model
    voxel_model = SparseVoxelModel(cfg.model)
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)

    # Output path suffix
    suffix = args.suffix
    if not args.suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"
    
    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode
        if not args.suffix:
            suffix += f"_{args.overwrite_vox_geo_mode}"

    # Fuse sdf and rgb
    volume = Fuser(
        xyz=voxel_model.grid_pts_xyz,
        bandwidth=voxel_model.vox_size.min().item() * 20,
        # bandwidth=torch.zeros([len(voxel_model.grid_pts_xyz)], dtype=torch.float32, device="cuda").index_reduce_(
        #     dim=0,
        #     index=voxel_model.vox_key.flatten(),
        #     source=voxel_model.vox_size.repeat(1, 8).flatten(),
        #     reduce="amax") * 3,
        use_trunc=True,
        fuse_tsdf=True,
        feat_dim=3)

    for cam in tqdm(data_pack.get_train_cameras()):
        median_depth, median_idx = voxel_model.render_median(cam)
        volume.integrate(cam=cam, feat=cam.image.cuda(), depth=median_depth)

    voxel_model._shs.data.fill_(0)
    voxel_model._sh0.data.copy_(
        volume.feature.nan_to_num_()[voxel_model.vox_key].mean(dim=1))
    voxel_model._geo_grid_pts.data.copy_(
        volume.tsdf.nan_to_num_())

    del volume
    torch.cuda.empty_cache()

    # Start rendering
    voxel_model.freeze_vox_geo()

    if not args.skip_train:
        render_set(
            "train", loaded_iter, suffix, args,
            data_pack.get_train_cameras(), voxel_model)

    if not args.skip_test:
        render_set(
            "test", loaded_iter, suffix, args,
            data_pack.get_test_cameras(), voxel_model)
