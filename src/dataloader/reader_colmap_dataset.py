# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import trimesh
import natsort
import numpy as np
from PIL import Image
from pathlib import Path
import concurrent.futures

from src.utils.colmap_utils import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat
)
from src.utils.camera_utils import focal2fov


def read_colmap_dataset(source_path, image_dir_name, mask_dir_name, camera_creator):

    source_path = Path(source_path)

    # Parse colmap meta data
    sparse_path = source_path / "sparse" / "0"
    if not sparse_path.exists():
        sparse_path = source_path / "colmap" / "sparse" / "0"
    if not sparse_path.exists():
        raise Exception("Can not find COLMAP reconstruction.")

    # Read cameras (intrinsics)
    cameras_txt_file = sparse_path / "cameras.txt"
    if cameras_txt_file.exists():
        cam_intrinsics = read_intrinsics_text(str(cameras_txt_file))
    else:
        raise Exception(f"Can not find cameras file in {sparse_path}")

    # Read images (extrinsics)
    images_txt_file = sparse_path / "images.txt"
    if images_txt_file.exists():
        cam_extrinsics = read_extrinsics_text(str(images_txt_file))
    else:
        raise Exception(f"Can not find images file in {sparse_path}")

    # Read 3D points
    pcd = trimesh.load(sparse_path / "points3D.ply")
    point_cloud = pcd.vertices

    # Sort key by filename
    keys = natsort.natsorted(
        cam_extrinsics.keys(),
        key=lambda k: cam_extrinsics[k].name)

    # Load all images and cameras
    todo_lst = []
    for key in keys:

        frame = cam_extrinsics[key]
        intr = cam_intrinsics[frame.camera_id]

        # Load image
        image_path = source_path / image_dir_name / frame.name
        if not image_path.exists():
            image_path = image_path.with_suffix('.png')
        if not image_path.exists():
            image_path = image_path.with_suffix('.jpg')
        if not image_path.exists():
            image_path = image_path.with_suffix('.JPG')
        if not image_path.exists():
            raise Exception(f"File not found: {str(image_path)}")
        image = Image.open(image_path)

        # Load camera intrinsic
        if intr.model == "SIMPLE_PINHOLE":
            focal_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            fovx = focal2fov(focal_x, intr.width)
            fovy = focal2fov(focal_x, intr.height)
            cx_p = cx / intr.width
            cy_p = cy / intr.height
        elif intr.model == "PINHOLE":
            focal_x = intr.params[0]
            focal_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            fovx = focal2fov(focal_x, intr.width)
            fovy = focal2fov(focal_y, intr.height)
            cx_p = cx / intr.width
            cy_p = cy / intr.height
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Load camera extrinsic
        # Convert quaternion and translation to 4x4 world-to-camera matrix
        R = qvec2rotmat(frame.qvec)  # 3x3 rotation matrix (camera to world rotation transposed)
        T = frame.tvec  # translation vector

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T

        # Load mask if there is
        mask_path = (source_path / mask_dir_name / frame.name).with_suffix('.png')
        if mask_path.exists():
            mask = Image.open(mask_path)
        else:
            mask = None

        todo_lst.append(dict(
            image=image,
            w2c=w2c,
            fovx=fovx,
            fovy=fovy,
            cx_p=cx_p,
            cy_p=cy_p,
            image_name=image_path.name,
            mask=mask,
        ))

    # Load all cameras concurrently
    import torch
    torch.inverse(torch.eye(3, device="cuda"))  # Fix module lazy loading bug:
                                                # https://github.com/pytorch/pytorch/issues/90613

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(camera_creator, **todo) for todo in todo_lst]
        cam_lst = [f.result() for f in futures]

    # Parse main scene bound if there is
    nerf_normalization_path = os.path.join(source_path, "nerf_normalization.json")
    if os.path.isfile(nerf_normalization_path):
        with open(nerf_normalization_path) as f:
            nerf_normalization = json.load(f)
        suggested_center = np.array(nerf_normalization["center"], dtype=np.float32)
        suggested_radius = np.array(nerf_normalization["radius"], dtype=np.float32)
        suggested_bounding = np.stack([
            suggested_center - suggested_radius,
            suggested_center + suggested_radius,
        ])
    else:
        suggested_bounding = None

    # Pack dataset
    dataset = {
        'train_cam_lst': cam_lst,
        'test_cam_lst': [],
        'suggested_bounding': suggested_bounding,
        'point_cloud': point_cloud,
    }
    return dataset
