# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from scipy.interpolate import make_interp_spline


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(0.5 * fov))

def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def interpolate_poses(poses, n_frame, periodic=True):

    assert len(poses) > 1

    poses = list(poses)
    bc_type = None

    if periodic:
        poses.append(poses[0])
        bc_type = "periodic"

    pos_lst = np.stack([pose[:3, 3] for pose in poses])
    lookat_lst = np.stack([pose[:3, 2] for pose in poses])
    right_lst = np.stack([pose[:3, 0] for pose in poses])

    ts = np.linspace(0, 1, len(poses))
    pos_interp_f = make_interp_spline(ts, pos_lst, bc_type=bc_type)
    lookat_interp_f = make_interp_spline(ts, lookat_lst, bc_type=bc_type)
    right_interp_f = make_interp_spline(ts, right_lst, bc_type=bc_type)

    samps = np.linspace(0, 1, n_frame+1)[:n_frame]
    pos_video = pos_interp_f(samps)
    lookat_video = lookat_interp_f(samps)
    right_video = right_interp_f(samps)
    interp_poses = []
    for i in range(n_frame):
        pos = pos_video[i]
        lookat = lookat_video[i] / np.linalg.norm(lookat_video[i])
        right_ = right_video[i] / np.linalg.norm(right_video[i])
        down = np.cross(lookat, right_)
        right = np.cross(down, lookat)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = lookat
        c2w[:3, 3] = pos
        interp_poses.append(c2w)

    return interp_poses


def gen_circular_poses(radius,
                       n_frame,
                       starting=1.5 * np.pi, # Starting from -z
                       ):
    poses = []
    for rad in np.linspace(starting, starting + 2 * np.pi, n_frame):
        pos = radius * np.array([np.cos(rad), 0, np.sin(rad)])
        lookat = -pos / np.linalg.norm(pos)
        down = np.array([0, 1, 0])
        right = np.cross(down, lookat)
        right = right / np.linalg.norm(right)
        down = np.cross(lookat, right)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = lookat
        c2w[:3, 3] = pos
        poses.append(c2w)
    return poses
