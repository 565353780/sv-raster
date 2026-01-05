# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pycolmap
import numpy as np

from typing import NamedTuple


class PointCloud(NamedTuple):
    points: np.array
    colors: np.array
    errors: np.array
    corr: dict


def parse_colmap_pts(sfm: pycolmap.Reconstruction, transform: np.array =None):
    """
    Parse COLMAP points and correspondents.

    Input:
        @sfm        Reconstruction from COLMAP.
        @transform  3x3 matrix to transform xyz.
    Output:
        @xyz        Nx3 point positions.
        @rgb        Nx3 point colors.
        @err        N   errors.
        @corr       Dictionary from file name to point indices.
    """

    xyz = []
    rgb = []
    err = []
    points_id = []
    for k, v in sfm.points3D.items():
        points_id.append(k)
        xyz.append(v.xyz)
        rgb.append(v.color)
        err.append(v.error)
        if transform is not None:
            xyz[-1] = transform @ xyz[-1]

    xyz = np.array(xyz)
    rgb = np.array(rgb)
    err = np.array(err)
    points_id = np.array(points_id)

    points_idmap = np.full([points_id.max()+2], -1, dtype=np.int64)
    points_idmap[points_id] = np.arange(len(xyz))

    corr = {}
    for image in sfm.images.values():
        idx = np.array([p.point3D_id for p in image.points2D if p.has_point3D()])
        corr[image.name] = points_idmap[idx]
        assert corr[image.name].min() >= 0 and corr[image.name].max() < len(xyz)

    return PointCloud(points=xyz, colors=rgb, errors=err, corr=corr)
