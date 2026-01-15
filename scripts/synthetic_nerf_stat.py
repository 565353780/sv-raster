# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import json
from argparse import ArgumentParser
import glob
import pandas as pd

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('result_root')
args = parser.parse_args()

scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']

psnr = []
ssim = []
lpips = []
tr_time = []
fps = []
n_voxels = []
max_iter = sorted(glob.glob(f'{args.result_root}/{scenes[0]}/test_stat/iter*.json'))[-1].split('/')[-1]
for scene in scenes:
    eval_path = f'{args.result_root}/{scene}/test_stat/{max_iter}'
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ret = json.load(f)
            psnr.append(ret['psnr'])
            tr_time.append(ret['elapsed'] / 1000)
            fps.append(ret['fps'])
            n_voxels.append(ret['n_voxels'])
    else:
        psnr.append(0)
        tr_time.append(0)
        fps.append(0)
        n_voxels.append(0)

    eval_path = f'{args.result_root}/{scene}/results.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            ret = ret[sorted(ret.keys())[-1]]
            psnr[-1] = ret['PSNR']
            ssim.append(ret['SSIM'])
            lpips.append(ret['LPIPS'])
    else:
        ssim.append(0)
        lpips.append(0)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['psnr'] = df['psnr'].round(2)
    df['ssim'] = df['ssim'].round(3)
    df['lpips'] = df['lpips'].round(3)
    df['tr-mins'] = (df['tr-mins'] / 60).round(1)
    df['fps'] = df['fps'].round(1)
    df['#vox(M)'] = (df['#vox(M)'] / 1_000_000).round(1)
    return df.to_string(index=False)

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df = pd.DataFrame({
    'scene': scenes,
    'psnr': psnr,
    'ssim': ssim,
    'lpips': lpips,
    'tr-mins': tr_time,
    'fps': fps,
    '#vox(M)': n_voxels,
})
df = add_avg_row(df)

print(format_df_string(df))
