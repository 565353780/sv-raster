import sys
sys.path.append('../../MATCH/camera-control')
sys.path.append('../../RECON/octree-shape')

import os
import cv2
import pickle
from tqdm import trange

from sv_raster.Config.config import TrainerConfig, ModelConfig, ProcedureConfig, InitConfig
from sv_raster.Module.trainer import Trainer


def demo():
    """
    示例：从 mesh 文件初始化并训练
    """
    # 小妖怪头
    shape_id = '003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f'
    # 女人上半身
    shape_id = '017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873'
    # 长发男人头
    shape_id = '0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb'

    home = os.environ['HOME']
    save_data_folder_path = home + "/chLi/Dataset/pixel_align/" + shape_id + '/'
    gen_mesh_file_path = save_data_folder_path + 'stage2_64_n.ply'
    octree_depth = 7

    print("=" * 50)
    print("Demo: Train from Mesh")
    print("=" * 50)

    # 创建一些虚拟相机用于演示
    save_camera_file_path = save_data_folder_path + 'camera.pkl'
    assert os.path.exists(save_camera_file_path)
    with open(save_camera_file_path, 'rb') as f:
        camera_list = pickle.load(f)

    # 创建自定义配置
    config = TrainerConfig(
        model=ModelConfig(
            sh_degree=1,
            white_background=True,
        ),
        procedure=ProcedureConfig(
            n_iter=3000,
            seed=42,
        ),
        init=InitConfig(
            init_n_level=octree_depth,
        ),
    )

    # 创建 Trainer
    trainer = Trainer(config)

    # 加载 mesh（替换为实际的 mesh 文件路径）
    assert os.path.exists(gen_mesh_file_path)
    trainer.loadMeshFile(gen_mesh_file_path, depth_max=octree_depth)

    trainer.addCameras(camera_list)

    print(f"[INFO] Added {len(trainer.train_cameras)} training cameras")

    # 开始训练
    trainer.train(verbose=True)

    print("[INFO] Demo completed (training skipped for demo)")

    output_folder = os.path.join(save_data_folder_path, "svraster", "render")
    os.makedirs(output_folder, exist_ok=True)

    print(f"[INFO] start save concatenated image to {output_folder}")
    for i in trange(len(camera_list)):
        render_image = trainer.render(camera_list[i], output_T=True, output_depth=True, output_normal=True)
        out_path = os.path.join(output_folder, f"{i:06d}.png")
        cv2.imwrite(out_path, render_image)

    trainer.exportMeshFile(
        save_data_folder_path + "svraster/tsdf_mesh.ply",
        final_lv=16,
        bbox_scale=1.0,
    )
    return trainer
