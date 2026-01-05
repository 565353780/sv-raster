cd ..
git clone --depth 1 https://github.com/rahul-goel/fused-ssim/

pip install numpy einops yacs tqdm natsort argparse \
  pillow imageio imageio-ffmpeg scikit-image pycolmap \
  plyfile shapely gpytoolbox lpips pytorch-msssim

pip install opencv-python==4.8.0.74
pip install opencv-contrib-python==4.8.0.74
pip install trimesh==4.0.4
pip install open3d==0.18.0
