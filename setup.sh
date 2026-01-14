cd ..
git clone --depth 1 https://github.com/rahul-goel/fused-ssim/

pip install ninja

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

pip install numpy einops yacs tqdm natsort argparse \
  pillow imageio imageio-ffmpeg scikit-image pycolmap \
  plyfile shapely gpytoolbox lpips pytorch-msssim

pip install opencv-python==4.8.0.74
pip install opencv-contrib-python==4.8.0.74
pip install trimesh==4.0.4
pip install open3d==0.18.0
pip install numpy==1.26.4

cd fused-ssim
python setup.py install

cd ../sv-raster/cuda
python setup.py install
