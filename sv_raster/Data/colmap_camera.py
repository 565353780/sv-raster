import torch
import numpy as np

from camera_control.Module.rgbd_camera import RGBDCamera

import svraster_cuda


class ColmapCamera:
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
            self._w2c = self.rgbd_camera.world2cameraColmap.float().cuda()
        return self._w2c

    @property
    def c2w(self) -> torch.Tensor:
        """相机到世界变换矩阵"""
        if self._c2w is None:
            self._c2w = self.rgbd_camera.camera2worldColmap.float().cuda()
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
