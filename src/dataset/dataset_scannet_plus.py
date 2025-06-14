import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from scipy.spatial.transform import Rotation as R
import quaternion
import decimal
import cv2
from scipy import interpolate
import math

@dataclass
class DatasetScannetPlusCfg(DatasetCfgCommon):
    name: Literal["scannet_plus"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    use_index_to_load_chunk: Optional[bool] = False
    load_depth: bool = True


class DatasetScannetPlus(IterableDataset):
    cfg: DatasetScannetPlusCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetScannetPlusCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # near/far
        self.near = cfg.near if cfg.near > 0 else 0.1
        self.far  = cfg.far  if cfg.far  > 0 else 1000.0

        # 每个场景一个文件夹
        base = Path(cfg.roots[0]) / ("Training" if stage=="train" else "Validation")
        # self.scenes = [p for p in sorted(base.iterdir()) if p.is_dir()]
        raw_scenes = [p for p in sorted(base.iterdir()) if p.is_dir()]
        valid = []
        for scene in raw_scenes:
            depth_subdir = "highres_depth" if cfg.highres else "lowres_depth"
            if not (scene/"lowres_wide").exists():
                print(f"Warning: 场景 {scene.name} 缺少 lowres_wide 文件夹，已跳过。")
                continue
            if not (scene/"lowres_wide.traj").exists():
                print(f"Warning: 场景 {scene.name} 缺少 lowres_wide.traj，已跳过。")
                continue
            if not (scene/"lowres_wide_intrinsics").exists():
                print(f"Warning: 场景 {scene.name} 缺少 lowres_wide_intrinsics ，已跳过。")
                continue
            if not (scene/depth_subdir).exists():
                print(f"Warning: 场景 {scene.name} 缺少 {depth_subdir}，已跳过。")
                continue
            valid.append(scene)
        self.scenes = valid

    def get_up_vectors(self, pose_device_to_world):
        """获取设备上方向量在世界坐标系中的表示"""
        return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))

    def get_right_vectors(self, pose_device_to_world):
        """获取设备右方向量在世界坐标系中的表示"""
        return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))

    def find_scene_orientation(self, poses_cam_to_world):
        """
        检测场景方向，返回方向标识和相应的旋转变换矩阵
        """
        if len(poses_cam_to_world) > 0:
            up_vector = sum(self.get_up_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
            right_vector = sum(self.get_right_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
            up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
        else:
            up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
            right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
            up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

        # 计算角度
        device_up_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                               up_vector), -1.0, 1.0)).item() * 180.0 / np.pi
        device_right_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                                  right_vector), -1.0, 1.0)).item() * 180.0 / np.pi

        up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(device_right_to_world_up_angle - 90.0)
        if up_closest_to_90:
            assert abs(device_up_to_world_up_angle - 90.0) < 45.0
            if device_right_to_world_up_angle > 90.0:
                sky_direction_scene = 'LEFT'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi / 2.0])
            else:
                sky_direction_scene = 'RIGHT'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, -math.pi / 2.0])
        else:
            assert abs(device_right_to_world_up_angle - 90.0) < 45.0
            if device_up_to_world_up_angle > 90.0:
                sky_direction_scene = 'DOWN'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
            else:
                sky_direction_scene = 'UP'
                cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
        
        sky_direction_scene = 'UP'
        cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0) # TODO
        cam_to_rotated = np.eye(4)
        cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
        rotated_to_cam = np.linalg.inv(cam_to_rotated)
        return sky_direction_scene, rotated_to_cam

    def read_trajectory_with_interpolation(self, traj_file, timestamps_selected):
        """
        读取轨迹文件并进行插值，返回插值后的位姿
        """
        timestamps = []
        poses = []
        quaternions = []
        poses_cam_to_world = []
        
        with open(traj_file) as f:
            for line in f.readlines():
                tokens = line.split()
                if len(tokens) != 7:
                    continue
                    
                traj_timestamp = float(tokens[0])
                timestamps.append(traj_timestamp)
                
                # 轴角表示的旋转
                angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
                t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
                
                # 构造 world to camera 变换
                pose_w_to_p = np.eye(4)
                pose_w_to_p[:3, :3] = r_w_to_p
                pose_w_to_p[:3, 3] = t_w_to_p
                
                # 转换为 camera to world
                pose_p_to_w = np.linalg.inv(pose_w_to_p)
                poses_cam_to_world.append(pose_p_to_w)
                
                # 提取旋转四元数和平移
                r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
                t_p_to_w = pose_p_to_w[:3, 3]
                poses.append(t_p_to_w)
                quaternions.append(r_p_to_w_as_quat)
        
        if len(timestamps) == 0:
            return None, None, None
            
        # 转换为numpy数组进行插值
        poses = np.array(poses)
        quaternions = np.array(quaternions, dtype=np.quaternion)
        quaternions = quaternion.unflip_rotors(quaternions)
        timestamps = np.array(timestamps)
        timestamps_selected = np.array(timestamps_selected)
        
        # 进行插值
        spline = interpolate.interp1d(timestamps, poses, kind='linear', axis=0)
        try:
            interpolated_rotations = quaternion.squad(quaternions, timestamps, timestamps_selected)
            interpolated_positions = spline(timestamps_selected)
        except ValueError:
            # 如果插值失败，使用最近邻
            interpolated_rotations = []
            interpolated_positions = []
            for ts in timestamps_selected:
                closest_idx = np.argmin(np.abs(timestamps - ts))
                interpolated_rotations.append(quaternions[closest_idx])
                interpolated_positions.append(poses[closest_idx])
            interpolated_rotations = np.array(interpolated_rotations)
            interpolated_positions = np.array(interpolated_positions)
        
        return interpolated_rotations, interpolated_positions, poses_cam_to_world

    def apply_image_rotation(self, image_tensor, depth_tensor, sky_direction_scene):
        """
        根据场景方向旋转图像和深度图
        """
        if sky_direction_scene == 'RIGHT':
            # 逆时针旋转90度
            image_tensor = torch.rot90(image_tensor, k=1, dims=[-2, -1])
            depth_tensor = torch.rot90(depth_tensor, k=1, dims=[-2, -1])
            # self.cfg.image_shape = (self.cfg.image_shape[-2], self.cfg.image_shape[-1])  # 更新图像形状
        elif sky_direction_scene == 'LEFT':
            # 顺时针旋转90度
            image_tensor = torch.rot90(image_tensor, k=-1, dims=[-2, -1])
            depth_tensor = torch.rot90(depth_tensor, k=-1, dims=[-2, -1])
            # self.cfg.image_shape = (self.cfg.image_shape[-2], self.cfg.image_shape[-1])  # 更新图像形状
        elif sky_direction_scene == 'DOWN':
            # 旋转180度
            image_tensor = torch.rot90(image_tensor, k=2, dims=[-2, -1])
            depth_tensor = torch.rot90(depth_tensor, k=2, dims=[-2, -1])
        
        return image_tensor, depth_tensor

    def adjust_intrinsics_for_rotation(self, fx, fy, cx, cy, w, h, sky_direction_scene):
        """
        根据旋转调整内参
        """
        if sky_direction_scene == 'RIGHT' or sky_direction_scene == 'LEFT':
            # 交换宽高和相应的内参
            return fy / h, fx / w, cy / h, cx / w, h, w
        else:
            return fx / w, fy / h, cx / w, cy / h, w, h

    def __iter__(self):
        scenes = self.scenes.copy()
        if self.stage == "train":
            torch.random.manual_seed(0)
            scenes = [scenes[i] for i in torch.randperm(len(scenes))]

        for scene_dir in scenes:
            # 1. 列出 wide 文件并按时间戳排序
            wide_dir = scene_dir / "lowres_wide"
            wide_files = sorted(wide_dir.iterdir(),key=lambda p: float(p.stem.split("_", 1)[1]))
            if len(wide_files) < self.view_sampler.num_context_views + self.view_sampler.num_target_views:
                continue

            # 2. 读取内参文件夹
            intrinsics_dir = scene_dir / "lowres_wide_intrinsics"
            intr_map = {}
            for f in intrinsics_dir.glob("*.pincam"):
                stem = f.stem
                w, h, fx, fy, cx, cy = map(float, f.read_text().split())
                intr_map[stem] = (w, h, fx, fy, cx, cy)

            # 3. 准备轨迹插值
            traj_file = scene_dir / "lowres_wide.traj"
            if not traj_file.exists():
                print(f"Warning: Trajectory file {traj_file} does not exist, skipping scene {scene_dir.name}.")
                continue

            # 4. 筛选有效帧
            valid_data = []
            depth_dir = scene_dir / ("highres_depth" if self.cfg.highres else "lowres_depth")
            for i, f in enumerate(wide_files):
                stem = f.stem
                ts = float(stem.split("_", 1)[1])
                if stem not in intr_map:
                    continue
                    
                depth_path = depth_dir / f.name
                image_path = wide_dir / f.name
                if not depth_path.exists() or not image_path.exists():
                    continue
                    
                valid_data.append((i, f, ts, stem))

            if len(valid_data) < self.view_sampler.num_context_views + self.view_sampler.num_target_views:
                continue

            # 5. 进行轨迹插值
            timestamps_selected = [data[2] for data in valid_data]
            interpolated_rotations, interpolated_positions, poses_cam_to_world = \
                self.read_trajectory_with_interpolation(traj_file, timestamps_selected)

            if interpolated_rotations is None:
                continue

            # 6. 检测场景方向
            sky_direction_scene, rotated_to_cam = self.find_scene_orientation(poses_cam_to_world)

            # 7. 构建poses和加载数据
            poses = []
            images = []
            depths = []
            
            for idx, (_, f, ts, stem) in enumerate(valid_data):
                # 获取内参
                w, h, fx, fy, cx, cy = intr_map[stem]
                
                # 根据旋转调整内参
                fx_adj, fy_adj, cx_adj, cy_adj, w_adj, h_adj = \
                    self.adjust_intrinsics_for_rotation(fx, fy, cx, cy, w, h, sky_direction_scene)

                # 构建相机位姿
                pose = np.eye(4)
                pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[idx])
                pose[:3, 3] = interpolated_positions[idx]
                
                # 应用场景方向校正
                corrected_pose = pose @ rotated_to_cam
                
                # # 转换为OpenCV坐标系
                # arkit_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
                # corrected_pose = corrected_pose @ arkit_to_opencv
                
                # 转换为世界到相机的变换
                w2c = np.linalg.inv(corrected_pose)[:3]  # shape: (3, 4)
                
                poses.append(
                    torch.cat([
                        torch.tensor([fx_adj, fy_adj, cx_adj, cy_adj], dtype=torch.float32),
                        torch.tensor([self.near, self.far], dtype=torch.float32),
                        torch.tensor(w2c.reshape(-1), dtype=torch.float32)
                    ])
                )
                
                # 加载图像
                image_path = wide_dir / f.name
                depth_path = depth_dir / f.name
                images.append(image_path.read_bytes())
                depths.append(depth_path.read_bytes())

            if len(poses) == 0:
                continue
                
            cams = torch.stack(poses, dim=0)  # [num, 18]

            # 8. 转换 poses，采样 indices
            extrinsics, intrinsics = self.convert_poses(cams)
            try:
                context_idx, target_idx = self.view_sampler.sample(
                    scene_dir.name, extrinsics, intrinsics
                )
            except ValueError:
                continue

            # 9. 加载并处理图像/深度
            context_idx = context_idx[:1]
            if context_idx.max().item() >= len(images) or target_idx.max().item() >= len(images):
                print(f"Warning: 采样索引超出范围，场景 {scene_dir.name} 跳过。")
                continue
                
            # 加载图像 处理深度图
            ctx_imgs = []
            ctx_deps = []
            for i in context_idx:
                img = self.to_tensor(Image.open(BytesIO(images[i])))
                dep = torch.from_numpy(
                    np.array(Image.open(BytesIO(depths[i])), dtype=np.float32)
                ).unsqueeze(0)
                img, dep = self.apply_image_rotation(img, dep, sky_direction_scene)
                ctx_imgs.append(img)
                ctx_deps.append(dep)
            ctx_imgs = torch.stack(ctx_imgs)
            ctx_deps = torch.stack(ctx_deps, dim=0) / 1000.0
            
            tgt_imgs = []
            tgt_deps = []
            for i in target_idx:
                img = self.to_tensor(Image.open(BytesIO(images[i])))
                dep = torch.from_numpy(
                    np.array(Image.open(BytesIO(depths[i])), dtype=np.float32)
                ).unsqueeze(0)
                img, dep = self.apply_image_rotation(img, dep, sky_direction_scene)
                tgt_imgs.append(img)
                tgt_deps.append(dep)
            tgt_imgs = torch.stack(tgt_imgs)
            tgt_deps = torch.stack(tgt_deps, dim=0) / 1000.0

            # 8. 构建 example 并 yield
            example = {
                "context": {
                    "extrinsics": extrinsics[context_idx],
                    "intrinsics": intrinsics[context_idx],
                    "image": ctx_imgs,
                    "depth": ctx_deps,
                    "near": self.get_bound("near", len(context_idx)),
                    "far": self.get_bound("far", len(context_idx)),
                    "index": context_idx,
                },
                "target": {
                    "extrinsics": extrinsics[target_idx],
                    "intrinsics": intrinsics[target_idx],
                    "image": tgt_imgs,
                    "depth": tgt_deps,
                    "near": self.get_bound("near", len(target_idx)),
                    "far": self.get_bound("far", len(target_idx)),
                    "index": target_idx,
                },
                "scene": scene_dir.name,
            }

            # augment & crop
            if self.stage=="train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_crop_shim(example, tuple(self.cfg.image_shape))
            
    def compute_c2w_from_pose(self, rx, ry, rz, tx, ty, tz):
        """
        从 axis-angle + translation 构建相机外参，并判断方向。
        """
        # 1. 构造旋转矩阵 (C2W)
        rot = R.from_rotvec([rx, ry, rz]).as_matrix()  # numpy, shape (3, 3)

        # 2. 构造 4x4 C2W 矩阵
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.tensor(rot, dtype=torch.float32)
        c2w[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)

        # 4. 判断相机朝向（Z 轴方向）
        z_vec = rot[2, :]  # 第三行，表示相机 Z 轴方向
        z_orien = np.array([
            [0.0, -1.0, 0.0],  # upright
            [-1.0,  0.0, 0.0],  # left
            [0.0,  1.0, 0.0],  # upside-down
            [1.0,  0.0, 0.0],  # right
        ])
        corr = np.matmul(z_orien, z_vec)
        orientation_index = int(np.argmax(corr))

        return c2w, orientation_index

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def __len__(self) -> int:
        n_scenes = len(self.scenes)
        if self.stage == "test" and self.cfg.test_len is not None and self.cfg.test_len > 0:
            return min(n_scenes, self.cfg.test_len)
        return n_scenes * self.cfg.train_times_per_scene

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)