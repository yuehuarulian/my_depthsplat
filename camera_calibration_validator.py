import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from io import BytesIO
import math
from scipy.spatial.transform import Rotation as R
import quaternion
import cv2
import decimal
import os
import imageio

class CameraCalibrationValidator:
    def __init__(self, scene_dir, cfg):
        self.scene_dir = Path(scene_dir)
        self.cfg = cfg
        self.near = cfg.near if cfg.near > 0 else 0.1
        self.far = cfg.far if cfg.far > 0 else 1000.0
        
    def get_up_vectors(self, pose):
        return np.matmul(pose, np.array([[0.0], [-1.0], [0.0], [0.0]]))

    def get_right_vectors(self, pose):
        return np.matmul(pose, np.array([[1.0], [0.0], [0.0], [0.0]]))
            
    def find_scene_orientation(self, poses_cam_to_world):
        if len(poses_cam_to_world) > 0:
            up_vector = sum(self.get_up_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
            right_vector = sum(self.get_right_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
            up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
        else:
            up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
            right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
            up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

        # value between 0, 180
        device_up_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                               up_vector), -1.0, 1.0)).item() * 180.0 / np.pi
        device_right_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                                  right_vector), -1.0, 1.0)).item() * 180.0 / np.pi

        up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(device_right_to_world_up_angle - 90.0)
        if up_closest_to_90:
            assert abs(device_up_to_world_up_angle - 90.0) < 45.0
            # LEFT
            if device_right_to_world_up_angle > 90.0:
                sky_direction_scene = 'LEFT'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi / 2.0])
            else:
                sky_direction_scene = 'RIGHT'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, -math.pi / 2.0])
        else:
            # right is close to 90
            assert abs(device_right_to_world_up_angle - 90.0) < 45.0
            if device_up_to_world_up_angle > 90.0:
                sky_direction_scene = 'DOWN'
                cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
            else:
                sky_direction_scene = 'UP'
                cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
        # sky_direction_scene = 'UP'
        # cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
        cam_to_rotated = np.eye(4)
        cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
        rotated_to_cam = np.linalg.inv(cam_to_rotated)
        return sky_direction_scene, rotated_to_cam

    def load_scene_data(self):

        # Step 1: 读取图像文件
        image_dir = self.scene_dir / "lowres_wide"
        depth_dir = self.scene_dir / "lowres_depth"
        images = sorted(image_dir.iterdir(), key=lambda p: float(p.stem.split("_", 1)[1]))

        # Step 2: 加载内参
        intr_dir = self.scene_dir / "lowres_wide_intrinsics"
        intr_map = {}
        for f in intr_dir.glob("*.pincam"):
            stem = f.stem
            w, h, fx, fy, cx, cy = map(float, f.read_text().split())
            intr_map[stem] = (w, h, fx, fy, cx, cy)

        # Step 3: 加载轨迹，构建 pose_cam_to_world
        traj_file = self.scene_dir / "lowres_wide.traj"
        poses_cam_to_world = []
        raw_traj = []

        for line in traj_file.read_text().splitlines():
            toks = line.split()
            ts = float(toks[0])
            rx, ry, rz = map(float, toks[1:4])
            tx, ty, tz = map(float, toks[4:7])
            angle_axis = [rx, ry, rz]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray([tx, ty, tz])
            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p
            poses_cam_to_world.append(np.linalg.inv(pose_w_to_p))  # pose_cam_to_world
            raw_traj.append((ts, pose_w_to_p))

        # Step 4: 判断朝向并构建旋转矩阵
        sky_direction_scene, rotated_to_cam = self.find_scene_orientation(poses_cam_to_world)
        print(f"Scene orientation: {sky_direction_scene}")

        # Step 5: 构造有效帧
        valid_data = []
        for f in images[:1000]:
            stem = f.stem
            ts = float(stem.split("_", 1)[1])
            if stem not in intr_map:
                continue
            try:
                _, pose_w_to_c = min(raw_traj, key=lambda x: abs(x[0] - ts))
                pose_c_to_w = np.linalg.inv(pose_w_to_c)
                pose_c_to_w_rotated = pose_c_to_w @ rotated_to_cam  # 修正方向

                # 加载图像和深度
                img_path = image_dir / f.name
                depth_path = depth_dir / f.name
                if not img_path.exists() or not depth_path.exists():
                    continue
                image = Image.open(img_path)
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

                # 应用方向旋转
                if sky_direction_scene == 'LEFT':
                    image = image.transpose(Image.ROTATE_270)
                    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                elif sky_direction_scene == 'RIGHT':
                    image = image.transpose(Image.ROTATE_90)
                    depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif sky_direction_scene == 'DOWN':
                    image = image.transpose(Image.ROTATE_180)
                    depth = cv2.rotate(depth, cv2.ROTATE_180)

                # resize depth to match image
                depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.float32) / 1000.0

                w, h, fx, fy, cx, cy = intr_map[stem]
                if sky_direction_scene in ['LEFT', 'RIGHT']:
                    fx, fy = fy, fx
                    cx, cy = cy, cx
                    w, h = h, w

                fx /= w
                fy /= h
                cx /= w
                cy /= h
                c2w = torch.tensor(pose_c_to_w_rotated, dtype=torch.float32)

                valid_data.append({
                    'image': image,
                    'depth': depth,
                    'intrinsics': (fx, fy, cx, cy),
                    'c2w': c2w,
                    'timestamp': ts,
                    'filename': f.name
                })

            except Exception as e:
                print(f"加载帧 {f.name} 出错: {e}")
                continue
        return valid_data

    def pixel_to_world(self, depth_map, intrinsics, c2w, sample_step=10):
        """将深度图转换为世界坐标系下的3D点"""
        fx, fy, cx, cy = intrinsics
        h, w = depth_map.shape
        
        # 创建像素坐标网格，进行下采样以减少点数
        v, u = np.meshgrid(np.arange(0, h, sample_step), np.arange(0, w, sample_step), indexing='ij')
        u = u.flatten()
        v = v.flatten()
        
        # 获取对应的深度值
        depth_values = depth_map[v, u]
        
        # 过滤无效深度
        valid_mask = (depth_values > 0) & (depth_values < 10.0)  # 假设有效深度范围
        u = u[valid_mask]
        v = v[valid_mask]
        depth_values = depth_values[valid_mask]
        
        # 反归一化像素坐标
        u_norm = u / w
        v_norm = v / h
        
        # 转换为相机坐标系
        x_cam = (u_norm - cx) * depth_values / fx
        y_cam = (v_norm - cy) * depth_values / fy
        z_cam = depth_values
        
        # 齐次坐标
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=0)
        
        # 转换到世界坐标系
        c2w_np = c2w.numpy()
        points_world = c2w_np @ points_cam
        
        return points_world[:3].T, (u, v)  # 返回3D点和对应的像素坐标
    
    def project_to_image(self, points_3d, intrinsics, c2w):
        """将3D点投影到图像平面"""
        fx, fy, cx, cy = intrinsics
        
        # 转换为相机坐标系
        w2c = torch.inverse(c2w)
        points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
        points_cam = (w2c.numpy() @ points_homo.T)[:3]
        
        # 投影到图像平面
        x_cam, y_cam, z_cam = points_cam
        
        # 过滤掉后面的点
        valid_mask = z_cam > 0
        x_cam = x_cam[valid_mask]
        y_cam = y_cam[valid_mask]
        z_cam = z_cam[valid_mask]
        
        # 透视投影
        u_norm = fx * x_cam / z_cam + cx
        v_norm = fy * y_cam / z_cam + cy
        
        return np.column_stack([u_norm, v_norm]), valid_mask
    
    def compute_point_cloud_alignment(self, points_1, points_2):
        """计算两个点云之间的对齐度"""
        from scipy.spatial.distance import cdist
        
        # 计算最近邻距离
        if len(points_1) > 1000:
            sample_1 = points_1[np.random.choice(len(points_1), 1000, replace=False)]
        else:
            sample_1 = points_1
            
        if len(points_2) > 1000:
            sample_2 = points_2[np.random.choice(len(points_2), 1000, replace=False)]
        else:
            sample_2 = points_2
        
        # 计算距离矩阵
        distances = cdist(sample_1, sample_2)
        min_distances = np.min(distances, axis=1)
        
        print(f"点云对齐统计:")
        print(f"  平均最近邻距离: {np.mean(min_distances):.4f} m")
        print(f"  最近邻距离标准差: {np.std(min_distances):.4f} m")
        print(f"  最近邻距离中位数: {np.median(min_distances):.4f} m")
        print(f"  95%分位数距离: {np.percentile(min_distances, 95):.4f} m")
        
        # 如果平均距离很小，说明外参可能是正确的
        if np.mean(min_distances) < 0.05:  # 5cm
            print("相机外参可能是正确的（点云对齐良好）")
        else:
            print("相机外参可能有问题（点云对齐较差）")
    
# 使用示例
class MockConfig:
    def __init__(self):
        self.highres = False  # 设置为True使用高分辨率深度图
        self.near = 0.1
        self.far = 1000.0

def detailed_reprojection_analysis(scene_path):
    """详细的重投影误差分析"""
    cfg = MockConfig()
    validator = CameraCalibrationValidator(scene_path, cfg)
    
    valid_data = validator.load_scene_data()
    if len(valid_data) < 2:
        print("数据不足")
        return
    
    frame1 = valid_data[0]
    frame2 = valid_data[20]
    print(f"使用帧 {frame1['filename']} 和 {frame2['filename']} 进行详细重投影分析")
    
    # 获取frame1的3D点
    points_3d_1, pixels_1 = validator.pixel_to_world(
        frame1['depth'], frame1['intrinsics'], frame1['c2w'], sample_step=1
    )
    # 获取frame2的3D点
    points_3d_2, pixels_2 = validator.pixel_to_world(
        frame2['depth'], frame2['intrinsics'], frame2['c2w'], sample_step=1
    )
    
    # 投影到frame2
    projected_pixels, valid_mask = validator.project_to_image(
        points_3d_1, frame2['intrinsics'], frame2['c2w']
    )
    
    # 检查投影的像素是否在图像范围内
    h, w = frame2['depth'].shape
    valid_proj = (projected_pixels[:, 0] >= 0) & (projected_pixels[:, 0] < w) & \
                 (projected_pixels[:, 1] >= 0) & (projected_pixels[:, 1] < h)
    
    print(f"投影到frame2的有效点数: {np.sum(valid_proj)} / {len(projected_pixels)}")
    
    if np.sum(valid_proj) > 0:
        valid_proj_pixels = projected_pixels[valid_proj].astype(int)
        
        # 获取投影位置的深度值
        proj_depths = frame2['depth'][valid_proj_pixels[:, 1], valid_proj_pixels[:, 0]]
        
        # 计算预期深度（从3D点到frame2相机的距离）
        w2c2 = torch.inverse(frame2['c2w'])
        points_homo = np.column_stack([points_3d_1[valid_mask][valid_proj], np.ones(np.sum(valid_proj))])
        points_cam2 = (w2c2.numpy() @ points_homo.T)[:3]
        expected_depths = points_cam2[2]  # Z坐标就是深度
        
        # 计算深度误差
        depth_errors = np.abs(proj_depths - expected_depths)
        valid_depth_mask = (proj_depths > 0) & (expected_depths > 0)
        
        if np.sum(valid_depth_mask) > 0:
            valid_errors = depth_errors[valid_depth_mask]
            print(f"深度重投影误差统计:")
            print(f"  平均误差: {np.mean(valid_errors):.4f} m")
            print(f"  误差标准差: {np.std(valid_errors):.4f} m")
            print(f"  误差中位数: {np.median(valid_errors):.4f} m")
            print(f"  95%分位数误差: {np.percentile(valid_errors, 95):.4f} m")
            
            if np.mean(valid_errors) < 0.1:  # 10cm
                print("深度重投影误差较小，外参可能正确")
            else:
                print("深度重投影误差较大，外参可能有问题")
    
    return [points_3d_1, points_3d_2]

def visualize_3d_points(validation_results, max_points=5000):
    """合并两帧的点云并保存为文件"""
    points_1, points_2 = validation_results
    
    # 随机采样减少点数
    if len(points_1) > max_points:
        indices_1 = np.random.choice(len(points_1), max_points, replace=False)
        points_1 = points_1[indices_1]
    if len(points_2) > max_points:
        indices_2 = np.random.choice(len(points_2), max_points, replace=False)
        points_2 = points_2[indices_2]
    
    # 使用 Open3D 创建两个点云并赋予颜色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_1)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_2)
    pcd2.paint_uniform_color([0, 0, 1])  # 蓝色

    # 合并点云
    combined = pcd1 + pcd2
    
    # 保存为 .ply 点云文件
    o3d.io.write_point_cloud("merged_pointcloud.ply", combined)
    print("✅ 点云保存为 merged_pointcloud.ply")

    # # 以固定视角渲染成 PNG 图像
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)
    # vis.add_geometry(combined)
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.5)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("merged_pointcloud.png")
    # vis.destroy_window()
    # print("✅ 渲染图像保存为 merged_pointcloud.png")

def visualize_3d_points2(validation_results, max_points=5000):
    """可视化并保存为 PNG"""
    points_1, points_2 = validation_results

    if len(points_1) > max_points:
        points_1 = points_1[np.random.choice(len(points_1), max_points, replace=False)]
    if len(points_2) > max_points:
        points_2 = points_2[np.random.choice(len(points_2), max_points, replace=False)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_1[:, 0], points_1[:, 1], points_1[:, 2], c='red', s=1, label='Frame 1')
    ax.scatter(points_2[:, 0], points_2[:, 1], points_2[:, 2], c='blue', s=1, label='Frame 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Merged Point Cloud (Frame 1 + Frame 2)")
    plt.tight_layout()
    plt.savefig("merged_pointcloud_matplotlib.png", dpi=300)
    print("✅ 图像保存为 merged_pointcloud_matplotlib.png")

def generate_pointcloud_sequence_video(output_video="pointcloud_sequence.mp4", max_points=5000):
    """
    为每帧 valid_data 生成点云可视化图像，并组合成视频。
    
    Args:
        valid_data: 来自 load_scene_data() 的数据列表
        validator: CameraCalibrationValidator 实例
        output_video: 输出视频文件名
        max_points: 每帧最多渲染的点数
    """
    cfg = MockConfig()
    validator = CameraCalibrationValidator(scene_path, cfg)
    
    valid_data = validator.load_scene_data()
    if len(valid_data) < 2:
        print("数据不足")
        return
    
    temp_dir = "temp_seq_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    
    all_points = []
    for frame in valid_data:
        points, _ = validator.pixel_to_world(
            frame['depth'], frame['intrinsics'], frame['c2w'], sample_step=8
        )
        if len(points) > max_points:
            points = points[np.random.choice(len(points), max_points, replace=False)]
        all_points.append(points)
    all_points = np.concatenate(all_points, axis=0)

    xlim = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
    ylim = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    zlim = (np.min(all_points[:, 2]), np.max(all_points[:, 2]))

    for idx, frame in enumerate(valid_data):
        points, _ = validator.pixel_to_world(
            frame['depth'], frame['intrinsics'], frame['c2w'], sample_step=4
        )

        if len(points) > max_points:
            points = points[np.random.choice(len(points), max_points, replace=False)]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='green', s=1)

        ax.set_title(f"Frame {idx}: {frame['filename']}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=30)

        # 固定坐标轴范围
        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)

        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        frame_paths.append(frame_path)

    # 合成视频
    with imageio.get_writer(output_video, fps=10) as writer:
        for path in frame_paths:
            image = imageio.v2.imread(path)
            writer.append_data(image)

    print(f"点云序列视频保存为 {output_video}")

    # 可选：删除临时帧
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_dir)

if __name__ == "__main__":
    # 使用示例
    scene_path = "/home/featurize/data/my_depthsplat/datasets/ARKitScenes/raw/Training/40753679"  # 替换为你的场景路径
    
    # print("\n=== 详细重投影误差分析 ===")
    # validation_results = detailed_reprojection_analysis(scene_path)

    # visualize_3d_points2(validation_results)

    print("\n=== 生成点云序列视频 ===")
    generate_pointcloud_sequence_video()