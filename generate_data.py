"""
MyVLA 数据生成脚本 - 简化版
只使用 CAM_FRONT 的 ego_pose，生成 VLA 训练数据
"""

import os
import json
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def locate_message(utimes, utime):
    """在 CAN bus 时间序列中定位最近的消息"""
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


class SimpleVLADataGenerator:
    def __init__(self, dataroot, version='v1.0-trainval'):
        """
        初始化数据生成器

        参数:
            dataroot: nuScenes 数据根目录
            version: 数据集版本
        """
        print(f"Loading nuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

        # 尝试加载 CAN bus（可能不存在）
        try:
            self.nusc_can = NuScenesCanBus(dataroot=dataroot)
            self.has_can_bus = True
            print("✅ CAN bus data loaded")
        except:
            self.nusc_can = None
            self.has_can_bus = False
            print("⚠️  CAN bus data not available")

    def get_cam_ego_pose(self, sample, cam_name='CAM_FRONT'):
        """
        获取相机的 ego_pose

        参数:
            sample: nuScenes sample
            cam_name: 相机名称

        返回:
            pose_record: ego_pose 字典
            cs_record: calibrated_sensor 字典
        """
        cam_token = sample['data'][cam_name]
        cam_data = self.nusc.get('sample_data', cam_token)
        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        return pose_record, cs_record

    def get_scene_samples(self, sample):
        """
        获取当前 sample 所在场景的所有 samples

        返回:
            scene_samples: 场景中的所有 samples
            this_idx: 当前 sample 在场景中的索引
        """
        scene_samples = []
        this_idx = 0

        # 向后收集
        temp = sample
        scene_samples.append(temp)
        while temp['next'] != '':
            temp = self.nusc.get('sample', temp['next'])
            scene_samples.append(temp)

        # 向前收集
        temp = sample
        while temp['prev'] != '':
            temp = self.nusc.get('sample', temp['prev'])
            scene_samples.insert(0, temp)
            this_idx += 1

        return scene_samples, this_idx

    def compute_trajectories(self, scene_samples, this_idx):
        """
        计算场景中所有时刻的轨迹（转换到当前时刻的坐标系）

        返回:
            ego_trajs: 所有时刻的轨迹 [N, 2]，已转换到当前时刻的 BEV 坐标系
        """
        # 获取当前时刻的参考坐标系
        ref_sample = scene_samples[this_idx]
        ref_pose, ref_cs = self.get_cam_ego_pose(ref_sample)

        # 计算所有时刻的全局位置（ego 车辆中心的位置）
        ego_trajs_global = []
        for sample in scene_samples:
            pose, cs = self.get_cam_ego_pose(sample)
            # 只需要 ego 的全局位置，不需要考虑 sensor 偏移
            global_pos = np.array(pose['translation'])
            ego_trajs_global.append(global_pos)

        ego_trajs_global = np.array(ego_trajs_global)

        # 转换到当前时刻的 ego 坐标系（BEV）
        # Step 1: 平移到当前 ego 原点
        ego_trajs = ego_trajs_global - np.array(ref_pose['translation'])

        # Step 2: 旋转到当前 ego 坐标系
        rot_mat = Quaternion(ref_pose['rotation']).inverse.rotation_matrix
        ego_trajs = np.dot(rot_mat, ego_trajs.T).T

        # 只保留 x, y（BEV 视角）
        # 在 ego 坐标系中：x-前，y-左，z-上
        ego_trajs = ego_trajs[:, :2]

        return ego_trajs

    def get_can_bus_data(self, scene_samples, this_idx):
        """
        获取场景中所有时刻的 CAN bus 数据

        返回:
            all_ego_status: 所有时刻的状态 [acc_x, acc_y, w_z, vel_x, vel_y, vel_z, steer]
        """
        scene_token = scene_samples[0]['scene_token']
        scene = self.nusc.get('scene', scene_token)

        try:
            # 获取 CAN bus 消息
            pose_msgs = self.nusc_can.get_messages(scene['name'], 'pose')
            steer_msgs = self.nusc_can.get_messages(scene['name'], 'steeranglefeedback')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]

            all_ego_status = []
            for sample in scene_samples:
                # 定位最近的 CAN bus 消息
                pose_idx = locate_message(pose_uts, sample['timestamp'])
                steer_idx = locate_message(steer_uts, sample['timestamp'])

                pose_data = pose_msgs[pose_idx]
                steer_data = steer_msgs[steer_idx]

                # 格式: [acc_x, acc_y, acc_z, w_x, w_y, w_z, vel_x, vel_y, vel_z, steer]
                ego_status = list(pose_data['accel']) + list(pose_data['rotation_rate']) + list(pose_data['vel']) + [steer_data['value']]
                all_ego_status.append(ego_status)

            return all_ego_status
        except:
            print(f"Warning: CAN bus data not available for scene {scene['name']}")
            return None

    def format_qa(self, sample, ego_trajs, this_idx, all_ego_status):
        """
        格式化为问答对

        参数:
            sample: 当前 sample
            ego_trajs: 场景轨迹 [N, 2]
            this_idx: 当前索引
            all_ego_status: CAN bus 数据

        返回:
            qa_dict: 问答对字典，如果数据不足返回 None
        """
        # 检查是否有足够的历史和未来数据
        hist_len = 2  # 需要过去 2 个点（1秒）
        fut_len = 6   # 需要未来 6 个点（3秒）

        if this_idx < hist_len or this_idx + fut_len >= len(ego_trajs):
            return None

        # 获取图像路径
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        image_path = cam_data['filename']  # 直接使用相对路径，如 samples/CAM_FRONT/xxx.jpg

        # 提取历史轨迹
        hist_trajs = ego_trajs[this_idx - hist_len : this_idx + 1]  # 包含当前时刻
        hist_status = all_ego_status[this_idx - hist_len : this_idx + 1] if all_ego_status else None

        # 提取未来轨迹
        fut_trajs = ego_trajs[this_idx + 1 : this_idx + 1 + fut_len]

        # 构建问题
        question = "You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. "
        question += f"Your task is to do your best to predict future waypoints for the vehicle over the next {fut_len//2} timesteps, "
        question += "given the vehicle's intent inferred from the images."
        question += f"Provided are the previous ego vehicle status recorded over the last {hist_len*0.5} seconds (at 0.5-second intervals). "
        question += "This includes the x and y coordinates of the ego vehicle. "
        question += "Positive x means forward direction while positive y means leftwards. "
        question += "The data is presented in the format [x, y]:."

        # 添加历史数据
        for i, (traj, status) in enumerate(zip(hist_trajs, hist_status if hist_status else [None]*len(hist_trajs))):
            t_offset = -(hist_len - i) * 0.5
            x, y = traj  # BEV 坐标系已经是 (forward, leftward)

            question += f"(t{t_offset:+.1f}s) [{x:.2f}, {y:.2f}]"

            if status:
                acc_x, acc_y = status[0], status[1]
                vel = np.sqrt(status[6]**2 + status[7]**2)  # 速度大小
                steer = status[9]
                question += f", Acceleration: X {acc_x:.2f}, Y {acc_y:.2f} m/s^2, Velocity: {vel:.2f} m/s, Steering angle: {steer:.2f} (positive: left turn, negative: right turn)"

            if i < len(hist_trajs) - 1:
                question += ", "

        question += "\n"

        # 构建答案
        answer = f"<PLANNING>Predicted future movement details for the next {fut_len//2} seconds "
        answer += "(sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). "
        answer += "Positive x means forward direction while positive y means leftwards. "
        answer += "The output is formatted as [x, y]: "

        for i, traj in enumerate(fut_trajs):
            x, y = traj  # BEV 坐标系已经是 (forward, leftward)
            answer += f"[{x:.2f}, {y:.2f}]"
            if i < len(fut_trajs) - 1:
                answer += ", "

        answer += "</PLANNING>"

        # 构建输出字典
        qa_dict = {
            "id": sample['token'],
            "images": [image_path],
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

        return qa_dict

    def generate(self, output_file, split='all', max_samples=None):
        """
        生成数据集

        参数:
            output_file: 输出 JSON 文件路径
            split: 'train', 'val', 或 'all'
            max_samples: 最大样本数（None 表示全部）
        """
        print(f"Generating {split} dataset...")

        # 根据 split 筛选 samples
        if split == 'all':
            valid_samples = self.nusc.sample
        elif split == 'train':
            # 使用 scene 的 name 判断（nuScenes 约定：scene-xxxx 小于 700 是训练集）
            valid_samples = [s for s in self.nusc.sample if int(self.nusc.get('scene', s['scene_token'])['name'].split('-')[1]) < 700]
        else:
            valid_samples = [s for s in self.nusc.sample if int(self.nusc.get('scene', s['scene_token'])['name'].split('-')[1]) >= 700]

        if max_samples:
            valid_samples = valid_samples[:max_samples]

        all_qa = []

        for sample in tqdm(valid_samples, desc=f"Processing {split}"):
            try:
                # 获取场景数据
                scene_samples, this_idx = self.get_scene_samples(sample)

                # 计算轨迹
                ego_trajs = self.compute_trajectories(scene_samples, this_idx)

                # 获取 CAN bus 数据
                all_ego_status = self.get_can_bus_data(scene_samples, this_idx)

                # 格式化为问答对
                qa = self.format_qa(sample, ego_trajs, this_idx, all_ego_status)

                if qa:
                    all_qa.append(qa)

            except Exception as e:
                print(f"Error processing sample {sample['token']}: {e}")
                continue

        # 保存到文件
        print(f"Saving {len(all_qa)} samples to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(all_qa, f, indent=2)

        print(f"Done! Generated {len(all_qa)} samples.")


def main():
    # 配置
    DATAROOT = '/home/ldx/project/MyVLA/data/nuscenes/'
    OUTPUT_DIR = '/home/ldx/project/MyVLA/data'
    VERSION = 'v1.0-mini'

    # 创建生成器
    generator = SimpleVLADataGenerator(dataroot=DATAROOT, version=VERSION)

    # 生成数据集（mini 版本较小，生成全部数据）
    generator.generate(
        output_file=os.path.join(OUTPUT_DIR, 'nuscenes_mini.json'),
        split='all',
        max_samples=None  # 设为数字可以限制样本数，用于测试
    )

    print(f"\n✅ 数据生成完成！")
    print(f"输出文件: {os.path.join(OUTPUT_DIR, 'nuscenes_mini.json')}")


if __name__ == '__main__':
    main()
