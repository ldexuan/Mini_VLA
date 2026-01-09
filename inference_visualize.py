"""
推理并可视化：同时显示预测轨迹和真实轨迹
"""
import json
import re
import numpy as np
import cv2
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes

# ========== 配置 ==========
BASE_MODEL = "/home/ldx/project/MyVLA/models/Qwen2-VL-2B-Instruct"
LORA_WEIGHTS = "/home/ldx/project/MyVLA/checkpoints"
DATA_FILE = "/home/ldx/project/MyVLA/data/nuscenes_mini.json"
NUSCENES_DATAROOT = "/home/ldx/project/MyVLA/data/nuscenes/"  # nuScenes数据根目录
IMAGE_ROOT = "/home/ldx/project/MyVLA/"  # 图像路径根目录（用于拼接JSON中的相对路径）
OUTPUT_DIR = "/home/ldx/project/MyVLA/output_vis/"
VERSION = "v1.0-mini"

# ========== 加载 nuScenes（用于相机参数） ==========
print("Loading nuScenes...")
nusc = NuScenes(version=VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

# ========== 加载模型 ==========
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model = model.merge_and_unload()
model.eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL)
print("✅ Model loaded!")

# ========== 辅助函数 ==========
def parse_trajectory(text):
    """从 <PLANNING>...</PLANNING> 中解析轨迹"""
    pattern = r'<PLANNING>(.*?)</PLANNING>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    planning_text = match.group(1)
    coord_pattern = r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]'
    coords = re.findall(coord_pattern, planning_text)

    if not coords:
        return None

    return np.array([[float(x), float(y)] for x, y in coords])


def bev_to_image(bev_points, sample_token):
    """将 BEV 坐标投影到图像"""
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    cam_height = abs(cs_record['translation'][2])

    N = len(bev_points)
    points_3d = np.zeros((N, 3))
    points_3d[:, 0] = -bev_points[:, 1]  # y_left → -x_cam
    points_3d[:, 1] = cam_height
    points_3d[:, 2] = bev_points[:, 0]   # x_forward → z_cam

    points_2d_homo = cam_intrinsic @ points_3d.T
    img_points = np.zeros((N, 2))
    img_points[:, 0] = points_2d_homo[0, :] / points_2d_homo[2, :]
    img_points[:, 1] = points_2d_homo[1, :] / points_2d_homo[2, :]

    img_width = cam_data['width']
    img_height = cam_data['height']
    valid_mask = (
        (img_points[:, 0] >= 0) & (img_points[:, 0] < img_width) &
        (img_points[:, 1] >= 0) & (img_points[:, 1] < img_height) &
        (points_2d_homo[2, :] > 0)
    )

    return img_points, valid_mask


def visualize_comparison(sample, prediction_text, output_path):
    """可视化预测和真实轨迹的对比"""
    # 解析轨迹
    pred_traj = parse_trajectory(prediction_text)
    gt_traj = parse_trajectory(sample['messages'][1]['content'])

    if pred_traj is None or gt_traj is None:
        print(f"Failed to parse trajectories")
        return

    # 读取图像
    img_path = f"{IMAGE_ROOT}{sample['images'][0]}"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    sample_token = sample['id']

    # 投影预测轨迹（蓝色）
    pred_img_points, pred_valid = bev_to_image(pred_traj, sample_token)
    for i, (u, v) in enumerate(pred_img_points):
        # 跳过NaN值
        if np.isnan(u) or np.isnan(v):
            continue
        color = (255, 100, 0) if pred_valid[i] else (150, 50, 0)  # 蓝色
        radius = 10 if i == 0 else 7
        cv2.circle(img, (int(u), int(v)), radius, color, -1)
        cv2.putText(img, f"P{i+1}", (int(u)+12, int(v)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i > 0:
            prev_u, prev_v = pred_img_points[i-1]
            if not (np.isnan(prev_u) or np.isnan(prev_v)):
                cv2.line(img, (int(prev_u), int(prev_v)), (int(u), int(v)), color, 3)

    # 投影真实轨迹（绿色）
    gt_img_points, gt_valid = bev_to_image(gt_traj, sample_token)
    for i, (u, v) in enumerate(gt_img_points):
        # 跳过NaN值
        if np.isnan(u) or np.isnan(v):
            continue
        color = (0, 255, 0) if gt_valid[i] else (0, 150, 0)  # 绿色
        radius = 8 if i == 0 else 6
        cv2.circle(img, (int(u), int(v)), radius, color, -1)
        cv2.putText(img, f"G{i+1}", (int(u)+12, int(v)+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i > 0:
            prev_u, prev_v = gt_img_points[i-1]
            if not (np.isnan(prev_u) or np.isnan(prev_v)):
                cv2.line(img, (int(prev_u), int(prev_v)), (int(u), int(v)), color, 2)

    # 添加图例
    cv2.rectangle(img, (10, 10), (350, 110), (0, 0, 0), -1)
    cv2.putText(img, "Prediction vs Ground Truth", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.circle(img, (30, 60), 8, (255, 100, 0), -1)
    cv2.putText(img, "Predicted (Blue)", (50, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(img, (30, 90), 8, (0, 255, 0), -1)
    cv2.putText(img, "Ground Truth (Green)", (50, 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 计算误差
    if len(pred_traj) == len(gt_traj):
        errors = np.linalg.norm(pred_traj - gt_traj, axis=1)
        avg_error = np.mean(errors)
        cv2.putText(img, f"Avg Error: {avg_error:.2f}m", (20, img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 保存
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Saved: {output_path}")


# ========== 主流程 ==========
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

# 推理所有样本
NUM_SAMPLES = len(data)
print(f"Total samples: {NUM_SAMPLES}")
for idx in range(NUM_SAMPLES):
    print(f"\n{'='*60}")
    print(f"Processing sample {idx+1}/{NUM_SAMPLES}...")

    sample = data[idx]
    image_path = f"{IMAGE_ROOT}{sample['images'][0]}"
    question = sample['messages'][0]['content']

    # 推理
    image = Image.open(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    prediction = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"Prediction: {prediction[:100]}...")

    # 可视化
    output_path = f"{OUTPUT_DIR}/comparison_{idx:03d}_{sample['id']}.jpg"
    visualize_comparison(sample, prediction, output_path)

print(f"\n{'='*60}")
print(f"✅ All done! Check results in: {OUTPUT_DIR}")
