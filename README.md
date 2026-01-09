# MyVLA - 视觉-语言-动作自动驾驶轨迹预测

<div align="center">
  <img src="output_trajectory.gif" alt="Trajectory Prediction Demo" width="800"/>
  <p><i>预测轨迹（蓝色）vs 真实轨迹（绿色）</i></p>
</div>

基于 Qwen2-VL 的端到端自动驾驶轨迹预测系统，使用 nuScenes 数据集训练。

## 环境配置

### 要求
- Python 3.8+
- CUDA 12.1

### 安装步骤

1. **创建 conda 环境**
```bash
conda create -n myvla python=3.10
conda activate myvla
```

2. **安装 PyTorch (CUDA 12.1)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **安装其他依赖**
```bash
pip install -r requirements.txt
```

4. **验证安装**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

看到 `CUDA: True` 说明环境配置成功！

## 数据准备

### 下载 nuScenes 数据

1. **下载数据文件**

   - nuScenes 数据: [nuscenes_.zip](https://pan.baidu.com/s/1q6fjty-YcsT9ruIrs7nqKA?pwd=1qk4) 提取码: `1qk4`
   - CAN bus 数据: [can_bus.zip](https://pan.baidu.com/s/1XYiIi5Xe9LlRiRnXfClj-w?pwd=38nq) 提取码: `38nq`

2. **解压并组织数据**

```bash
# 解压数据到项目目录
unzip nuscenes_.zip -d data/
unzip can_bus.zip -d data/nuscenes/

# 最终的数据结构应该如下:
# data/
# └── nuscenes/
#     ├── can_bus/          # CAN bus 数据
#     ├── maps/             # 地图数据
#     ├── samples/          # 图像数据
#     │   └── CAM_FRONT/    # 前视相机图像
#     ├── v1.0-mini/        # v1.0-mini 元数据
#     └── LICENSE
```

3. **验证数据**

```bash
python -c "from nuscenes import NuScenes; nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes/', verbose=True); print(f'✓ 数据加载成功! 共 {len(nusc.sample)} 个样本')"
```

### 下载基础模型

1. **下载 Qwen2-VL-2B-Instruct 模型**

   模型下载: [Qwen2-VL-2B-Instruct](https://www.modelscope.ai/models/iic/gme-Qwen2-VL-2B-Instruct)

2. **组织模型文件**

```bash
# 将下载的模型放到 models 目录
# 最终结构:
# models/
# └── Qwen2-VL-2B-Instruct/
#     ├── config.json
#     ├── model.safetensors (或 pytorch_model.bin)
#     ├── tokenizer_config.json
#     └── ...
```

3. **验证模型**

```bash
python -c "from transformers import AutoProcessor; processor = AutoProcessor.from_pretrained('models/Qwen2-VL-2B-Instruct'); print('✓ 模型加载成功!')"
```

## 快速开始

### 1. 数据生成

```bash
python generate_data.py
```

生成的数据将保存到 `data/nuscenes_mini.json`

### 2. 模型训练

```bash
llamafactory-cli train train/configs/lora_train.yaml
```

训练的 checkpoints 保存在 `checkpoints/` 目录

### 3. 模型推理

```bash
# 可视化推理（生成图像）
python inference_visualize.py
```

可视化结果保存在 `output_vis/` 目录

## 项目结构

```
MyVLA/
├── generate_data.py          # 数据生成脚本
├── inference.py              # 基础推理（文本）
├── inference_visualize.py    # 可视化推理
├── train/
│   └── configs/
│       └── lora_train.yaml   # 训练配置
├── data/
│   ├── nuscenes/             # nuScenes 数据集
│   └── dataset_info.json     # 数据集配置
├── models/                   # 基础模型目录
├── checkpoints/              # 训练产生的 LoRA 权重
└── requirements.txt          # 依赖列表
```

## 数据格式

- **输入**: 前视相机图像 + 历史1秒轨迹 + CAN bus 数据
- **输出**: 未来3秒轨迹预测（6个航点，每0.5秒一个）
- **坐标系**: BEV（Bird's Eye View），x-前，y-左

## 训练配置

- 基础模型: Qwen2-VL-2B-Instruct
- 微调方法: LoRA (rank=16, alpha=32)
- 训练轮数: 10 epochs
- Batch size: 1 × 4 (gradient accumulation)
- 学习率: 3e-5