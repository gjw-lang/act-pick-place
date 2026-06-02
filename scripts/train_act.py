"""
训练 ACT (Action Chunking Transformer) 的独立脚本。
用法：python examples/train_act.py
功能：使用 LeRobot 框架训练 ACT 模型，用于机械臂抓取任务
"""
import sys
import os
import time

# 设置 MuJoCo 渲染后端，避免报错
os.environ.setdefault("MUJOCO_GL", "glfw")
# 指定 HuggingFace 数据集缓存目录，防止默认路径占满系统盘
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_py310"
os.environ["HF_HUB_OFFLINE"] = "1"  # 不连 Hub，仅读取本地数据

# 定位工作目录（自动跳转到项目根目录）
# __file__ 是当前脚本路径，os.path.dirname 取父目录，然后拼接目标路径
work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA")
# 转成绝对路径，避免路径混乱
work_dir = os.path.abspath(work_dir)
# 切换工作目录
os.chdir(work_dir)
# 把工作目录加入 Python 导入路径，确保能 import 项目内的包
sys.path.insert(0, work_dir)

# ------------------- 深度学习库导入 -------------------
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io

# LeRobot 数据集相关
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
# ACT 模型配置与模型定义
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
# 特征类型（图像/状态/动作）
from lerobot.configs.types import FeatureType
# 时间戳解析（用于动作块 Action Chunk）
from lerobot.common.datasets.factory import resolve_delta_timestamps

# 保留手腕相机，帮助模型感知夹爪-物体距离，学习夹爪时机
# LeRobot 数据集相关
# 原版 LeRobot 读取 parquet 图片可能报错，这里重写加载函数
import lerobot.common.datasets.utils as ds_utils
import lerobot.common.datasets.lerobot_dataset as ds_mod

def hf_transform_to_torch_patched(items_dict):
    """
    把数据集里的图片/状态转成 PyTorch Tensor
    修复：bytes 格式图片读取报错问题
    """
    to_tensor = T.ToTensor()
    # 遍历数据集的每一列（image, state, action...）
    for key in items_dict:
        first_item = items_dict[key][0]
        # 如果是 PIL 图片 → 转 tensor
        if isinstance(first_item, Image.Image):
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        # 如果是字典格式（bytes 二进制图片）→ 解码成 RGB 再转 tensor
        elif isinstance(first_item, dict) and ("bytes" in first_item or "path" in first_item):
            out = []
            for x in items_dict[key]:
                if isinstance(x, dict) and x.get("bytes") is not None:
                    img = Image.open(io.BytesIO(x["bytes"])).convert("RGB")
                    out.append(to_tensor(img))
                else:
                    out.append(x)
            items_dict[key] = out
        # 其他数值 → 直接转 tensor
        elif first_item is not None:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict

# 把原版函数替换成我们修复后的版本
ds_utils.hf_transform_to_torch = hf_transform_to_torch_patched
ds_mod.hf_transform_to_torch = hf_transform_to_torch_patched

# ------------------- 训练配置 -------------------
# 自动选择 GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Train] 使用设备: {device}")

# 总训练步数
training_steps = 6000
# 每 50 步打印一次日志
log_freq = 50
# 模型保存路径
save_dir = "./ckpt/act_y"
# batch_size: 16 for 8GB VRAM, 降到 8 如果还 OOM
BATCH_SIZE = 16

# ------------------- 加载数据集元信息 -------------------
print("[Train] 加载数据集...")
# 加载数据集（名称 + 根目录）
dataset_metadata = LeRobotDatasetMetadata("datawhale_eai_pnp", root="./demo_data")

# 把数据集特征转成模型能识别的格式
features = dataset_to_policy_features(dataset_metadata.features)
# 分离输出特征 = 动作(action)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
# 分离输入特征 = 图像 + 状态
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 去掉腕部图像（只保留主相机）
# 保留手腕相机，帮助模型感知夹爪-物体距离，学习夹爪时机

# ------------------- 创建 ACT 模型 -------------------
print("[Train] 创建 ACT 策略 (chunk_size=10)...")
# 初始化 ACT 配置
# chunk_size=10：一次预测未来 10 步动作
# n_action_steps=10：一次执行 10 步动作
cfg = ACTConfig(input_features=input_features, output_features=output_features,
                chunk_size=50, n_action_steps=50,
                use_vae=True)  # 开启 CVAE
save_dir = f"./ckpt/act_y_cs{cfg.chunk_size}"  # 不同帧数存不同目录
# 解析动作块时间戳（Transformer 需要知道未来帧时序）
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 创建 ACT 模型（Transformer 结构 + 变分自编码器 VAE）
print("[Train] 从头初始化...")
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
policy.train()
policy.to(device)

# 打印可训练参数量
num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"[Train] 可训练参数: {num_params:,}")

# ------------------- 创建数据加载器 -------------------
# 正式加载数据集（带时间戳）
dataset = LeRobotDataset("datawhale_eai_pnp", delta_timestamps=delta_timestamps, root="./demo_data")
# Windows 下必须设 0，Linux 可以设 4~8 加速
num_workers = 0
# 构建 DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset, num_workers=num_workers, batch_size=BATCH_SIZE,
    shuffle=True, pin_memory=(device.type != "cpu"), drop_last=False,
)
print(f"[Train] 数据帧数: {dataset.num_frames}, batch数: {len(dataloader)}")

# 优化器：Adam 学习率 1e-4
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# ------------------- 开始训练 -------------------
print(f"[Train] 开始训练 {training_steps} 步...")
step = 0
done = False
start_time = time.time()

# 循环训练直到达到指定步数
while not done:
    # 遍历所有 batch
    for batch in dataloader:
        # 把数据搬到 GPU
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # 前向传播 → 计算损失
        # ACT 损失 = 动作重构损失 + KL 散度损失
        loss, output_dict = policy.forward(inp_batch)
        # 反向传播更新梯度
        loss.backward()
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()

        # 定时打印日志
        if step % log_freq == 0:
            kld = output_dict.get("kld_loss", float("nan"))
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  step {step:5d} | loss: {loss.item():.4f} | KL: {kld:.4f} | GPU: {mem_used:.1f}/{mem_reserved:.1f}G")

        step += 1
        # 达到训练步数就停止
        if step >= training_steps:
            done = True
            break

# 训练耗时
elapsed = time.time() - start_time
print(f"[Train] 训练完成! 耗时: {elapsed:.1f}s")

# ------------------- 保存模型 -------------------
os.makedirs(save_dir, exist_ok=True)
policy.save_pretrained(save_dir)
print(f"[Train] 模型已保存到 {save_dir}")

# ------------------- 评估模型（动作预测误差） -------------------
print("[Train] 评估模型 MAE...")
# 切换到评估模式（关闭 Dropout/BatchNorm）
policy.eval()

# 存储预测动作、真实动作、图像
actions, gt_actions, images = [], [], []
# 取第一个 episode 的所有帧做评估
from_idx = dataset.episode_data_index["from"][0].item()
to_idx = dataset.episode_data_index["to"][0].item()
frame_ids = list(range(from_idx, to_idx))

# 构建评估集
sample = torch.utils.data.Subset(dataset, frame_ids)
test_loader = torch.utils.data.DataLoader(sample, batch_size=1, shuffle=False)

# 重置模型状态
policy.reset()
# 逐帧推理
for batch in test_loader:
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    # 模型预测一步动作
    action = policy.select_action(inp_batch)
    # 保存结果
    actions.append(action)
    gt_actions.append(inp_batch["action"][:, 0, :])
    images.append(inp_batch["observation.image"])

# 拼接所有结果
actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
# 计算平均绝对误差 MAE（越小越好）
mae = torch.mean(torch.abs(actions - gt_actions)).item()
print(f"[Train] Mean Action Error (MAE): {mae:.4f}")

# 逐关节/维度 MAE
per_dim_mae = torch.mean(torch.abs(actions - gt_actions), dim=0).cpu().numpy()
dim_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
print("[Train] 各维度 MAE:")
for name, val in zip(dim_names, per_dim_mae):
    print(f"  {name}: {val:.4f}")

print("\n[Train] 完成! 下一步: 运行 deploy 脚本部署模型。")
