"""
部署已训练的 ACT 策略，在 MuJoCo 环境中 rollout。
用法：python examples/deploy_act.py
"""
import sys
import os
import io

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_py310"
os.environ["HF_HUB_OFFLINE"] = "1"

work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA")
work_dir = os.path.abspath(work_dir)
os.chdir(work_dir)
sys.path.insert(0, work_dir)

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Patch for parquet compatibility
import lerobot.common.datasets.utils as ds_utils
import lerobot.common.datasets.lerobot_dataset as ds_mod

def hf_transform_to_torch_patched(items_dict):
    to_tensor = T.ToTensor()
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, Image.Image):
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif isinstance(first_item, dict) and ("bytes" in first_item or "path" in first_item):
            out = []
            for x in items_dict[key]:
                if isinstance(x, dict) and x.get("bytes") is not None:
                    img = Image.open(io.BytesIO(x["bytes"])).convert("RGB")
                    out.append(to_tensor(img))
                else:
                    out.append(x)
            items_dict[key] = out
        elif first_item is not None:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict

ds_utils.hf_transform_to_torch = hf_transform_to_torch_patched
ds_mod.hf_transform_to_torch = hf_transform_to_torch_patched

# --- 加载策略 ---
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Deploy] 设备: {device}")

print("[Deploy] 加载策略...")
dataset_metadata = LeRobotDatasetMetadata("datawhale_eai_pnp", root="./demo_data")
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

CHUNK = 50
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=CHUNK,
    n_action_steps=1,
    temporal_ensemble_coeff=0.0,  # 新旧预测等权
    use_vae=True,
)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy = ACTPolicy.from_pretrained(f"./ckpt/act_y_cs{CHUNK}", config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)
policy.eval()
policy.reset()
print("[Deploy] 策略加载完成")

# --- 创建环境 ---
from mujoco_env.y_env import SimpleEnv

xml_path = "./asset/example_scene_y.xml"
env = SimpleEnv(xml_path, action_type="joint_angle")
print("[Deploy] MuJoCo 环境已创建")

img_transform = T.ToTensor()
step = 0
success_strict = 0  # 严格成功
success_cup = 0     # 杯子到盘子（宽松）
total = 0
env.reset()
policy.reset()

while env.env.is_viewer_alive():
    env.step_env()
    if env.env.loop_every(HZ=20):
        state = env.get_ee_pose()
        image, wrist_image = env.grab_image()
        image = Image.fromarray(image).resize((256, 256))
        wrist_image = Image.fromarray(wrist_image).resize((256, 256))
        image_pt = img_transform(image)
        wrist_pt = img_transform(wrist_image)

        data = {
            "observation.state": torch.tensor(np.array([state]), dtype=torch.float32).to(device),
            "observation.image": image_pt.unsqueeze(0).to(device),
            "observation.wrist_image": wrist_pt.unsqueeze(0).to(device),
            "task": ["Put mug cup on the plate"],
            "timestamp": torch.tensor([step / 20.0]).to(device),
        }

        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()

        # 末端靠近盘子时松夹爪
        plate_xy = env.obj_init_pose[3:5]
        ee_xy = state[:2]
        if np.linalg.norm(ee_xy - plate_xy) < 0.08 and state[2] < 0.90:
            action[6] = 0.0  # 靠近盘子强制松开

        if step % 50 == 0:
            print(f"  step{step:4d}: grip={action[6]:.3f}")

        env.step(action)
        env.render()
        step += 1

        # 杯子是否到盘子上（宽松判定：XY距离<8cm, Z<0.85）
        p_mug = env.env.get_p_body('body_obj_mug_5')
        p_plate = env.env.get_p_body('body_obj_plate_11')
        cup_on_plate = np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.08 and p_mug[2] < 0.85

        if env.check_success():
            success_strict += 1
            success_cup += 1
            total += 1
            print(f"[Deploy] ✓ Success! 步数: {step}, 严格: {success_strict}/{total}, 宽松: {success_cup}/{total}")
            policy.reset()
            env.reset()
            step = 0

        if step >= 500:
            if cup_on_plate:
                success_cup += 1
            total += 1
            print(f"[Deploy] 超时, 重置... 宽松: {success_cup}/{total}")
            policy.reset()
            env.reset()
            step = 0

env.env.close_viewer()
