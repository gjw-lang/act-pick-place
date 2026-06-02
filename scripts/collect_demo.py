"""
遥操作采集示教数据的独立脚本。
用法：python examples/collect_demo.py
任务：键盘控制机械臂抓起杯子放到盘子上。
功能：通过键盘遥控MuJoCo仿真机械臂完成任务，并将操作过程保存为LeRobot数据集，用于后续ACT模型训练
"""
import sys
import os
import numpy as np
from PIL import Image

# 设置MuJoCo的渲染后端，避免图形报错
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_py310"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_py310"
os.environ["HF_HUB_OFFLINE"] = "1"

# ------------------- 路径配置 -------------------
# 获取当前脚本所在目录，并拼接上项目根目录
work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA")
# 转换为绝对路径，防止路径出错
work_dir = os.path.abspath(work_dir)
# 切换工作目录到项目根目录
os.chdir(work_dir)
# 将项目目录加入Python搜索路径，确保能导入自定义模块
sys.path.insert(0, work_dir)

# 导入自定义的MuJoCo机械臂环境
from mujoco_env.y_env import SimpleEnv
# 导入LeRobot数据集工具，用于保存采集到的数据
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import glfw  # 用于检测 P 键开始/停止录制

# ------------------- 采集超参数配置 -------------------
SEED = None  # 随机种子，None表示每次重置环境，物体位置随机
REPO_NAME = "datawhale_eai_pnp"  # 数据集名称
NUM_DEMO = 20  # 追加 20 条特殊情况数据
ROOT = "./demo_data"  # 数据集保存根路径
TASK_NAME = "Put mug cup on the plate"  # 任务描述（用于标注数据）
xml_path = "./asset/example_scene_y.xml"  # MuJoCo场景文件（包含机械臂、杯子、盘子）

# ------------------- 创建/清空数据集 -------------------
# 如果数据集目录已存在，询问是否删除重新采集
create_new = True
if os.path.exists(ROOT):
    ans = input(f"目录 {ROOT} 已存在，是否删除？(y/n) ")
    if ans.lower() == "y":
        import shutil
        shutil.rmtree(ROOT)  # 删除旧数据集
    else:
        create_new = False

# 创建或加载 LeRobot 格式数据集
if create_new:
    dataset = LeRobotDataset.create(
    repo_id=REPO_NAME,        # 数据集ID
    root=ROOT,                # 保存路径
    robot_type="omy",         # 机器人类型
    fps=20,                   # 数据采集帧率（每秒20帧）
    # 定义数据集要保存的所有特征（列）
    features={
        "observation.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},  # 主相机图像
        "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},  # 腕部相机图像
        "observation.state": {"dtype": "float32", "shape": (6,)},  # 末端执行器姿态（x,y,z,roll,pitch,yaw）
        "action": {"dtype": "float32", "shape": (7,)},  # 机械臂动作（6个关节+1个夹爪）
        "obj_init": {"dtype": "float32", "shape": (6,)},  # 物体初始位置
    },
    image_writer_threads=2,  # 减少线程防崩溃
    image_writer_processes=1,
)
else:
    print("[Collect] 加载已有数据集...", end="", flush=True)
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)
    print(f" OK ({len(dataset.episodes or [])} 条)")

# ------------------- 创建MuJoCo仿真环境 -------------------
print("[Collect] 创建环境...")
# 初始化机械臂仿真环境，传入场景文件和随机种子
env = SimpleEnv(xml_path, seed=SEED, state_type="joint_angle")
print("[Collect] MuJoCo viewer 已打开。")

# 键盘操作说明
print("""
┌─────────────────────────────────────────────────────────────┐
│  键位说明                                                    │
│  WASD → XY 平面移动    R/F → Z 升降                        │
│  Q/E → Roll 倾斜       方向键 → Pitch / Yaw               │
│  空格 → 切换夹爪       Z → 重置（丢弃当前数据）             │
│                                                             │
│  目标：抓杯子 → 放盘子 → 松夹爪 → 抬手 → 自动保存          │
└─────────────────────────────────────────────────────────────┘
""")

# ------------------- 数据采集主循环 -------------------
action = np.zeros(7)  # 初始化动作（7维：6关节+1夹爪）
episode_id = 0        # 当前采集的演示序号
record_flag = False   # 是否开始录制数据的标志
p_pressed_last = False  # 检测 P 键上升沿
gripper_smooth = 0.0  # 夹爪平滑值（避免二值跳变）

# 循环条件：仿真窗口未关闭 + 未采集够指定数量的演示数据
while env.env.is_viewer_alive() and episode_id < NUM_DEMO:
    # 驱动仿真环境运行一步
    env.step_env()

    # 以20Hz频率执行控制逻辑（和采集帧率一致）
    if env.env.loop_every(HZ=20):
        # 检查是否完成任务（杯子是否放到盘子上）
        done = env.check_success()

        # ===================== 任务完成，保存数据 =====================
        # 延迟保存：松手后多录 1 秒记录抬升动作
        if hasattr(env, '_saving_delay'):
            env._saving_delay -= 1
            if env._saving_delay <= 0:
                if record_flag:
                    dataset.save_episode()
                    episode_id += 1
                    print(f"[Collect] 第 {episode_id}/{NUM_DEMO} 条数据已保存！")
                del env._saving_delay
                env.reset(seed=SEED)
                record_flag = False
                gripper_smooth = 0.0
        elif done and record_flag:
            env._saving_delay = 20  # 再录 20 帧 (1秒)

        # ===================== 键盘获取机械臂动作 =====================
        # 读取键盘输入，生成机械臂动作；reset=True表示按了Z键重置
        action, reset = env.teleop_robot()

        # ===================== P 键开始录制 / O 键停止录制 =====================
        p_pressed = env.env.is_key_pressed_once(key=glfw.KEY_P)
        o_pressed = env.env.is_key_pressed_once(key=glfw.KEY_O)
        if p_pressed and not p_pressed_last:
            record_flag = True
            print("[Collect] ▶ 开始录制...")
        if o_pressed and record_flag:
            record_flag = False
            dataset.save_episode()
            episode_id += 1
            print(f"[Collect] ⏹ 停止录制并保存！第 {episode_id}/{NUM_DEMO} 条")
        p_pressed_last = p_pressed

        # ===================== 按Z键重置，丢弃当前数据 =====================
        if reset:
            env.reset(seed=SEED)                     # 重置环境
            dataset.clear_episode_buffer()          # 清空当前未保存的数据
            record_flag = False                     # 停止录制
            gripper_smooth = 0.0                    # 重置夹爪平滑值
            if hasattr(env, '_saving_delay'):
                del env._saving_delay  # 清理延迟
            print("[Collect] 已重置，数据已丢弃。")

        # ===================== 获取观测数据 =====================
        ee_pose = env.get_ee_pose()  # 获取末端执行器位姿（6维）
        # 获取主相机和腕部相机图像
        agent_image, wrist_image = env.grab_image()

        # 将图像resize到256x256，并转成numpy数组
        agent_image = Image.fromarray(agent_image).resize((256, 256))
        wrist_image = Image.fromarray(wrist_image).resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)

        # 发送动作给机械臂，获取当前关节角度
        joint_q = env.step(action)

        # 夹爪平滑：每帧最多变 0.02，过渡约 50 帧 (2.5秒)
        grip_target = float(np.clip(action[-1], 0.0, 1.0))
        grip_diff = grip_target - gripper_smooth
        grip_diff = np.clip(grip_diff, -0.02, 0.02)
        gripper_smooth += grip_diff
        joint_q[-1] = gripper_smooth  # 替换为平滑值

        # ===================== 录制数据帧 =====================
        if record_flag:
            # 往数据集里添加一帧数据
            dataset.add_frame({
                "observation.image": agent_image,      # 主相机画面
                "observation.wrist_image": wrist_image, # 腕部相机画面
                "observation.state": ee_pose,          # 末端姿态
                "action": joint_q,                      # 关节动作
                "obj_init": env.obj_init_pose,          # 物体初始位置
            }, task=TASK_NAME)  # 附带任务描述

        # 渲染画面，显示键盘遥控的提示信息
        env.render(teleop=True)

# 关闭仿真窗口
env.env.close_viewer()

# 清理临时图像缓存
import shutil
shutil.rmtree(dataset.root / "images", ignore_errors=True)
print("[Collect] 完成！数据保存在", ROOT)
