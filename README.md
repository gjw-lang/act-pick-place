# ACT Pick-and-Place (MuJoCo)

在 MuJoCo 仿真中复现 Action Chunking Transformer (ACT)，完成"抓杯子→放盘子"任务。

- 数据集: 99 episodes, 20Hz
- 模型: ACT + CVAE, chunk_size=50, dim_model=512
- 训练步数: 6000 steps
- 成功率: ~90% (宽松)

## 快速开始

```bash
# 采集数据
python scripts/collect_demo.py

# 训练
python scripts/train_act.py

# 部署
python scripts/deploy_act.py
```

## Rollout 演示

点击查看 rollout 录屏。
https://github.com/gjw-lang/act-pick-place/blob/main/rollout.gif
## 致谢

- ACT 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- LeRobot 框架: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- 教程环境: [datawhale/every-embodied](https://github.com/datawhalechina/every-embodied)
