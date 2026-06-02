#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Action Chunking Transformer Policy
动作分块Transformer策略（ACT），出自论文《Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware》
核心思想：一次预测一长串连续动作（动作块），而非逐帧预测，让机器人轨迹更平滑、精细

本文件实现：
1. ACTPolicy：LeRobot框架的策略接口，负责推理、训练、动作队列、归一化
2. ACT：核心神经网络，包含VAE、Transformer编码器/解码器、视觉主干ResNet
3. 位置编码、编码器层、解码器层、时间融合器
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

# 导入ACT配置类
from lerobot.common.policies.act.configuration_act import ACTConfig
# 数据归一化/反归一化工具
from lerobot.common.policies.normalize import Normalize, Unnormalize
# LeRobot预训练策略基类
from lerobot.common.policies.pretrained import PreTrainedPolicy


# ==============================================
# 1. 策略主类：ACT 策略（框架接口层）
# 负责：推理、动作队列、训练损失、数据归一化、模型调用
# ==============================================
class ACTPolicy(PreTrainedPolicy):
    """
    ACT 策略包装类
    对接 LeRobot 框架，管理模型生命周期、推理、训练、动作输出
    """

    config_class = ACTConfig  # 绑定配置类
    name = "act"             # 策略名称

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        初始化策略
        config: 配置参数
        dataset_stats: 数据集均值/方差，用于归一化
        """
        super().__init__(config)
        config.validate_features()  # 校验输入特征是否合法
        self.config = config

        # 输入归一化（图像、状态 → 标准化）
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        # 目标（动作）归一化
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        # 输出反归一化（模型输出 → 原始动作范围）
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # 核心ACT模型
        self.model = ACT(config)

        # 启用时间融合（动作平滑）时初始化
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        # 重置动作队列
        self.reset()

    def get_optim_params(self) -> dict:
        """
        获取优化参数
        分为两部分：主干网络（ResNet） + 其他参数，可设置不同学习率
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """环境重置时清空动作队列 / 时间融合器"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        【推理核心】
        输入观测 → 模型预测 → 输出单步动作
        内部维护动作队列，一次预测多步，分批执行
        """
        self.eval()

        # 输入归一化
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # 方案1：时间融合（平滑动作），逐帧推理
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # 方案2：动作队列（一次预测多步，分批执行）
        if len(self._action_queue) == 0:
            # 模型预测 chunk_size 步，取前 n_action_steps 步
            actions = self.model(batch)[0][:, : self.config.n_action_steps]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # 存入队列
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        【训练核心】
        前向传播 + 计算损失（L1损失 + VAE的KL散度）
        """
        # 输入归一化
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # 动作标签归一化
        batch = self.normalize_targets(batch)
        # 模型前向
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # L1损失（动作回归）
        l1_loss = (F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        loss_dict = {"l1_loss": l1_loss.item()}

        # 如果使用VAE，增加KL损失
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


# ==============================================
# 2. 时间融合器：动作平滑算法（ACT论文Algorithm 2）
# ==============================================
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """
        时间融合：加权平均历史预测动作，让机器人运动更平滑
        越新的帧权重越高（指数加权）
        """
        self.chunk_size = chunk_size
        # 预计算指数权重
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """重置融合状态"""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        在线更新融合动作
        输入：(batch, chunk_size, action_dim)
        输出：融合后的当前步动作
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=actions.device)
        else:
            # 指数加权在线更新
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # 拼接最新预测的最后一步
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat([self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])])

        # 取出并丢弃第一个动作
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


# ==============================================
# 3. ACT 核心神经网络：VAE + Transformer + ResNet
# ==============================================
class ACT(nn.Module):
    """
    ACT 神经网络主体
    结构：
    - VAE编码器（训练时用，输入动作序列→隐变量）
    - ResNet视觉主干（提取图像特征）
    - Transformer编码器（融合观测：图像+状态+隐变量）
    - Transformer解码器（生成动作序列）
    - 动作头（输出动作）
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # ===================== VAE 编码器（训练专用） =====================
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)  # 分类token

            # 机器人状态投影
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )

            # 动作序列投影
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )

            # 隐变量分布投影（μ + logσ²）
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

            # VAE编码器位置编码
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1

            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # ===================== 视觉主干 ResNet =====================
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 取ResNet layer4输出
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ===================== Transformer 编码器/解码器 =====================
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # 编码器输入投影
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)

        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        # 图像特征投影
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)

        # 1D 特征位置编码
        n_1d_tokens = 1  # 隐变量
        if self.config.robot_state_feature: n_1d_tokens += 1
        if self.config.env_state_feature: n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # 2D 图像位置编码
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # 解码器可学习位置编码（动作序列）
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # 最终动作输出头
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """Transformer 参数 Xavier 初始化"""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        模型前向传播
        训练时：输入图像+状态+动作 → 输出动作序列 + VAE隐变量分布
        推理时：输入图像+状态 → 输出动作序列
        """
        if self.config.use_vae and self.training:
            assert "action" in batch, "训练VAE必须提供动作标签"

        # 获取批次大小
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # ===================== 1. 训练阶段：VAE 编码动作得到隐变量 =====================
        mu = log_sigma_x2 = None
        if self.config.use_vae and "action" in batch:
            # 构造VAE输入：[CLS, 机器人状态, 动作序列]
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)

            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(1)

            action_embed = self.vae_encoder_action_input_proj(batch["action"])

            # 拼接输入
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]

            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            # 掩码
            cls_joint_is_pad = torch.full((batch_size, 2 if self.config.robot_state_feature else 1), False, device=batch["observation.state"].device)
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            # VAE编码 → 得到隐变量分布
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]

            # 重参数化采样
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 推理时隐变量为0
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(batch["observation.state"].device)

        # ===================== 2. 构造 Transformer 编码器输入 =====================
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # 添加机器人状态
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))

        # 添加环境状态
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch["observation.environment_state"]))

        # 图像特征 + 2D位置编码
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            for img in batch["observation.images"]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # 展平为序列
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # 堆叠序列
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # ===================== 3. Transformer 编码 → 解码 =====================
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # 解码器输入（全0）
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # 转换形状 → 动作头输出
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


# ==============================================
# 4. Transformer 编码器（支持VAE编码器+主编码器）
# ==============================================
class ACTEncoder(nn.Module):
    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """单层Transformer编码器：自注意力 + 前馈网络"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm: x = self.norm2(x)
        return x


# ==============================================
# 5. Transformer 解码器
# ==============================================
class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed, encoder_pos_embed)
        return self.norm(x)


class ACTDecoderLayer(nn.Module):
    """单层Transformer解码器：自注意力 + 交叉注意力 + 前馈网络"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm: x = self.norm3(x)
        return x


# ==============================================
# 工具函数：1D/2D 位置编码、激活函数
# ==============================================
def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """标准1D正弦位置编码（Transformer原论文）"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D正弦位置编码（用于图像特征图）"""
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """获取激活函数：relu/gelu/glu"""
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"不支持激活函数 {activation}")
