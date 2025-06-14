from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

# from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.promptda import PromptDA
from .unimatch.dpt_head import DPTHead


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        # DepthSplat 的深度预测器：同时处理多视图 cost volume 和单目先验
        self.depth_predictor = PromptDA(
            cfg = cfg,
            encoder = cfg.monodepth_vit_type,
            num_scales=cfg.num_scales,
        )

        # 如果仅做深度监督训练，提前返回，不再构建与高斯溅射相关的模块
        if self.cfg.train_depth_only:
            return

        feature_channels = self.depth_predictor.feature_out_channels
        
        # gaussians adapter
        # 把高斯参数（位置、协方差、颜色、SH 系数等）转成渲染可用的格式
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        # in_channels = 3 + 1 + 1 + feature_upsampler_channels
        in_channels = 3 + 1 + feature_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        # 把高维输入先做一次“特征萃取 + 融合”，输出固定 channels 通道的中间表示，后续再交给 gaussian_head 细化
        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        # 预测每个像素对应的高斯属性向量（包含 SH 系数、协方差分量、offset、opacity 等）
        # in_channels = 3 + feature_upsampler_channels + channels + 1
        in_channels = 3 + feature_channels + channels
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(in_channels, num_gaussian_parameters,
                      3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(num_gaussian_parameters,
                      num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
        )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # ==================== 1. 视图选择（最近邻视图索引） ====================
        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # ==================== 2. 深度预测（多视图+单目融合） ====================
        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            context["depth"],
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H_pad, W_pad]
        depth = depth_preds[-1]

        # ==================== 3. 仅深度训练分支 ====================
        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # ==================== 6. Gaussian 参数预测 ====================
        features = results_dict["features_mono_intermediate"][-1]
        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"), # [bv, C, H, W]
            rearrange(depth, "b v h w -> (b v) () h w"), # [bv, H, W]
            features,
        ), dim=1)

        # 先用一个小网络（gaussian_regressor）对拼接后的特征做一次预处理
        # 相当于论文里描述的轻量 U-Net，用来把图像、深度、匹配概率和特征融合起来，初步预测高斯参数
        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    ]

        out = torch.cat(concat, dim=1)

        # 再用 gaussian_head 输出最终的高斯元参数 (α, Σ, c)
        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        # ==================== 7. 组织 depths & densities & raw_gaussians ====================
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # [B, V, H*W, 1, 1]
        # densities = rearrange(match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            # densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, 1, 1] [1, 4, 758016, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # ==================== 8. 从特征到 Gaussian 参数 ====================
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        # ==================== 9. 调用 Gaussian Adapter 构建最终 Gaussians 对象 ====================
        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # ==================== 10. 可视化输出（可选） ====================
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        # ==================== 11. 返回结果 ====================
        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
