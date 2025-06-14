import torch
import torch.nn as nn
import torch.nn.functional as F
from .promptda_dpt import DPTHead
from pathlib import Path
from huggingface_hub import hf_hub_download
from torchvision.transforms import Pad
from .vit_fpn import ViTFeaturePyramid
model_configs = {
    'vits': {'encoder': 'vits', 'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'in_channels': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]}
}

class PromptDA(nn.Module):
    patch_size     = 14   # DINOv2 patch size
    use_bn         = False
    use_clstoken   = False
    output_act     = 'sigmoid'

    def __init__(self,
                cfg,
                num_scales=1,
                encoder='vitl',
                 ):
        super().__init__()

        self.num_scales = num_scales
        self.model_config = model_configs[encoder]
        # load DINOv2 backbone
        module_path = Path(__file__)
        base_dir = str(Path(*module_path.parts[:-5]))
        self.pretrained = torch.hub.load(
            f'{base_dir}/torchhub/facebookresearch_dinov2_main',
            f'dinov2_{encoder}14',
            source='local',
            pretrained=False)
        self.feature_out_channels = self.pretrained.blocks[0].attn.qkv.in_features # 383
        

        # DPTHead 回归单通道深度
        self.depth_head = DPTHead(
            nclass=1,
            in_channels=self.feature_out_channels,
            features=self.model_config['features'],
            out_channels=self.model_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
        )

        # upsampler
        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=self.model_config['in_channels'],
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # 用于 normalize 输入 prompt_depth
        self.register_buffer('_mean', torch.tensor([0.485,0.456,0.406])[None,:,None,None])
        self.register_buffer('_std',  torch.tensor([0.229,0.224,0.225])[None,:,None,None])

        # self.load_checkpoint(ckpt_path)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        for param in self.depth_head.parameters():
            param.requires_grad = False

        # 切到 eval 模式，关闭 batchnorm/drpout 等
        self.pretrained.eval()
        self.depth_head.eval()

    @classmethod
    def from_pretrained(cls, repo_or_path=None, **hf_kwargs):
        # 同 PromptDA API
        return super().from_pretrained(repo_or_path, **hf_kwargs)

    def load_checkpoint(self, ckpt_path):
        if Path(ckpt_path).exists():
            ck = torch.load(ckpt_path, map_location='cpu')
            # strip possible "model." prefix
            st = {k.replace('model.',''):v for k,v in ck['state_dict'].items()}
            self.load_state_dict(st, strict=False)

    def forward(self, image, prompt_depth):
        """
        Args:
          image:       Tensor[B,3,H,W]  RGB image                   
          prompt_depth:Tensor[B,1,h_p,w_p]  LiDAR prompt depth      
        """
        # normalize prompt_depth to [0,1]
        if image.ndim == 5:
            B, V, C, H, W = image.shape
            image = image.view(B*V, C, H, W)
            prompt_depth = prompt_depth.view(B*V, 1, prompt_depth.shape[-2], prompt_depth.shape[-1])
        else:
            B, C, H, W = image.shape
            V = 1

        N, C, H, W = image.shape
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

        prompt_depth, mn, mx = self._normalize_prompt(prompt_depth)
        x = (F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") - self._mean) / self._std # [1, 3, 756, 1008]
        # print(f"Image shape after padding: {x.shape} -> {(N, C, H + pad_h, W + pad_w)}")
        feats = self.pretrained.get_intermediate_layers(
            x, 
            self.model_config['layer_idxs'],
            return_class_token=True)

        # depth 回归
        _, _, H_pad, W_pad = x.shape
        ph, pw = H_pad // self.patch_size, W_pad // self.patch_size
        depth_preds = self.depth_head(feats, ph, pw, prompt_depth)      # [B,1,ph*ps, pw*ps]
        depth_preds = self._denormalize_depth(depth_preds, mn, mx)  # [N,1,H_pad,W_pad]
        depth_preds = depth_preds[..., :H, :W]                          # [N,1,H,W]
        depth_preds = depth_preds.reshape(B, V, H, W).contiguous()

        # features_mono_intermediate
        feats_int = []
        for i in range(len(feats)):
            curr_features = (
                feats[i][0]
                .reshape(N, ph, pw, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (H, W),
                mode="bilinear",
                align_corners=True,
            )
            feats_int.append(curr_features) # list of [N, C_i, H_int, W_int]

        # # 处理 feats
        # mono_feat = feats_int[-1]
        # if self.num_scales > 1:
        #     ms_feats = self.mono_pyramid(mono_feat)  # 得到一个 list 长度 == num_scales
        # else:
        #     ms_feats = [mono_feat]
        
        return {"features_mono_intermediate": feats_int, 'depth_preds': [depth_preds], 'match_probs': None}

    @torch.no_grad()
    def predict(self, image, prompt_depth):
        return self.forward(image, prompt_depth)

    def _normalize_prompt(self, pd):
        B = pd.shape[0]
        # mn = pd.view(B,-1).amin(dim=1, keepdim=True)[:,None,None,None]
        # mx = pd.view(B,-1).amax(dim=1, keepdim=True)[:,None,None,None]
        min_val = torch.quantile(pd.reshape(B, -1), 0., dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(pd.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        pd = (pd - min_val) / (max_val - min_val)
        return pd, min_val, max_val

    def _denormalize_depth(self, d, mn, mx):
        return d*(mx-mn) + mn
