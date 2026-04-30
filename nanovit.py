import math
from typing import Optional, List, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange

from timm.models.registry import register_model

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any


def load_backbone_from_checkpoint_into_fcn(
    ckpt_path: str,
    fcn_model: torch.nn.Module,
    *,
    ckpt_model_key: str = "model_ema",
    fallback_model_key: str = "model",
    src_backbone_prefix: Optional[str] = None,
    dst_backbone_attr: Optional[str] = "backbone",
    strip_prefixes: Tuple[str, ...] = ("module.",),
    ignore_if_contains: Tuple[str, ...] = ("classifier", "fc", "head", "heads", "linear", "logits"),
    map_location: str = "cpu",
    verbose: bool = True,
    # NEW (PyTorch 2.6+):
    trust_checkpoint: bool = False,   # if True, allow weights_only=False fallback
) -> Dict[str, Any]:
    """
    Transfer backbone weights from a classification checkpoint into an FCN/segmentation model.

    PyTorch 2.6+ note:
      - torch.load now defaults to weights_only=True.
      - If your checkpoint includes non-tensor objects (e.g., numpy scalars in args),
        weights_only=True may fail unless you allowlist certain globals.
      - If you TRUST the checkpoint source, set trust_checkpoint=True to allow
        weights_only=False fallback (can execute arbitrary code if malicious).
    """

    def _get_attr(root: torch.nn.Module, dotted: Optional[str]) -> torch.nn.Module:
        if dotted is None or dotted == "":
            return root
        obj = root
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return obj

    def _strip_any_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
        return key

    def _to_state_dict(obj) -> Dict[str, torch.Tensor]:
        if hasattr(obj, "state_dict"):
            return dict(obj.state_dict())
        if isinstance(obj, dict):
            return dict(obj)
        raise TypeError(f"Unsupported checkpoint model container type: {type(obj)}")

    # -------- robust torch.load for PyTorch 2.6+ --------
    def _torch_load_robust(path: str):
        # First try weights_only=True with safe globals allowlisted
        # (this addresses: numpy._core.multiarray.scalar)
        try:
            # add_safe_globals exists in modern PyTorch; fallback to context manager if needed
            add_safe = getattr(torch.serialization, "add_safe_globals", None)
            if add_safe is not None:
                add_safe([np._core.multiarray.scalar])  # allow numpy scalar
                return torch.load(path, map_location=map_location, weights_only=True)

            # context manager alternative
            safe_globals = getattr(torch.serialization, "safe_globals", None)
            if safe_globals is not None:
                with safe_globals([np._core.multiarray.scalar]):
                    return torch.load(path, map_location=map_location, weights_only=True)

            # if neither exists, just try plain load (older torch)
            return torch.load(path, map_location=map_location)

        except Exception as e_wo:
            # If weights_only path fails, only fallback to full unpickle if user trusts checkpoint
            if not trust_checkpoint:
                raise RuntimeError(
                    "torch.load failed under weights_only=True (PyTorch 2.6+ default).\n"
                    "Fix options:\n"
                    "  - If you trust the checkpoint source, call this function with trust_checkpoint=True\n"
                    "    (it will retry with weights_only=False).\n"
                    "  - Or keep trust_checkpoint=False and ensure required globals are allowlisted.\n"
                    f"Original error: {repr(e_wo)}"
                ) from e_wo

            # Trusted fallback:
            return torch.load(path, map_location=map_location, weights_only=False)

    ckpt = _torch_load_robust(ckpt_path)

    # pick weights container: EMA first, else model
    src = ckpt.get(ckpt_model_key, None)
    used_key = ckpt_model_key
    if src is None:
        src = ckpt.get(fallback_model_key, None)
        used_key = fallback_model_key
    if src is None:
        raise KeyError(
            f"Checkpoint has keys {list(ckpt.keys())} but neither '{ckpt_model_key}' nor '{fallback_model_key}' exists."
        )

    src_sd = _to_state_dict(src)

    # ---- prepare destination module/state
    dst_module = _get_attr(fcn_model, dst_backbone_attr)
    dst_sd = dict(dst_module.state_dict())

    # ---- normalize / filter source keys
    normalized_src = {}
    dropped_head = []
    for k, v in src_sd.items():
        k2 = _strip_any_prefix(k, strip_prefixes)

        if any(tok in k2 for tok in ignore_if_contains):
            dropped_head.append(k2)
            continue

        normalized_src[k2] = v

    # ---- optionally keep only backbone-prefixed keys (or infer)
    used_prefix = src_backbone_prefix
    if used_prefix is not None:
        tmp = {k[len(used_prefix):]: v for k, v in normalized_src.items() if k.startswith(used_prefix)}
        normalized_src = tmp
    else:
        # heuristic inference
        for p in ("backbone.", "model.backbone.", "encoder.", "features."):
            if any(k.startswith(p) for k in normalized_src.keys()):
                used_prefix = p
                normalized_src = {k[len(p):]: v for k, v in normalized_src.items() if k.startswith(p)}
                break

    # ---- match by key + shape
    matched = {}
    missing_in_dst = []
    shape_mismatch = []

    for k, v in normalized_src.items():
        if k not in dst_sd:
            missing_in_dst.append(k)
            continue
        if tuple(v.shape) != tuple(dst_sd[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(dst_sd[k].shape)))
            continue
        matched[k] = v

    # ---- load
    load_res = dst_module.load_state_dict(matched, strict=False)

    report = {
        "ckpt_path": ckpt_path,
        "used_ckpt_key": used_key,
        "used_src_backbone_prefix": used_prefix,
        "dst_backbone_attr": dst_backbone_attr,
        "trust_checkpoint": trust_checkpoint,
        "num_src_tensors": len(src_sd),
        "num_dst_tensors": len(dst_sd),
        "num_loaded": len(matched),
        "num_dropped_head_keys": len(dropped_head),
        "num_missing_in_dst": len(missing_in_dst),
        "num_shape_mismatch": len(shape_mismatch),
        "missing_keys_after_load": list(load_res.missing_keys),
        "unexpected_keys_after_load": list(load_res.unexpected_keys),
        "examples": {
            "loaded": list(matched.keys())[:25],
            "dropped_head": dropped_head[:25],
            "missing_in_dst": missing_in_dst[:25],
            "shape_mismatch": shape_mismatch[:10],
            "missing_keys_after_load": list(load_res.missing_keys)[:25],
            "unexpected_keys_after_load": list(load_res.unexpected_keys)[:25],
        },
    }

    if verbose:
        print("=== Backbone transfer report ===")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Using ckpt key: {used_key}")
        print(f"Source backbone prefix: {used_prefix!r} (None means 'not used / not found')")
        print(f"Destination module: {dst_backbone_attr or '<whole model>'}")
        print(f"Loaded tensors: {report['num_loaded']} / dst tensors: {report['num_dst_tensors']}")
        print(f"Dropped head-like keys: {report['num_dropped_head_keys']}")
        print(f"Missing-in-dst (pre-filter): {report['num_missing_in_dst']}")
        print(f"Shape mismatches: {report['num_shape_mismatch']}")
        if report["num_shape_mismatch"] > 0:
            k, s_src, s_dst = report["examples"]["shape_mismatch"][0]
            print(f"  Example shape mismatch: {k}: src{s_src} vs dst{s_dst}")
        print(f"Missing keys after load (strict=False): {len(report['missing_keys_after_load'])}")
        print(f"Unexpected keys after load (strict=False): {len(report['unexpected_keys_after_load'])}")

    return report


#source : ultralytics - https://docs.ultralytics.com/reference/nn/modules/conv/#ultralytics.nn.modules.conv.autopad
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


#CBN -> Convolution BatchNorm
#format : B C H W
class cbn(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=None, groups=1, bias=True, act=None):
        super(cbn, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, autopad(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = act() if act is not None else nn.Identity()
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x
        


class conv_rep(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, inference_mode = False, identity=True, branches = 1, act=nn.Identity()):
        super().__init__()
        self.inference_mode = inference_mode
        self.groups         = groups
        self.stride         = stride
        self.kernel_size    = kernel_size
        self.padding        = padding
        self.dilation       = dilation
        self.c_in           = c_in
        self.c_out          = c_out
        self.branches       = branches

        self.activation = act #fastest however not the best archi

        if inference_mode: 
            self.reparam_conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=True)
        else:
            #Re-parameterizable skip connection if possible !
            self.rbr_skip = nn.BatchNorm2d(c_in) if c_out == c_in and stride == 1 and identity else None

            #Re-parameterizable conv branches
            rbr_conv = []
            for _ in range(self.branches):
                rbr_conv.append(self._conv_bn(self.kernel_size, self.padding))
            
            self.rbr_conv = nn.ModuleList(rbr_conv)

            

    def forward(self, x):
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))
        
        #multi-branched train-time forward pass
        identity = 0
        if self.rbr_skip is not None:
            identity = self.rbr_skip(x)
        
        #Other branches
        out = identity
        for conv in self.rbr_conv:
            out += conv(x)
        
        return self.activation(out)

    def _get_kernel_bias(self):
        #getter for weights and bias of skip branch
        #by the way : it only happens once in the CNN part of our net
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        
        #getter for weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for i in range(self.branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[i])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        and
        https://github.com/apple/ml-mobileone/blob/main/mobileone.py

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c_in // self.groups
                kernel_value = torch.zeros((self.c_in,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.c_in):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def reparameterize(self):
        """
        Following works like 'RepVGG' and 'MobileOne'
        """
        if self.inference_mode:
            return
        
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(  self.rbr_conv[0].conv.in_channels,
                                        self.rbr_conv[0].conv.out_channels,
                                        self.rbr_conv[0].conv.kernel_size,
                                        self.rbr_conv[0].conv.stride,
                                        self.rbr_conv[0].conv.padding,
                                        dilation=self.rbr_conv[0].conv.dilation,
                                        groups  =self.rbr_conv[0].conv.groups,
                                        bias    =True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data   = bias

        """for param in self.parameters():
            param.detach_()"""
        if hasattr(self, 'rbr_conv'):
            self.__delattr__('rbr_conv')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.c_in,
                                              out_channels=self.c_out,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.c_out))
        return mod_list


class residual_separable_depthwise_rep(nn.Module):
    def __init__(self, c_in, c_out, stride=1, inference_mode=False, act=nn.GELU(), branches=1):
        super().__init__()
        self.dw = conv_rep(c_in, c_in, 7, stride, 7//2, groups=c_in, inference_mode=inference_mode, branches=branches)
        self.pw = conv_rep(c_in, c_out, 1, inference_mode=inference_mode, identity=False, branches=branches, act=act)
    def forward(self, x):
        out = self.dw(x) # f(x)[dw,bn] + x
        out = self.pw(out) #f(out)[pw,bn] + 0
        #We don't reparam the previous residuals as we want a global residual here
        #Also, to reparam a residual, we need to do it before the activation function... which is not working
        #well in this contribution

        #global residual if possible :
        if out.shape == x.shape:
            out = out + x
        return out

class inception_head(nn.Module):
    def __init__(self, c_in, c_out, inference_mode=False, branches=2, act=nn.SiLU()):
        super().__init__()
        self.c1 = conv_rep(c_in, c_out, 1, inference_mode=inference_mode, branches=branches, act=act)
        self.c2 = conv_rep(c_in, c_out, 1, inference_mode=inference_mode, branches=branches, act=act)

    def forward(self, x):
        #act(f(x)[pw,bn,identity]) + act(g(x)[pw,bn,identity])
        return self.c1(x) + self.c2(x)


class residual_separable_depthwise(nn.Module):
    def __init__(self, c_in, c_out, stride=1, bias=False):
        super().__init__()
        self.stride = stride
        self.dw = nn.Conv2d(c_in, c_in, 7, stride, 7//2, groups=c_in)
        #self.dw = RepDWBlock(c_in, 7, stride, 7//2)
        self.norm1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)
        self.norm2 = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        
    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.norm1(x)
        if x.shape == res.shape:
            x = x+res
        #x = self.pw(x)
        x = self.norm2(self.pw(x))
        x = self.act(x)
        if res.shape == x.shape:
            x = x + res
        #x = self.act(x)*x
        return x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, act=nn.GELU):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        act(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )




def conv_1x1_bn(c_in, c_out, act=nn.SiLU):
    return cbn(c_in, c_out, bias=False, act=act)

def conv_nxn_bn(c_in, c_out, kernel_size=3, stride=1, act=nn.SiLU, groups=1):
    #groups = c_in if c_in == c_out else 1
    return cbn(c_in, c_out, kernel_size, stride, kernel_size//2, groups=groups, bias=False, act=act)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

#GroupNorm, putting all channels into a single group (equivalent with LayerNorm)
#We don't use permute, therefore saving some computation time.
#inspiration : https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
#https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/norm.py#L53
class PreGroupNorm1(nn.Module):
    def __init__(self, dim, fn):
        super(PreGroupNorm1, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

def mvit_feedforward(dim, mlp_dim, dropout = 0., dense = nn.Linear, act=nn.SiLU):
    expansion_factor = mlp_dim // dim
    return FeedForward(dim, expansion_factor, dropout, dense, act)


class ConvMLP_feedforward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0., act=nn.SiLU):
        super().__init__()
        expansion_factor = mlp_dim // dim
        hidden_dim = int(dim * expansion_factor)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = act() if act is not None else nn.Identity()
        self.d1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.d2 = nn.Dropout(dropout)
    def forward(self, x):
        return self.d2(self.fc2(self.d1(self.act(self.fc1(x)))))


class mvit_scale_dot_product(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda t: rearrange(t,'b p n (h d) -> b p h n d', h = self.heads), qkv)


        dots = torch.matmul(q, k.transpose(-1,-2))*self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out), q, k, v


class linear_self_attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(embed_dim, 1+(2*embed_dim), 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
    def forward(self, x):
        qkv = self.qkv_proj(x)

        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

        context_scores = F.softmax(query, dim=-1)
        
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

class linear_transformer_block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,act=nn.SiLU):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreGroupNorm1(dim, linear_self_attention(dim)),
                PreGroupNorm1(dim, ConvMLP_feedforward(dim, mlp_dim, dropout, act=act))
            ]))
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class mvit_transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,act=nn.SiLU):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, mvit_scale_dot_product(dim, heads, dim_head, dropout)),
                PreNorm(dim, mvit_feedforward(dim, mlp_dim, dropout,act=act))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)[0] + x
            x = ff(x) + x
        return x


class MobileViTv4_linear_block(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., act=nn.SiLU):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        #tester local and global residual..
        self.local_convs = nn.Sequential(
            #Standard 3x3  ---- like MobileViTv1 
            #MobileViTv2 & v3 --> DEPTHWISE
            conv_nxn_bn(channel, channel, kernel_size, act=act, groups=channel),
            conv_1x1_bn(channel, dim, act=act)
        )

        self.transformer = linear_transformer_block(dim, depth, 4, 8, mlp_dim, dropout,act=act)

        #one pointwise before residual for fusion
        self.fusion_conv_1 = conv_1x1_bn(dim, channel,act=act)

        #MobileViTv1 and v2  --> Standard convolution after concatenation (double the cost of input channel)
        #self.fusion_conv_2 = conv_nxn_bn(2*channel, channel, kernel_size,act=act)

    def forward(self, x):
        y = x.clone()
        B,C,H,W = x.shape
        new_h, new_w = math.ceil(H/self.ph) * self.ph, math.ceil(W/self.pw) * self.pw
        num_patch_h, num_patch_w = new_h // self.ph, new_w // self.pw
        num_patches = num_patch_h * num_patch_w
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        
        #local representations
        x = self.local_convs(x)

        #Global representations
        #_,_,h,w = x.shape
        #x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw).contiguous()
        #B,C,H,W --> B,C,P,N
        C = x.shape[1]
        x = x.reshape(B,C,num_patch_h, self.ph, num_patch_w, self.pw).permute(0,1,3,5,2,4)
        x = x.reshape(B,C,-1,num_patches)
    
        x = self.transformer(x)

        #x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw).contiguous()
        x = x.reshape(B,C,self.ph, self.pw, num_patch_h, num_patch_w).permute(0,1,4,2,5,3)
        x = x.reshape(B,C,num_patch_h*self.ph, num_patch_w*self.pw)

        #Fusion
        x = self.fusion_conv_1(x)
        x = x + y

        return x
        
class MobileViTv4_block(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., act=nn.SiLU):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        #tester local and global residual..
        self.local_convs = nn.Sequential(
            #Standard 3x3  ---- like MobileViTv1 & v2
            #MobileViTv3 --> DEPTHWISE
            conv_nxn_bn(channel, channel, kernel_size, act=act, groups=channel),
            conv_1x1_bn(channel, dim, act=act)
        )

        self.transformer = mvit_transformer(dim, depth, 4, 8, mlp_dim, dropout,act=act)

        #one pointwise before residual for fusion
        self.fusion_conv_1 = conv_1x1_bn(dim, channel,act=act)

        #MobileViTv1 and v2  --> Standard convolution after concatenation (double the cost of input channel)
        #self.fusion_conv_2 = conv_nxn_bn(2*channel, channel, kernel_size,act=act)

    def forward(self, x):
        y = x.clone()

        #local representations
        x = self.local_convs(x)

        #Global representations
        _,_,h,w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw).contiguous()
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw).contiguous()

        #Fusion
        x = self.fusion_conv_1(x)
        x = x + y

        return x


class MobileViTv4(nn.Module):
    def __init__(self, image_size, dims, channels, nclasses=10, kernel_size=3, patch_size=[2,2,2], inference_mode=False):
        super().__init__()
        #assert image_size % patch_size == 0

        L = [2,4,3]

        p_ = 32

        self.stem = nn.Sequential(conv_nxn_bn(3, 32, stride=2, act=nn.SiLU), conv_1x1_bn(32,channels[0], act=nn.SiLU))
        #self.stem = nn.Sequential(conv_nxn_bn(3, 32, stride=2,act=nn.GELU), conv_1x1_bn(32,channels[0], act=nn.GELU))
        biases = True
        self.mbv2 = nn.ModuleList([])
        #relu6
        self.mbv2.append(residual_separable_depthwise_rep(channels[0], channels[1], inference_mode=inference_mode, act=nn.SiLU()))
        self.mbv2.append(residual_separable_depthwise_rep(channels[1], channels[2], stride=2,inference_mode=inference_mode, act=nn.SiLU()))
        self.mbv2.append(residual_separable_depthwise_rep(channels[2], channels[3],inference_mode=inference_mode, act=nn.SiLU()))
        self.mbv2.append(residual_separable_depthwise_rep(channels[2], channels[3],inference_mode=inference_mode, act=nn.SiLU()))
        self.mbv2.append(residual_separable_depthwise_rep(channels[3], channels[4], stride=2,inference_mode=inference_mode, act=nn.SiLU()))

        self.mbv2.append(residual_separable_depthwise_rep(channels[5], channels[6], stride=2,inference_mode=inference_mode, act=nn.SiLU()))
        self.mbv2.append(residual_separable_depthwise_rep(channels[7], channels[8], stride=2,inference_mode=inference_mode, act=nn.SiLU()))

        self.mvit = nn.ModuleList([])


        #full GELU
        self.mvit.append(MobileViTv4_block(dims[0], L[0], channels[5], kernel_size, patch_size[0], int(dims[0]*2), act=nn.SiLU))
        self.mvit.append(MobileViTv4_block(dims[1], L[1], channels[7], kernel_size, patch_size[1], int(dims[1]*4), act=nn.SiLU))
        self.mvit.append(MobileViTv4_block(dims[2], L[2], channels[9], kernel_size, patch_size[2], int(dims[2]*4), act=nn.SiLU))

        #self.out = RepHeadConv(channels[-2], channels[-1])
        #self.out = conv_rep(channels[-2], channels[-1], 1, inference_mode=inference_mode, act=nn.SiLU(), branches=3)
        self.out = inception_head(channels[-2], channels[-1], inference_mode=inference_mode, branches=2, act=nn.SiLU())

        self.pool = nn.AvgPool2d(image_size//p_, 1)
        self.head = nn.Linear(channels[-1], nclasses, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.mbv2[0](x)

        x = self.mbv2[1](x)
        x = self.mbv2[2](x)
        x = self.mbv2[3](x)

        x = self.mbv2[4](x)
        x = self.mvit[0](x)

        x = self.mbv2[5](x)
        x = self.mvit[1](x)

        x = self.mbv2[6](x)
        x = self.mvit[2](x)


        x = self.out(x)
        x = self.pool(x).contiguous().view(-1, x.shape[1])
        x = self.head(x)
        return x


#0,1 - 1,2 - 2,3 - 2,3 - 3,4 - 5,6, - 7-8
"""
16        self.mbv2.append(residual_separable_depthwise_rep(channels[0], channels[1], inference_mode=inference_mode, act=nn.SiLU()))
24        self.mbv2.append(residual_separable_depthwise_rep(channels[1], channels[2], stride=2,inference_mode=inference_mode, act=nn.SiLU()))
24        self.mbv2.append(residual_separable_depthwise_rep(channels[2], channels[3],inference_mode=inference_mode, act=nn.SiLU()))
24        self.mbv2.append(residual_separable_depthwise_rep(channels[2], channels[3],inference_mode=inference_mode, act=nn.SiLU()))
48        self.mbv2.append(residual_separable_depthwise_rep(channels[3], channels[4], stride=2,inference_mode=inference_mode, act=nn.SiLU()))

64        self.mbv2.append(residual_separable_depthwise_rep(channels[5], channels[6], stride=2,inference_mode=inference_mode, act=nn.SiLU()))
80        self.mbv2.append(residual_separable_depthwise_rep(channels[7], channels[8], stride=2,inference_mode=inference_mode, act=nn.SiLU()))

"""

@register_model
def NanoVit_XXS(pretrained=False, img_size=256, inference_mode=False, **kwargs):
    model = MobileViTv4(img_size, dims=[64, 80, 96], patch_size=[2,2,1], channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320], nclasses=1000, inference_mode=inference_mode)
    return model


@register_model
def NanoVit_XS(pretrained=False,  img_size=256,inference_mode=False, **kwargs):
    model = MobileViTv4(img_size, dims=[96, 120, 144], patch_size=[2,2,1], channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], nclasses=1000,  inference_mode=inference_mode)
    return model

@register_model#old one : [96,120,144]
def NanoVit_S(pretrained=False,  img_size=256,inference_mode=False, **kwargs):
    model = MobileViTv4(img_size, dims=[128, 144, 180], patch_size=[2,2,1], channels=[16, 32, 48, 48, 96, 96, 160, 160, 160, 160, 640], nclasses=1000,  inference_mode=inference_mode)
    return model