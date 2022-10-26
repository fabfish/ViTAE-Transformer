""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .src.models.layers.fastlinear import FastLinear, ButterflyLinear, RandomLinear, SLLinear, \
#     SLXLinear, TopkLinear, TopkLrLinear, ButterflyGlobalLinear, NinjaTurtleLinear
# from .src.models.layers.maskedlinear import MaskLinearWrap
import math
from einops import rearrange

import hydra

from .src.ops.butterfly_factor import butterfly_factor_to_matrix

from .src.models.layers.monarch_linear import MonarchLinear

@torch.jit.script
def bias_gelu_scripted(x, bias):
    return F.gelu(x + bias)


class MonarchMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 act_fn=None, drop=0., drop_btw_fcs=True, linear1_cfg=None, linear2_cfg=None):
        """TD [2021-10-27] act_fn takes precedence over act_layer if set.
        This is to support Pytorch 1.10 Transformer interface that construct the activation
        *function*, not the activation *layer*.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # if linear1_cfg is None:
        #     self.fc1 = nn.Linear(hidden_features, out_features)
        # else:
        #     # self.fc1 = hydra.utils.instantiate(linear1_cfg, in_features, hidden_features,
        #     #                                     _recursive_=False)
        #     self.fc1 = MonarchLinear(in_features, hidden_features, _recursive_=False, nblocks = 4)
        self.fc1 = MonarchLinear(in_features, hidden_features, nblocks = 4)
        self.act = act_layer() if act_fn is None else act_fn
        # if linear2_cfg is None:
        #     self.fc2 = nn.Linear(hidden_features, out_features)
        # else:
        #     # self.fc2 = hydra.utils.instantiate(linear2_cfg, hidden_features, out_features,
        #     #                                    _recursive_=False)
        #     self.fc2 = MonarchLinear(hidden_features, out_features, _recursive_=False, nblocks = 4)
        self.fc2 = MonarchLinear(hidden_features, out_features, nblocks = 4)
        self.drop = nn.Dropout(drop)
        self.drop_btw_fcs = drop_btw_fcs
        # TD [2022-01-08] bias_gelu_scripted was working on Pytorch 1.10.1 but stops
        # working on Pytorch 1.11.0a0+b6df043 (nvcr.io pytorch 21.12) with error
        # RuntimeError: MALFORMED INPUT: Unhandled node kind (in computeValue): aten::gelu
        # So I'm disabling fused_bias_gelu for now
        # self._fused_bias_gelu = ((act_fn is F.gelu or act_layer is nn.GELU)
        #                          and self.fc1.bias is not None
        #                          and hasattr(self.fc1, 'forward_matmul'))
        self._fused_bias_gelu = False

    def forward(self, x):
        # print(self)
        # print(self.fc1)
        # print(x)
        # print(x.shape)
        # input shape is [64(batch size), 196, 384]
        if self._fused_bias_gelu and x.is_cuda:
            x = self.fc1.forward_matmul(x)
            x = bias_gelu_scripted(x, self.fc1.bias.to(dtype=x.dtype))
        else:
            x = self.fc1(x)
            x = self.act(x)
        if self.drop_btw_fcs:
            x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MonarchAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = MonarchLinear(dim, dim * 3, nblocks=4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MonarchLinear(dim, dim, nblocks=4)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MonarchAttentionPerformer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5):
        super().__init__()
        self.emb = dim * num_heads # we use 1, so it is no need here
        self.kqv = MonarchLinear(dim, 3 * self.emb, nblocks=4)
        self.dp = nn.Dropout(proj_drop)
        self.proj = MonarchLinear(self.emb, self.emb, nblocks=4)
        self.head_cnt = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.epsilon = 1e-8  # for stable in division
        self.drop_path = nn.Identity()

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.attn(x)
        return x