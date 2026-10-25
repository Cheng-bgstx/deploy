"""dp_v1 network: depth ResNet-18 tokens + history tokens -> DiT1d over the motion chunk (x0 prediction).

Shapes: depth (B,N_DEPTH,2,120,160) [inverse depth, mask]; depth_age_s (B,N_DEPTH); history (B,9,41);
history_mask (B,9); go_clock (B,); x (B,T,38) normalized noisy/prior chunk; tau (B,T) per-token flow time;
prefix (B,T) / prior_mask (B,T) flags. Output: normalized x0 estimate (B,T,38).
Per-token AdaLN-Zero (each token gets global condition + its own time embedding, as tokens are at
different noise levels under the soft-RTC prior); cross-attention to the 89 observation tokens; RoPE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import resnet as torchvision  # sim2real: torchvision-free ResNet-18, see SOURCES.json

from . import contract as C


def resnet18_gn(in_ch: int) -> nn.Module:
    net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
    net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    net.fc = nn.Identity()
    net.avgpool = nn.Identity()
    return net


class DepthEncoder(nn.Module):
    """(B,N,2,H,W) -> (B, N*20, d): 4x5 spatial tokens per frame + frame/age embeddings."""

    def __init__(self, d: int):
        super().__init__()
        self.backbone = resnet18_gn(2)
        self.proj = nn.Linear(512, d)
        self.spatial = nn.Parameter(torch.zeros(1, 1, 20, d))
        self.frame = nn.Parameter(torch.zeros(1, C.N_DEPTH, 1, d))
        self.age = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.spatial, std=.02)
        nn.init.normal_(self.frame, std=.02)

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(M,2,H,W) -> (M,20,d) frame features (cacheable per frame at deployment)."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))                 # (M,512,4,5)
        return self.proj(x.flatten(2).transpose(1, 2))

    def assemble(self, feats: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """(B,N,20,d) per-frame features (oldest first) + ages (B,N) -> (B,N*20,d) tokens."""
        f = feats + self.spatial + self.frame + self.age(age.unsqueeze(-1)).unsqueeze(2)
        return f.flatten(1, 2)

    def forward(self, depth: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        B, N = depth.shape[:2]
        return self.assemble(self.encode_frames(depth.flatten(0, 1)).view(B, N, 20, -1), age)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float().unsqueeze(-1) * 1000 * freqs
    return torch.cat([a.cos(), a.sin()], -1)


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary embedding over the sequence axis of (B,heads,T,hd)."""
    T, hd = x.shape[-2], x.shape[-1]
    half = hd // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    ang = torch.arange(T, device=x.device).float()[:, None] * freqs
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.h = heads
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(approximate='tanh'), nn.Linear(mlp_ratio * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, -1).transpose(1, 2)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, c: torch.Tensor, cond_mask: torch.Tensor | None):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(c).chunk(9, -1)          # each (B,T,d)
        B, T, d = x.shape
        h = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).chunk(3, -1)
        q, k, v = rope(self._heads(q)), rope(self._heads(k)), self._heads(v)
        x = x + g1 * self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, d))
        h = self.n2(x) * (1 + s2) + b2
        k2, v2 = self.kv2(cond_tokens).chunk(2, -1)
        att_mask = None if cond_mask is None else cond_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(self._heads(self.q2(h)), self._heads(k2), self._heads(v2), attn_mask=att_mask)
        x = x + g2 * self.o2(a.transpose(1, 2).reshape(B, T, d))
        h = self.n3(x) * (1 + s3) + b3
        return x + g3 * self.mlp(h)


class DPModel(nn.Module):
    def __init__(self, d: int = 384, layers: int = 8, heads: int = 6, chunk: int = C.CHUNK):
        super().__init__()
        self.d, self.chunk = d, chunk
        self.depth = DepthEncoder(d)
        self.hist = nn.Sequential(nn.Linear(C.HISTORY_DIM + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.hist_pos = nn.Parameter(torch.zeros(1, C.HISTORY_N, d))
        self.glob = nn.Sequential(nn.Linear(C.HISTORY_DIM + 2, d), nn.SiLU(), nn.Linear(d, d))
        self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Linear(C.MOTION_DIM, d)
        self.flags = nn.Embedding(3, d)                 # 0 free, 1 soft prior, 2 hard prefix
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, C.MOTION_DIM)
        nn.init.normal_(self.hist_pos, std=.02)
        for m in (self.out_ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def observe(self, depth, depth_age_s, history, history_mask, go_clock, depth_feats=None):
        """Observation tokens (B,89,d) and global condition (B,d); reusable across flow steps.
        ``depth_feats`` (B,N,20,d) from DepthEncoder.encode_frames replaces ``depth`` (runtime frame cache)."""
        dt = self.depth(depth, depth_age_s) if depth_feats is None else self.depth.assemble(depth_feats, depth_age_s)
        m = history_mask.to(history.dtype).unsqueeze(-1)
        ht = self.hist(torch.cat([history, m], -1)) + self.hist_pos
        tokens = torch.cat([dt, ht], 1)
        g = self.glob(torch.cat([history[:, -1], m[:, -1], go_clock.unsqueeze(-1).to(history.dtype)], -1))
        return tokens, g

    def denoise(self, tokens, g, x, tau, prefix, prior_mask):
        flags = torch.where(prefix, 2, prior_mask.long())
        c = g.unsqueeze(1) + self.temb(timestep_embedding(tau, 256))
        h = self.inp(x) + self.flags(flags)
        for blk in self.blocks:
            h = blk(h, tokens, c, None)
        sh, sc = self.out_ada(c).chunk(2, -1)
        return self.out(self.out_norm(h) * (1 + sc) + sh)

    def forward(self, depth, depth_age_s, history, history_mask, go_clock, x, tau, prefix, prior_mask):
        tokens, g = self.observe(depth, depth_age_s, history, history_mask, go_clock)
        return self.denoise(tokens, g, x, tau, prefix, prior_mask)
