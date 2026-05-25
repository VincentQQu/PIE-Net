"""
PIE-Net Model Architecture

Probabilistic Intensity-Event Modeling for High Quality Event-Based Video Generation.

Two pretrained variants are shipped:
  - PIE-Net      — full model, best overall quality
  - PIE-Net-Lite — 2× fewer params, faster inference
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VariantName = Literal["pie-net", "pie-net-lite"]

# ----------------------------
# Pretrained variant configs
# ----------------------------

CONFIG_PIENET = {
    "n_lyr": 3,
    "n_chs": [8, 16, 24, 32],
    "n_rep_dn": [0, 0, 0],
    "n_rep_up": [1, 1, 1],
    "pool_szs": [2, 2, 5],
    "lstm_types": ["fused", "fused", "fused", "fused"],
    "lstm_expans": [2, 2, 2],
    "dn_MBC_types": ["normal", "normal", "normal", "normal"],
    "dn_expans": [2, 4, 4],
    "up_MBC_types": ["normal", "normal", "normal", "normal"],
    "up_expans": [2, 4, 4],
    "use_mcse": True,
    "use_ugsg": True,
}

CONFIG_PIENET_LITE = {
    "n_lyr": 2,
    "n_chs": [12, 16, 24, 32],
    "n_rep_dn": [1, 1, 1],
    "n_rep_up": [1, 1, 1],
    "pool_szs": [2, 2, 2],
    "lstm_types": ["fused", "fused", "fused", "fused"],
    "lstm_expans": [2, 2, 2],
    "dn_MBC_types": ["normal", "normal", "normal", "normal"],
    "dn_expans": [2, 4, 4],
    "up_MBC_types": ["normal", "normal", "normal", "normal"],
    "up_expans": [2, 4, 4],
    "use_mcse": True,
    "use_ugsg": True,
}

VARIANT_REGISTRY: Dict[str, Dict] = {
    "pie-net": {
        "config": CONFIG_PIENET,
        "weight_file": "model.pth",
        "params": "~154K",
        "flops_240x180": "1.59G",
        "encoder_layers": 3,
    },
    "pie-net-lite": {
        "config": CONFIG_PIENET_LITE,
        "weight_file": "model_lite.pth",
        "params": "~79K",
        "flops_240x180": "1.58G",
        "encoder_layers": 2,
    },
}

_VARIANT_ALIASES = {
    "pie-net": "pie-net",
    "pienet": "pie-net",
    "full": "pie-net",
    "standard": "pie-net",
    "pie-net-lite": "pie-net-lite",
    "pienet-lite": "pie-net-lite",
    "pienet_lite": "pie-net-lite",
    "lite": "pie-net-lite",
}


def resolve_variant(variant: str) -> str:
    key = variant.lower().replace("_", "-")
    if key not in _VARIANT_ALIASES:
        available = ", ".join(sorted(VARIANT_REGISTRY))
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {available}")
    return _VARIANT_ALIASES[key]


def list_variants() -> Dict[str, Dict]:
    """Return metadata for all shipped pretrained variants."""
    return VARIANT_REGISTRY.copy()


# ----------------------------
# Building Blocks
# ----------------------------

def _act():
    return nn.SiLU()


def _gn(ch):
    return nn.GroupNorm(1, ch)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, act=True):
        super().__init__()
        self.convna = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                padding=kernel_size // 2, padding_mode="zeros",
                groups=groups, bias=True,
            ),
            _gn(out_ch),
        )
        if act:
            self.convna.add_module("act", _act())

    def forward(self, x):
        return self.convna(x)


class SE_Block_conv(nn.Module):
    def __init__(self, c, squeeze_ratio=8):
        super().__init__()
        sq = max(1, c // squeeze_ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, sq, 1, bias=False)
        self.fc2 = nn.Conv2d(sq, c, 1, bias=False)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = self.avg(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.sig(s)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, MBC_type="fused", expansion=4):
        super().__init__()
        exp = in_ch * expansion
        self.use_res = in_ch == out_ch

        if MBC_type == "depthwise":
            self.mbconv = nn.Sequential(
                ConvNormAct(in_ch, exp, 1),
                ConvNormAct(exp, exp, 3, groups=exp),
                SE_Block_conv(exp),
                ConvNormAct(exp, out_ch, 1, act=False),
            )
        elif MBC_type == "fused":
            self.mbconv = nn.Sequential(
                ConvNormAct(in_ch, exp, 3),
                SE_Block_conv(exp),
                ConvNormAct(exp, out_ch, 1, act=False),
            )
        else:
            self.mbconv = ConvNormAct(in_ch, out_ch, 3)

    def forward(self, x):
        y = self.mbconv(x)
        if self.use_res:
            y = y + x
        return y


class ConvRNN_AttnDeform(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, gate_kernel="fused", expansion=2):
        super().__init__()
        self.hidden_size = hidden_size
        in_ch = input_size + hidden_size
        self.Gates = MBConv(in_ch, 4 * hidden_size, gate_kernel, expansion=2)
        self.norm_h = nn.GroupNorm(4, hidden_size)
        self.norm_c = nn.GroupNorm(4, hidden_size)

    def forward(self, x, prev_state=None):
        batch, _, height, width = x.shape
        if prev_state is None:
            h_prev = x.new_zeros(batch, self.hidden_size, height, width)
            c_prev = x.new_zeros(batch, self.hidden_size, height, width)
        else:
            h_prev, c_prev = prev_state

        gates = self.Gates(torch.cat([x, h_prev], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = self.norm_c(f * c_prev + i * g)
        h = self.norm_h(o * torch.tanh(c))
        return h, c


class ModalityFiLM_Fused(nn.Module):
    def __init__(self, ch_feat, ch_desc):
        super().__init__()
        self.film = nn.Conv2d(2 * ch_desc, 2 * ch_feat, 1, bias=True)

    def forward(self, feat, e_desc, f_desc):
        desc = torch.cat([e_desc, f_desc], dim=1)
        gamma, beta = self.film(desc).chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


class UnifiedStemV3(nn.Module):
    def __init__(self, in_ch_event, in_ch_frame, out_ch):
        super().__init__()
        self.event_branch = MBConv(in_ch_event, out_ch, MBC_type="normal", expansion=2)
        self.frame_branch = MBConv(in_ch_frame, out_ch, MBC_type="normal", expansion=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_event, x_frame, return_desc=True):
        e = self.event_branch(x_event)
        f = self.frame_branch(x_frame)
        x = e + f
        if not return_desc:
            return x
        return x, self.pool(e), self.pool(f)


class DownConvRNN_ModalityV3(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        lstm_type="fused",
        lstm_expan=2,
        pool=2,
        MBC_type="fused",
        expansion=2,
        n_repeats=0,
        use_mcse=True,
        desc_ch=None,
    ):
        super().__init__()
        self.mbd = ConvRNN_AttnDeform(
            in_ch, out_ch, gate_kernel=lstm_type, expansion=lstm_expan,
        )

        blocks = []
        for i in range(n_repeats):
            blocks.append((
                f"mbconv_{i}",
                MBConv(out_ch, out_ch, MBC_type=MBC_type, expansion=expansion),
            ))
        self.out = nn.Sequential(OrderedDict(blocks)) if blocks else nn.Identity()

        self.pool = nn.MaxPool2d(pool)
        self.use_mcse = use_mcse
        self.film = ModalityFiLM_Fused(out_ch, desc_ch) if use_mcse and desc_ch else None

    def forward(self, x, p_state=None, e_desc=None, f_desc=None):
        h, c = self.mbd(x, p_state)
        y = self.out(h)
        if self.film is not None and e_desc is not None and f_desc is not None:
            y = self.film(y, e_desc, f_desc)
        return self.pool(y), (h, c), y


class UncertaintyGate(nn.Module):
    def __init__(self, skip_ch):
        super().__init__()
        self.gate_conv = nn.Conv2d(1, skip_ch, kernel_size=1, bias=True)

    def forward(self, skip, var_map):
        var_resized = F.interpolate(
            var_map, size=skip.shape[-2:], mode="bilinear", align_corners=True,
        )
        return skip * torch.sigmoid(self.gate_conv(var_resized))


class ShallowUpV3(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        scale_factor=2,
        MBC_type="normal",
        expansion=2,
        n_repeats=1,
        use_ugsg=True,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        blocks = [MBConv(in_ch + skip_ch, out_ch, MBC_type=MBC_type, expansion=expansion)]
        for _ in range(n_repeats - 1):
            blocks.append(MBConv(out_ch, out_ch, MBC_type=MBC_type, expansion=expansion))
        self.fuse = nn.Sequential(*blocks)

        self.use_ugsg = use_ugsg
        self.ugate = UncertaintyGate(skip_ch) if use_ugsg else None

    def forward(self, x1, x2, var_map=None):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        if self.ugate is not None and var_map is not None:
            x2 = self.ugate(x2, var_map)
        return self.fuse(torch.cat([x2, x1], dim=1))


class SingleEncoderEffUNet_v3(nn.Module):
    """Efficient U-Net with recurrent encoder for event-based reconstruction."""

    def __init__(
        self,
        input_size_event=5,
        input_size_frame=1,
        output_size=3,
        params=None,
    ):
        super().__init__()
        self.params = (params or CONFIG_PIENET).copy()
        n_lyr = self.params["n_lyr"]
        n_chs = self.params["n_chs"]

        self.setup = {"n_lyr": n_lyr}
        self.stem = UnifiedStemV3(input_size_event, input_size_frame, n_chs[0])

        self.downs = nn.ModuleList([
            DownConvRNN_ModalityV3(
                n_chs[i], n_chs[i + 1],
                lstm_type=self.params["lstm_types"][i],
                lstm_expan=self.params["lstm_expans"][i],
                pool=self.params["pool_szs"][i],
                MBC_type=self.params["dn_MBC_types"][i],
                expansion=self.params["dn_expans"][i],
                n_repeats=self.params["n_rep_dn"][i],
                use_mcse=self.params["use_mcse"],
                desc_ch=n_chs[0],
            )
            for i in range(n_lyr)
        ])

        self.ups = nn.ModuleList([
            ShallowUpV3(
                n_chs[i + 1], n_chs[i + 1], n_chs[i],
                scale_factor=self.params["pool_szs"][i],
                MBC_type=self.params["up_MBC_types"][i],
                expansion=self.params["up_expans"][i],
                n_repeats=self.params["n_rep_up"][i],
                use_ugsg=self.params["use_ugsg"],
            )
            for i in range(n_lyr - 1, -1, -1)
        ])

        self.outc = nn.Conv2d(n_chs[0], output_size, kernel_size=1)

    def forward(self, x_event, x_frame, p_states=None, var_map=None):
        n_lyr = self.setup["n_lyr"]
        if p_states is None:
            p_states = [None] * n_lyr

        x, e_desc, f_desc = self.stem(x_event, x_frame)
        skips, new_states = [], []
        for i, down in enumerate(self.downs):
            x, state, skip = down(x, p_states[i], e_desc, f_desc)
            skips.append(skip)
            new_states.append(state)

        for i, up in enumerate(self.ups):
            skip = skips[n_lyr - 1 - i]
            x = up(x, skip, var_map=var_map)

        return self.outc(x), new_states


class Z2F1(nn.Module):
    """Transform latent space to intensity prediction with uncertainty."""

    def forward(self, mean_exp_z, var_exp_z, k, f0):
        c = f0 + k
        return c * mean_exp_z - k, (c ** 2) * var_exp_z


class SoftClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad):
        return grad


class PIENet(nn.Module):
    """
    PIE-Net: Probabilistic Intensity-Event Network.

    Reconstructs high-quality intensity frames from event voxel grids with
    per-pixel uncertainty estimation via Probabilistic Intensity-Event Mapping (PIEM).

    Args:
        variant: ``"pie-net"`` (default, full model) or ``"pie-net-lite"``.
        input_size_event: Number of temporal bins in the event voxel grid.
        input_size_frame: Number of input frame channels.
    """

    DEFAULT_VARIANT = "pie-net"

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        input_size_event: int = 5,
        input_size_frame: int = 1,
    ):
        super().__init__()
        self.variant = resolve_variant(variant)
        meta = VARIANT_REGISTRY[self.variant]

        self.model = SingleEncoderEffUNet_v3(
            input_size_event=input_size_event,
            input_size_frame=input_size_frame,
            output_size=3,
            params=meta["config"],
        )

        self.posify = nn.ReLU()
        self.cal_k = MBConv(3, 1, MBC_type="normal", expansion=2)
        self.Z_to_f1 = Z2F1()

        self.f0 = None
        self.p_states = None
        self.prev_var = None

    @property
    def variant_info(self) -> Dict:
        return VARIANT_REGISTRY[self.variant].copy()

    def forward(self, events: torch.Tensor) -> dict:
        if self.f0 is None:
            batch, _, height, width = events.shape
            self.f0 = torch.zeros(
                batch, 1, height, width, device=events.device, dtype=events.dtype,
            )

        feats, new_p_states = self.model(
            x_event=events,
            x_frame=self.f0,
            p_states=self.p_states,
            var_map=self.prev_var,
        )

        k = self.posify(self.cal_k(feats))
        mean_exp_z = self.posify(feats[:, 0:1])
        var_exp_z = self.posify(feats[:, 1:2])
        f0_ref = self.posify(feats[:, 2:3])

        mean_f1, var_f1 = self.Z_to_f1(mean_exp_z, var_exp_z, k, f0_ref)
        mean_f1 = SoftClamp.apply(mean_f1)

        self.f0 = mean_f1.detach()
        self.p_states = [(h.detach(), c.detach()) for h, c in new_p_states]
        self.prev_var = var_f1.detach()

        return {"image": mean_f1, "var": var_f1}

    def reset_states(self):
        """Reset internal streaming states. Call between sequences."""
        self.f0 = None
        self.p_states = None
        self.prev_var = None


class PIENetLite(PIENet):
    """PIE-Net-Lite: faster 2-layer variant with half the parameters."""

    DEFAULT_VARIANT = "pie-net-lite"

    def __init__(self, input_size_event: int = 5, input_size_frame: int = 1):
        super().__init__(
            variant=self.DEFAULT_VARIANT,
            input_size_event=input_size_event,
            input_size_frame=input_size_frame,
        )


def _pretrained_path(variant: str) -> str:
    meta = VARIANT_REGISTRY[resolve_variant(variant)]
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "pretrained", meta["weight_file"])


def load_model(
    pretrained: bool = True,
    device: Union[str, torch.device] = "cuda",
    variant: str = "pie-net",
) -> PIENet:
    """
    Load a PIE-Net model with optional pretrained weights.

    Args:
        pretrained: Load shipped checkpoint weights.
        device: Target device (``"cuda"`` or ``"cpu"``).
        variant: ``"pie-net"`` (default) or ``"pie-net-lite"`` / ``"lite"``.

    Returns:
        Configured :class:`PIENet` instance.
    """
    resolved = resolve_variant(variant)
    model: PIENet = PIENetLite() if resolved == "pie-net-lite" else PIENet()

    if pretrained:
        weight_path = _pretrained_path(resolved)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Pretrained weights not found at {weight_path}. "
                "Reinstall with: pip install pie-net"
            )
        state_dict = torch.load(weight_path, map_location=device)
        missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Checkpoint incompatible with {resolved}: missing keys {missing[:5]}"
            )

    return model.to(device)


def load_model_lite(
    pretrained: bool = True,
    device: Union[str, torch.device] = "cuda",
) -> PIENetLite:
    """Convenience loader for PIE-Net-Lite."""
    return load_model(pretrained=pretrained, device=device, variant="pie-net-lite")  # type: ignore[return-value]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
