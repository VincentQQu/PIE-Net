#!/usr/bin/env python3
"""Minimal inference example for PIE-Net and PIE-Net-Lite."""

import torch
from pie_net import load_model, load_model_lite, count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- PIE-Net (full) ---
model = load_model(pretrained=True, device=device, variant="pie-net")
model.eval()
print(f"PIE-Net params: {count_parameters(model):,}")

events = torch.randn(1, 5, 180, 240, device=device)
with torch.no_grad():
    out = model(events)
print("PIE-Net output:", out["image"].shape, out["var"].shape)

model.reset_states()

# --- PIE-Net-Lite ---
lite = load_model_lite(pretrained=True, device=device)
lite.eval()
print(f"PIE-Net-Lite params: {count_parameters(lite):,}")

with torch.no_grad():
    out_lite = lite(events)
print("PIE-Net-Lite output:", out_lite["image"].shape, out_lite["var"].shape)
