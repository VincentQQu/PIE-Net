# PIE-Net

**Probabilistic Intensity-Event Modeling for High-Quality Event-Based Video Generation**

[![PyPI version](https://img.shields.io/pypi/v/event-pienet.svg)](https://pypi.org/project/event-pienet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Turn asynchronous **event camera** streams into **high-quality grayscale video** in real time — with **per-pixel uncertainty maps** and a principled probabilistic formulation grounded in event camera physics.

<p align="center">
  <b>Two models. One pip install. Ready for research and deployment.</b>
</p>

---

## Highlights

| Feature | Description |
|---------|-------------|
| **Probabilistic reconstruction** | PIEM maps polarity events to intensity via a closed-form lognormal model |
| **Uncertainty-aware** | Every pixel gets a confidence map — useful for downstream robotics & vision |
| **Real-time capable** | 30+ FPS on modern GPUs; Lite variant for edge devices |
| **Tiny footprint** | 154K params (full) / 79K params (lite) — orders of magnitude smaller than competitors |
| **Plug & play** | Pretrained weights ship with the package — no manual download |
| **Benchmark-ready** | EVREAL configs included for ECD, MVSEC, and HQF |
| **PIEM representation export** | Full 5-channel learned representation for downstream tasks (following the idea of EvRepSL) |

---

## Model Zoo

Two pretrained variants are included:

| | **PIE-Net** | **PIE-Net-Lite** |
|---|-------------|------------------|
| **Encoder depth** | 3 layers | 2 layers |
| **Parameters** | 154K | 79K |
| **FLOPs @ 240×180** | 1.59G | 1.58G |
| **Best for** | Highest quality | Speed & edge deployment |

### Benchmark performance (EVREAL eval)

Metrics from the shipped checkpoints on standard benchmarks:

#### PIE-Net

| Dataset | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---------|-------|--------|---------|
| **IJRR (ECD)** | 0.0257 | 0.6122 | 0.1957 |
| **MVSEC** | 0.0484 | 0.3798 | 0.4356 |
| **HQF** | 0.0204 | 0.6302 | 0.2248 |

#### PIE-Net-Lite

| Dataset | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---------|-------|--------|---------|
| **IJRR (ECD)** | 0.0221 | 0.6197 | 0.2079 |
| **MVSEC** | 0.0428 | 0.3889 | 0.4418 |
| **HQF** | 0.0267 | 0.5993 | 0.2494 |

> **PIE-Net** leads on perceptual quality (LPIPS) and HQF. **PIE-Net-Lite** wins on IJRR MSE/SSIM with half the parameters — ideal when latency matters.

---

## Installation

### From PyPI (recommended)

```bash
pip install event-pienet
```

### With optional dependencies

```bash
# Real-time event camera demo (DVS / DAVIS)
pip install event-pienet[realtime]

# Benchmark evaluation helpers
pip install event-pienet[eval]

# Everything
pip install event-pienet[all]
```

### From source

```bash
git clone https://github.com/VincentQQu/pie-net.git
cd pie-net
pip install -e .
```

### CUDA PyTorch

Install PyTorch with CUDA support first if you have a GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install event-pienet
```

---

## Quick Start

### Python API

```python
import torch
from pie_net import load_model, load_model_lite, stack_piem_representation

# PIE-Net (full model — default)
model = load_model(pretrained=True, device="cuda")
model.eval()

# PIE-Net-Lite (faster, smaller)
lite = load_model_lite(pretrained=True, device="cuda")
# or: lite = load_model(variant="pie-net-lite", device="cuda")

events = torch.randn(1, 5, 180, 240).cuda()  # [B, bins, H, W]

with torch.no_grad():
    output = model(events)

    # Reconstruction (backward-compatible keys)
    frame = output["image"]       # alias for mean_f1 — [1, 1, H, W]
    uncertainty = output["var"]   # alias for var_f1  — [1, 1, H, W]

    # Full PIEM latent maps (for representation / downstream tasks)
    mean_z = output["mean_exp_z"]  # expected log-intensity change
    var_z  = output["var_exp_z"]   # uncertainty of Z
    k      = output["k"]             # PIEM scaling parameter
    mean_f1 = output["mean_f1"]    # reconstructed intensity
    var_f1  = output["var_f1"]     # frame uncertainty

    # Stack into a 5-channel PIEM representation [1, 5, H, W]
    piem_rep = stack_piem_representation(output)

model.reset_states()  # call between sequences
```

**Output keys** (since v1.1.2):

| Key | Shape | Description |
|-----|-------|-------------|
| `mean_exp_z` | `[B, 1, H, W]` | Expected log-intensity change (Z mean) |
| `var_exp_z` | `[B, 1, H, W]` | Uncertainty of Z |
| `k` | `[B, 1, H, W]` | Learned PIEM scaling parameter |
| `mean_f1` / `image` | `[B, 1, H, W]` | Reconstructed intensity frame |
| `var_f1` / `var` | `[B, 1, H, W]` | Per-pixel frame uncertainty |

Use `stack_piem_representation(output)` to combine the five PIEM maps into a single `[B, 5, H, W]` tensor — compatible with [EvRepSL](https://github.com/VincentQQu/EvRepSL) representation pipelines.

### Real-time demo (event camera)

Connect a DVS/DAVIS camera and run:

```bash
# PIE-Net (default, best quality)
pie-net-demo

# PIE-Net-Lite (faster)
pie-net-demo --variant pie-net-lite

# Options
pie-net-demo --variant pie-net --no-visualize-voxel --use-amp --frame-interval 16
```

Or via the script:

```bash
python -m pie_net.demo --variant pie-net-lite
```

Press **q** to quit.

---

## Method Overview

Event cameras do not capture full frames at fixed intervals. They asynchronously report pixel-level brightness changes as **events** `(x, y, polarity, t)`. Given a previous intensity frame and the events that follow, PIE-Net reconstructs the next frame using **Probabilistic Intensity-Event Mapping (PIEM)**.

```text
Previous frame + Event stream  →  PIE-Net  →  Reconstructed next frame + uncertainty
```

### Core idea

Events describe intensity changes: positive events mean brightness increased, negative events mean it decreased. By accumulating polarity-weighted events over time, we estimate how much each pixel's intensity has changed.

Real event data is noisy — thresholds vary across pixels and some events are unreliable. PIEM therefore models intensity change **probabilistically**, estimating both the reconstructed image and a per-pixel uncertainty map.

### Probabilistic Intensity-Event Mapping (PIEM)

PIEM links events to frame reconstruction in three steps:

1. **Accumulate events** — count positive and negative events per pixel to estimate log-intensity change
2. **Model uncertainty** — treat event counts and thresholds as uncertain, yielding a latent change variable `Z` with mean `μZ` and variance `σZ²`
3. **Reconstruct the next frame** — apply the probabilistic intensity change to the previous (or refined) frame:

```text
next frame ≈ previous frame × event-based intensity change
```

### PIE-Net architecture

PIE-Net estimates the probabilistic variables required by PIEM. It has two main parts:

**Probabilistic Event Priors Estimator (PEPE)** — a dual-branch encoder that takes a voxel-grid event tensor and the previous intensity frame, fuses motion/change and appearance features, and outputs `μZ`, `σZ²`, and a refined previous-frame representation.

**Probabilistic Intensity-Event Mapper** — applies PIEM to map `μZ`, `σZ²`, and the refined frame to the final reconstruction.

```
Event Voxel Grid [B, 5, H, W]  +  Previous Frame [B, 1, H, W]
        ↓
   Dual Stem (Event + Intensity)  →  Recurrent Encoder + MCSE
        ↓
   Decoder + UGSG (uncertainty-guided skip gating)
        ↓
   PIEM Head  →  Mean Intensity [B, 1, H, W]  +  Variance [B, 1, H, W]
```

Key components:

- **MCSE** — Modality-Conditioned Shared Encoder adapts to event vs. frame reliability
- **UGSG** — Uncertainty-Guided Skip Gating routes features by predicted confidence
- **PUAR loss** — Probabilistic Uncertainty-Aware Reconstruction penalizes confident wrong predictions more strongly than uncertain ones

### Pipeline summary

```text
1. Encode asynchronous events as voxel grids
2. Combine event data with the previous intensity frame
3. Estimate probabilistic intensity-change priors (PEPE)
4. Reconstruct the next frame via PIEM
5. Train with an uncertainty-aware reconstruction loss (PUAR)
```

---

## Evaluation on Benchmarks

We recommend [EVREAL](https://github.com/ercanburak/EVREAL) for standardized evaluation.

```bash
git clone https://github.com/ercanburak/EVREAL.git && cd EVREAL
pip install event-pienet
cp /path/to/pie-net/config/method/PIENet.json config/method/
cp /path/to/pie-net/config/method/PIENetLite.json config/method/
cp /path/to/pie-net/pie_net/evreal_wrapper.py model/PIENet.py

# Evaluate both variants
python eval.py -m PIENet     -c std -d ECD MVSEC HQF -qm mse ssim lpips
python eval.py -m PIENetLite -c std -d ECD MVSEC HQF -qm mse ssim lpips
```

---

## Project Structure

```
pie-net/
├── pie_net/
│   ├── model.py           # Architecture + load_model()
│   ├── demo.py            # Real-time camera demo (CLI entry point)
│   ├── evreal_wrapper.py  # EVREAL integration
│   └── pretrained/
│       ├── model.pth      # PIE-Net (full)
│       └── model_lite.pth # PIE-Net-Lite
├── config/method/         # EVREAL method configs
├── examples/              # Usage examples
├── scripts/               # Legacy script aliases
├── pyproject.toml
└── README.md
```

---

## Citation

PIE-Net is the next generation of E2HQV. If you use PIE-Net in your research, please cite:

```bibtex
@inproceedings{qu2024e2hqv,
  title={E2HQV: High-Quality Video Generation from Event Camera via Theory-Inspired Model-Aided Deep Learning},
  author={Qu, Qiang and Shen, Yiran and Chen, Xiaoming and Chung, Yuk Ying and Liu, Tongliang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4632--4640},
  year={2024}
}

@article{qu2024evrepsl,
  title={EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision},
  author={Qu, Qiang and Chen, Xiaoming and Chung, Yuk Ying and Shen, Yiran},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```

Paper: [IEEE TIP](https://ieeexplore.ieee.org/document/10758409) · [arXiv](https://arxiv.org/abs/2412.07080)

---

## Acknowledgments

- [EvRepSL](https://github.com/VincentQQu/EvRepSL) — event representation toolkit (PIEM export)
- [EVREAL](https://github.com/ercanburak/EVREAL) — evaluation framework
- [dv-processing](https://gitlab.com/inivation/dv/dv-processing) — event camera I/O

---

## License

MIT License — see [LICENSE](LICENSE) for details.
