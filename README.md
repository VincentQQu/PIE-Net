# PIE-Net

**Probabilistic Intensity-Event Modeling for High-Quality Event-Based Video Generation**

[![PyPI version](https://img.shields.io/pypi/v/pie-net.svg)](https://pypi.org/project/pie-net/)
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
pip install pie-net
```

### With optional dependencies

```bash
# Real-time event camera demo (DVS / DAVIS)
pip install pie-net[realtime]

# Benchmark evaluation helpers
pip install pie-net[eval]

# Everything
pip install pie-net[all]
```

### From source

```bash
git clone https://github.com/vincentquq/pie-net.git
cd pie-net
pip install -e .
```

### CUDA PyTorch

Install PyTorch with CUDA support first if you have a GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pie-net
```

---

## Quick Start

### Python API

```python
import torch
from pie_net import load_model, load_model_lite

# PIE-Net (full model — default)
model = load_model(pretrained=True, device="cuda")
model.eval()

# PIE-Net-Lite (faster, smaller)
lite = load_model_lite(pretrained=True, device="cuda")
# or: lite = load_model(variant="pie-net-lite", device="cuda")

events = torch.randn(1, 5, 180, 240).cuda()  # [B, bins, H, W]

with torch.no_grad():
    output = model(events)
    frame = output["image"]   # [1, 1, H, W] reconstructed intensity
    uncertainty = output["var"]  # [1, 1, H, W] per-pixel variance

model.reset_states()  # call between sequences
```

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

### Probabilistic Intensity-Event Mapping (PIEM)

PIE-Net formulates reconstruction as probabilistic inference:

1. **Event → log-intensity** — polarity-weighted events map to log-intensity changes
2. **Uncertainty propagation** — analytical pixel-wise variance from event noise
3. **Closed-form reconstruction** — prior intensity + predicted change → frame + confidence

### Architecture

```
Event Voxel Grid [B, 5, H, W]
        ↓
   Dual Stem (Event + Intensity)
        ↓
   Recurrent Encoder + MCSE (modality-conditioned FiLM)
        ↓
   Decoder + UGSG (uncertainty-guided skip gating)
        ↓
   PIEM Head
        ↓
Output: Mean Intensity [B, 1, H, W]  +  Variance [B, 1, H, W]
```

Key components:

- **MCSE** — Modality-Conditioned Shared Encoder adapts to event vs. frame reliability
- **UGSG** — Uncertainty-Guided Skip Gating routes features by predicted confidence
- **PUAR loss** — Probabilistic Uncertainty-Aware Reconstruction during training

---

## Evaluation on Benchmarks

We recommend [EVREAL](https://github.com/ercanburak/EVREAL) for standardized evaluation.

```bash
git clone https://github.com/ercanburak/EVREAL.git && cd EVREAL
pip install pie-net
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
```

---

## Acknowledgments

- [EVREAL](https://github.com/ercanburak/EVREAL) — evaluation framework
- [dv-processing](https://gitlab.com/inivation/dv/dv-processing) — event camera I/O

---

## License

MIT License — see [LICENSE](LICENSE) for details.
