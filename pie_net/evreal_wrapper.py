"""
EVREAL wrapper for PIE-Net and PIE-Net-Lite evaluation.

Usage:
    1. Copy this file to EVREAL/model/PIENet.py
    2. Copy config/method/PIENet.json and PIENetLite.json to EVREAL/config/method/
    3. Run:
         python eval.py -m PIENet -c std -d ECD MVSEC HQF -qm mse ssim lpips
         python eval.py -m PIENetLite -c std -d ECD MVSEC HQF -qm mse ssim lpips
"""

from pie_net import load_model, resolve_variant


class PIENet:
    """EVREAL-compatible wrapper for PIE-Net (full model)."""

    def __init__(self, device="cuda", **kwargs):
        self.model = load_model(pretrained=True, device=device, variant="pie-net")
        self.model.eval()

    def __call__(self, event_voxel):
        output = self.model(event_voxel)
        return output["image"]

    def reset_states(self):
        self.model.reset_states()


class PIENetLite:
    """EVREAL-compatible wrapper for PIE-Net-Lite."""

    def __init__(self, device="cuda", **kwargs):
        self.model = load_model(pretrained=True, device=device, variant="pie-net-lite")
        self.model.eval()

    def __call__(self, event_voxel):
        output = self.model(event_voxel)
        return output["image"]

    def reset_states(self):
        self.model.reset_states()


def get_model(device="cuda", variant="pie-net", **kwargs):
    resolved = resolve_variant(variant)
    if resolved == "pie-net-lite":
        return PIENetLite(device=device, **kwargs)
    return PIENet(device=device, **kwargs)
