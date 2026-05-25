#!/usr/bin/env python3
"""Real-time event camera demo for PIE-Net and PIE-Net-Lite."""

from __future__ import annotations

import argparse

import cv2 as cv
import numpy as np
import torch
from datetime import timedelta

from pie_net import load_model, resolve_variant, count_parameters


def _require_dv():
    try:
        import dv_processing as dv
        return dv
    except ImportError as exc:
        raise SystemExit(
            "dv-processing is required for the real-time demo.\n"
            "Install with: pip install pie-net[realtime]"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time event camera reconstruction with PIE-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="pie-net",
        help='Model variant: "pie-net" (full) or "pie-net-lite" (lite)',
    )
    parser.add_argument(
        "--no-visualize-voxel",
        dest="visualize_voxel",
        action="store_false",
        help="Show reconstruction only (hide voxel grid)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=33,
        help="Frame interval in milliseconds (~30 FPS at 33 ms)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA",
    )
    parser.set_defaults(visualize_voxel=True)
    return parser.parse_args()


class RealTimeReconstructor:
    def __init__(self, args):
        dv = _require_dv()
        self._dv = dv

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = resolve_variant(args.variant)
        self.visualize_voxel = args.visualize_voxel
        self.frame_interval = args.frame_interval
        self.use_amp = args.use_amp and self.device.type == "cuda"
        self.temporal_bins = 5
        self.n_frames = 0

        print(f"Device: {self.device}")
        print(f"Variant: {self.variant}")

        print("Initializing event camera...")
        self.capture = dv.io.CameraCapture()
        self.resolution = self.capture.getEventResolution()
        print(f"Camera resolution: {self.resolution[0]}x{self.resolution[1]}")

        self.voxel_grid = torch.zeros(
            (self.temporal_bins, self.resolution[1], self.resolution[0]),
            device=self.device,
            dtype=torch.float32,
        )
        self.slicer = dv.EventStreamSlicer()

        print("Loading model...")
        self.model = load_model(pretrained=True, device=str(self.device), variant=self.variant)
        self.model.eval()
        print(f"Loaded {self.variant} ({count_parameters(self.model):,} params)")

    def events_to_voxel_grid(self, events):
        self.voxel_grid.zero_()

        timestamps = torch.tensor(events.timestamps(), device=self.device, dtype=torch.float32)
        polarities = torch.tensor(
            (events.polarities().astype(int) * 2 - 1),
            device=self.device,
            dtype=torch.float32,
        )
        coords = torch.tensor(events.coordinates(), device=self.device, dtype=torch.int32)
        x, y = coords[:, 0], coords[:, 1]

        t0, t1 = timestamps[0], timestamps[-1]
        dt = max(t1 - t0, 1.0)
        t_norm = (self.temporal_bins - 1) * (timestamps - t0) / dt

        ti = t_norm.to(torch.int32)
        tf = t_norm - ti

        valid_l = ti < self.temporal_bins
        valid_r = (ti + 1) < self.temporal_bins

        self.voxel_grid.index_put_(
            (ti[valid_l], y[valid_l], x[valid_l]),
            polarities[valid_l] * (1.0 - tf[valid_l]),
            accumulate=True,
        )
        self.voxel_grid.index_put_(
            (ti[valid_r] + 1, y[valid_r], x[valid_r]),
            polarities[valid_r] * tf[valid_r],
            accumulate=True,
        )
        return self.voxel_grid

    @staticmethod
    def normalize_for_display(arr):
        arr = arr.float()
        lo = torch.quantile(arr, 0.01)
        hi = torch.quantile(arr, 0.99)
        arr = (arr - lo) / (hi - lo + 1e-8)
        return (torch.tanh(arr - 0.5) + 0.5).clamp(0, 1)

    def process_callback(self, events):
        voxel = self.events_to_voxel_grid(events)

        with torch.inference_mode():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(voxel.unsqueeze(0))
            else:
                output = self.model(voxel.unsqueeze(0))
            prediction = output["image"]

        pred_img = (prediction.squeeze() * 255).to(torch.uint8).cpu().numpy()

        if self.visualize_voxel:
            voxel_img = (self.normalize_for_display(voxel.sum(0)) * 255).to(torch.uint8).cpu().numpy()
            display = np.hstack((voxel_img, pred_img))
        else:
            display = pred_img

        self.n_frames += 1
        title = f"PIE-Net ({self.variant}) — Event Camera Reconstruction"
        cv.imshow(title, display)
        cv.waitKey(1)

    def run(self):
        print(f"Starting at ~{1000 / self.frame_interval:.0f} FPS — press 'q' to quit")
        self.slicer.doEveryTimeInterval(
            timedelta(milliseconds=self.frame_interval),
            self.process_callback,
        )

        try:
            while self.capture.isRunning():
                events = self.capture.getNextEventBatch()
                if events is not None:
                    self.slicer.accept(events)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv.destroyAllWindows()
            print(f"Processed {self.n_frames} frames")


def main():
    args = parse_args()
    RealTimeReconstructor(args).run()


if __name__ == "__main__":
    main()
