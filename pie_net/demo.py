#!/usr/bin/env python3
"""Real-time event camera demo for PIE-Net and PIE-Net-Lite."""

from __future__ import annotations

import argparse
import time

import cv2 as cv
import numpy as np
import torch
from datetime import timedelta

from pie_net import load_model, resolve_variant, count_parameters

WINDOW_TITLE = "PIE-Net — Event Camera Reconstruction"


def _require_dv():
    try:
        import dv_processing as dv
        return dv
    except ImportError as exc:
        raise SystemExit(
            "dv-processing is required for the real-time demo.\n"
            "Install with: pip install event-pienet[realtime]"
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print event throughput and frame stats periodically",
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
        self.n_event_batches = 0
        self.n_events_total = 0
        self.last_event_count = 0
        self.latest_display = None
        self.debug = args.debug

        print(f"Device: {self.device}")
        print(f"Variant: {self.variant}")

        print("Initializing event camera...")
        self.capture = dv.io.CameraCapture()
        if not self.capture.isConnected():
            raise SystemExit(
                "No event camera detected. On WSL, attach USB with:\n"
                "  usbipd bind --busid <BUSID>\n"
                "  usbipd attach --wsl --busid <BUSID>"
            )
        if not self.capture.isEventStreamAvailable():
            raise SystemExit("Camera connected but event stream is unavailable.")

        self.resolution = self.capture.getEventResolution()
        self.height = self.resolution[1]
        self.width = self.resolution[0]
        print(f"Camera: {self.capture.getCameraName()}")
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
        if events.isEmpty():
            return self.voxel_grid

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
    def soft_normalize_for_display(arr, desired_mean=0.3):
        """Match the reference real-time app visualization."""
        arr = arr.float()
        lo = torch.quantile(arr, 0.0)
        hi = torch.quantile(arr, 1.0)
        scaled = (arr - lo) / (hi - lo + 1e-8)
        soft = (torch.tanh(scaled - 0.5) + 0.5).clamp(0, 1)
        scale = desired_mean / (soft.mean() + 1e-8)
        adjusted = (torch.tanh(soft * scale - 0.5) + 0.5).clamp(0, 1)
        return adjusted

    @staticmethod
    def tensor_to_display_image(tensor):
        img = (tensor.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(img)

    def make_status_frame(self, message):
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        cv.putText(
            frame,
            message,
            (12, self.height // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            255,
            1,
            cv.LINE_AA,
        )
        if self.visualize_voxel:
            frame = np.hstack((frame, frame))
        return frame

    def process_callback(self, events):
        if events.isEmpty():
            return

        self.last_event_count = events.size()
        voxel = self.events_to_voxel_grid(events)

        with torch.inference_mode():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(voxel.unsqueeze(0))
            else:
                output = self.model(voxel.unsqueeze(0))
            prediction = output["image"]

        pred_img = self.tensor_to_display_image(prediction)

        if self.visualize_voxel:
            voxel_img = self.tensor_to_display_image(
                self.soft_normalize_for_display(voxel.sum(0))
            )
            display = np.hstack((voxel_img, pred_img))
        else:
            display = pred_img

        self.latest_display = display
        self.n_frames += 1

    def show_display(self):
        if self.latest_display is None:
            display = self.make_status_frame("Waiting for events...")
        else:
            display = self.latest_display
        cv.imshow(WINDOW_TITLE, display)

    def run(self):
        print(f"Starting at ~{1000 / self.frame_interval:.0f} FPS — press 'q' to quit")
        print("Move the camera or scene to generate events.")
        self.slicer.doEveryTimeInterval(
            timedelta(milliseconds=self.frame_interval),
            self.process_callback,
        )

        cv.namedWindow(WINDOW_TITLE, cv.WINDOW_NORMAL)
        cv.startWindowThread()
        self.show_display()

        last_stats = time.monotonic()
        try:
            while self.capture.isRunning():
                events = self.capture.getNextEventBatch()
                if events is not None:
                    self.n_event_batches += 1
                    self.n_events_total += events.size()
                    self.slicer.accept(events)

                self.show_display()

                if self.debug and time.monotonic() - last_stats >= 2.0:
                    print(
                        f"[debug] batches={self.n_event_batches} "
                        f"events={self.n_events_total} "
                        f"frames={self.n_frames} "
                        f"last_slice={self.last_event_count}"
                    )
                    last_stats = time.monotonic()

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv.destroyAllWindows()
            if self.n_frames == 0:
                print(
                    "No frames rendered. Check USB passthrough and move the scene "
                    "to generate events. Run with --debug for throughput stats."
                )
            print(f"Processed {self.n_frames} frames")


def main():
    args = parse_args()
    RealTimeReconstructor(args).run()


if __name__ == "__main__":
    main()
