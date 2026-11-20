#!/usr/bin/env python
"""Encode a numbered PNG sequence to H.264 MP4 with PyAV (the frame env's own
ffmpeg libraries; there is no ffmpeg binary on this machine).

    python scripts/frames_to_mp4.py <dir> [--pattern 'orbit_*.png'] [--fps 24] [--out orbit.mp4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def encode(frames: list[Path], out: Path, fps: int, crf: int = 23) -> Path:
    import av
    import numpy as np
    from PIL import Image

    first = Image.open(frames[0]).convert("RGB")
    w, h = first.size
    w -= w % 2
    h -= h % 2
    with av.open(str(out), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium", "movflags": "+faststart"}
        for p in frames:
            im = Image.open(p).convert("RGB")
            if im.size != (w, h):
                im = im.crop((0, 0, w, h))
            frame = av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pattern", default="orbit_*.png")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="orbit.mp4")
    args = ap.parse_args()
    frames = sorted(args.dir.glob(args.pattern))
    if not frames:
        sys.exit(f"no frames matching {args.pattern} in {args.dir}")
    out = encode(frames, args.dir / args.out, args.fps)
    print(f"{len(frames)} frame(s) at {args.fps} fps -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
