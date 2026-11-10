"""SF high-tech soldier - 3-view turnaround on a transparent background.

Worked example for forge's 3D mode: a full character built from ~130 SDF
primitives, rendered front / back / side.

    python soldier_turnaround.py --scale 0.25      # quick preview (960x540)
    python soldier_turnaround.py                   # 3840x2160 final
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from forge import post, sdf                                              # noqa: E402
from forge.render import turnaround                                      # noqa: E402
from forge.sdf import (align_y, box, capsule, cylinder, ellipsoid, intersect, lerp,  # noqa: E402
                       plane, rx, ry, rz, sphere, subtract, torus, union)


from forge.kit import soldier as build                                   # noqa: E402


VIEWS = (("front", 0), ("back", 180), ("side", -90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = 3840x2160, 0.25 = quick preview")
    ap.add_argument("--out", default="soldier_turnaround.png")
    ap.add_argument("--aa", type=int, default=4)
    args = ap.parse_args()
    W, H = int(3840 * args.scale) // 6 * 6, int(2160 * args.scale) // 2 * 2
    frames = turnaround(build(), VIEWS, width=W, height=H, target=(0, 0.975, 0), view_height=2.05,
                        aa=args.aa, verbose=True)
    img = post.sheet([post.develop(f) for f in frames])
    img.save(args.out)
    print("saved", args.out, img.size, f"{sum(f.seconds for f in frames):.0f}s")


if __name__ == "__main__":
    main()
