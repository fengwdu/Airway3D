import argparse
from pathlib import Path

import numpy as np


def load_obj_vertices(path):
    points = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, x, y, z, *_ = line.split()
            points.append([float(x), float(y), float(z)])
    return np.asarray(points, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Measure 3D distance between mouth landmarks 51 and 57 in an OBJ point file."
    )
    parser.add_argument(
        "obj",
        nargs="?",
        default="output/closed/lmk_closed_gt.obj",
        help="OBJ file containing landmark vertices in 0-based landmark order.",
    )
    args = parser.parse_args()

    points = load_obj_vertices(args.obj)
    if points.shape[0] <= 57:
        raise RuntimeError(
            f"Need complete 68 landmark vertices to use original indices 51 and 57, "
            f"but got {points.shape[0]} vertices from {args.obj}. "
            "This OBJ likely skipped invalid landmarks and lost the original landmark indices."
        )

    dist_m = float(np.linalg.norm(points[51] - points[57]))
    print(f"file: {args.obj}")
    print("landmark_pair: 51-57")
    print(f"distance_m: {dist_m:.8f}")
    print(f"distance_cm: {dist_m * 100.0:.4f}")


if __name__ == "__main__":
    main()
