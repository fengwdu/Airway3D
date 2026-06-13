import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.camera import backproject_window, merge_depth_frames
from core.capture import OrbbecCapture
from core.face_test import detect_face_bbox
from third_party.tufa.Test import get_lmk_single


def draw_points(rgb, points_2d, valid_mask, point_ids):
    vis = rgb.copy()
    for idx, (u_v, ok) in zip(point_ids, zip(points_2d, valid_mask)):
        u, v = u_v
        color = (0, 255, 0) if ok else (255, 0, 0)  # RGB: green / red
        center = (int(round(float(u))), int(round(float(v))))
        cv2.circle(vis, center, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            str(idx),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def capture_three_frames(cap):
    frames = []
    while len(frames) < 3:
        out = cap.read()
        if out is None:
            continue
        camera, rgb, depth = out
        frames.append((rgb.copy(), depth.copy(), camera))
    return frames


def analyze_capture(frames, window, trim):
    rgb = frames[0][0]
    camera = frames[0][2]
    depth = merge_depth_frames([frame[1] for frame in frames])

    bbox_xywh = detect_face_bbox(rgb)
    if bbox_xywh is None:
        raise RuntimeError("No face bbox detected.")

    landmarks_68 = np.asarray(get_lmk_single(rgb, bbox_xywh), dtype=np.float32)
    point_ids = np.r_[8, np.arange(17, 68)]
    landmarks_eval = landmarks_68[point_ids]

    _, valid_mask = backproject_window(
        landmarks_eval,
        depth,
        camera,
        window=window,
        trim=trim,
    )

    valid_count = int(valid_mask.sum())
    total = int(valid_mask.size)

    vis = draw_points(rgb, landmarks_eval, valid_mask, point_ids)

    x, y, w, h = [int(v) for v in bbox_xywh]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"valid {valid_count}/{total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if valid_count == total else (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return vis, valid_count, total


def main():
    parser = argparse.ArgumentParser(
        description="Capture 3 depth frames, merge them, and visualize landmark backprojection validity."
    )
    parser.add_argument("--out", default="tests/backprojection_success.jpg")
    parser.add_argument("--window", type=int, default=4, help="Depth window size for each landmark.")
    parser.add_argument("--trim", type=int, default=1, help="Trim count on each side after sorting depth values.")
    args = parser.parse_args()

    cap = OrbbecCapture(timeout_ms=100)
    cap.start()
    try:
        print("Preview started. Press SPACE or C to capture, Q or ESC to quit.")
        while True:
            out = cap.read()
            if out is None:
                continue
            _, rgb, _ = out
            cv2.imshow("Backprojection Preview", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return
            if key in (32, ord("c"), ord("C")):
                frames = capture_three_frames(cap)
                try:
                    vis, valid_count, total = analyze_capture(frames, args.window, args.trim)
                except Exception as exc:
                    print(f"Capture failed: {exc}")
                    continue

                out_path = ROOT / args.out
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                print(f"Backprojection success: {valid_count}/{total}")
                print(f"Saved visualization: {out_path}")
                cv2.imshow("Backprojection Result", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
