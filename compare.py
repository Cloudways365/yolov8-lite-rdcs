"""
一次性对比 Baseline（官方 YOLOv8n）与 YOLOv8n-Lite 的 Params/GFLOPs/mAP。
"""
import argparse
import time

import torch

import yolov8_lite_modules.register  # noqa: F401
from ultralytics import YOLO


def bench_one(name, weights_path, data, imgsz=640, iters=100, warmup=10, device="0"):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    model = YOLO(weights_path)

    # 参数量 / GFLOPs
    model.info(detailed=False)

    # 精度
    metrics = model.val(data=data, split="test", imgsz=imgsz, device=device, verbose=False)

    result = {
        "name": name,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_weights", type=str, required=True)
    parser.add_argument("--lite_weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="configs/rdcs.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    results = []
    results.append(bench_one("Baseline-YOLOv8n", args.baseline_weights, args.data, args.imgsz, device=args.device))
    results.append(bench_one("YOLOv8n-Lite", args.lite_weights, args.data, args.imgsz, device=args.device))

    print("\n\n" + "=" * 60)
    print(f"{'模型':<20}{'mAP50':<12}{'mAP50-95':<12}{'FPS':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20}{r['mAP50']:<12.4f}{r['mAP50-95']:<12.4f}{r['fps']:<10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
