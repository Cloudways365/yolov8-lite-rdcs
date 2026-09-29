"""
在测试集上验证精度 + 简单测速；同时会自动生成混淆矩阵/PR曲线/F1曲线/预测可视化图
（这些图默认只在训练"正常跑完"时才会生成，如果训练是 Ctrl+C 中途打断的，用这个脚本
对 best.pt 补跑一次验证就能把它们补出来）。用法：

  python val_and_bench.py --weights runs/rdcs/yolov8n-lite/weights/best.pt --data configs/rdcs.yaml
"""
import argparse
import time

import torch

import yolov8_lite_modules.register  # noqa: F401
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="训练好的权重路径 best.pt")
    parser.add_argument("--data", type=str, default="configs/rdcs.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100, help="测速迭代次数")
    parser.add_argument("--device", type=str, default=None, help="GPU编号如 '0'，或 'cpu'。不传时自动判断。")
    parser.add_argument("--project", type=str, default=None,
                         help="验证结果(混淆矩阵/PR曲线等图)保存目录，默认存到权重所在训练目录下的 val/ 子目录")
    parser.add_argument("--name", type=str, default="val")
    args = parser.parse_args()

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[提示] 未指定 --device，自动选择: {args.device}")

    if args.project is None:
        # 权重路径形如 runs/rdcs/yolov8n-lite/weights/best.pt，取到 runs/rdcs/yolov8n-lite 作为项目目录
        from pathlib import Path
        args.project = str(Path(args.weights).resolve().parents[1])

    model = YOLO(args.weights, task="detect")

    # 1) 精度验证（plots=True 会自动生成 confusion_matrix.png / PR_curve.png / F1_curve.png /
    #    P_curve.png / R_curve.png / val_batch*_labels.jpg / val_batch*_pred.jpg 等）
    print("\n===== 精度验证 =====")
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, device=args.device,
        plots=True, project=args.project, name=args.name,
    )
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"[提示] 混淆矩阵/PR曲线/F1曲线/预测可视化图已保存到: {args.project}/{args.name}/")

    # 2) 参数量 / GFLOPs
    print("\n===== 模型规模 =====")
    model.info(detailed=False)

    # 3) 简单测速（GPU，若无 GPU 会自动退化为 CPU）
    print("\n===== 推理测速 =====")
    device = model.device
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    net = model.model.eval()

    with torch.no_grad():
        for _ in range(args.warmup):
            net(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            net(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    fps = args.iters / elapsed
    latency_ms = elapsed / args.iters * 1000
    print(f"设备: {device}")
    print(f"平均延迟: {latency_ms:.2f} ms/张")
    print(f"FPS:      {fps:.1f}")


if __name__ == "__main__":
    main()
