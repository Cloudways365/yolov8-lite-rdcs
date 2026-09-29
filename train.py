"""
训练脚本
"""
import argparse

import torch

import yolov8_lite_modules.register  # noqa: F401  注册自定义模块，必须在 import YOLO 之前
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="configs/yolov8n-lite.yaml", help="模型结构 yaml")
    parser.add_argument("--data", type=str, default="configs/rdcs.yaml", help="数据集配置 yaml")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="预训练权重，用于迁移初始化")
    parser.add_argument("--epochs", type=int, default=55,
                         help="训练目标总轮数。--resume 续训时，这里填的是\"总共要跑到第几轮\"，不是\"再跑几轮\"")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30, help="早停耐心轮数")
    parser.add_argument("--project", type=str, default="runs/rdcs")
    parser.add_argument("--name", type=str, default="yolov8n-lite")
    parser.add_argument("--device", type=str, default=None,
                         help="GPU编号如 '0'，或 'cpu'。不传时自动判断：有可用GPU用'0'，没有则用'cpu'。")
    parser.add_argument("--resume", type=str, default=None,
                         help="断点续训：传入之前训练目录下的 weights/last.pt 路径（不要用best.pt，"
                              "last.pt才保存了完整的优化器状态）。传了这个参数后 --model/--pretrained/"
                              "--project/--name 都会被忽略，自动沿用原训练目录继续写入。")
    args = parser.parse_args()

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[提示] 未指定 --device，自动选择: {args.device}"
              + ("" if torch.cuda.is_available() else "（未检测到可用GPU，torch.cuda.is_available()为False，将用CPU训练，速度会慢很多）"))

    if args.resume:
        # 断点续训：直接从 last.pt 恢复，ultralytics 会自动读取同目录 args.yaml 里的原始超参数、
        # 优化器状态、已训练轮数，只需要用 epochs 指定"新的目标总轮数"即可继续训练到该轮数。
        model = YOLO(args.resume, task="detect")
        model.train(resume=True, epochs=args.epochs, device=args.device)
        return

    model = YOLO(args.model, task="detect")  # 显式指定 task，避免自定义yaml触发的task自动猜测失败
    if args.pretrained:
        try:
            model = model.load(args.pretrained)
        except Exception as e:
            print(f"[警告] 加载预训练权重失败（结构差异较大属正常现象，将从随机初始化开始训练）: {e}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        close_mosaic=10,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
