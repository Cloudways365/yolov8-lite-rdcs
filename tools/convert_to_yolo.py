"""
将 RDCS BOP/LINEMOD标注格式转换为 YOLO txt 格式。
"""
import argparse
import random
from pathlib import Path

import yaml
from PIL import Image


CLASS_NAME_TO_ID = {
    "general_satellite": 0,
    "b330": 1,
    "dragon2": 2,
    "cygnus_enhanced": 3,
    "nauka": 4,
    "cst100_starliner": 5,
    "dream_chaser": 6,
    "boeing_airlock": 7,
}


def normalize_class_name(folder_name: str) -> str:
    return folder_name.strip().lower().replace(" ", "_").replace("-", "_")


def get_class_id(folder_name: str) -> int:
    key = normalize_class_name(folder_name)
    if key not in CLASS_NAME_TO_ID:
        new_id = len(CLASS_NAME_TO_ID)
        CLASS_NAME_TO_ID[key] = new_id
        print(f"[提示] 类别文件夹 '{folder_name}' 不在预设映射表里，自动分配 id={new_id}，"
              f"请检查是否需要改名/合并，并同步更新 configs/rdcs.yaml 的 names。")
    return CLASS_NAME_TO_ID[key]


def candidate_image_paths(class_dir: Path, frame_idx):
    idx = int(frame_idx)
    return [
        class_dir / f"{idx}.png",
        class_dir / f"{idx:04d}.png",
        class_dir / f"{idx:06d}.png",
        class_dir / "rgb" / f"{idx}.png",
        class_dir / "rgb" / f"{idx:04d}.png",
        class_dir / "rgb" / f"{idx:06d}.png",
        class_dir / f"{idx}.jpg",
        class_dir / f"{idx:04d}.jpg",
        class_dir / "rgb" / f"{idx}.jpg",
        class_dir / "rgb" / f"{idx:04d}.jpg",
    ]


def find_image(class_dir: Path, frame_idx):
    for p in candidate_image_paths(class_dir, frame_idx):
        if p.exists():
            return p
    return None


def collect_samples(raw_root: Path):
    """遍历每个类别文件夹的 gt_<Class>.yaml，返回 [(image_path, class_id, obj_bb_xywh_px), ...]"""
    samples = []
    n_missing_img = 0

    class_dirs = [d for d in sorted(raw_root.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"在 {raw_root} 下没有找到任何类别子文件夹，请检查 --raw_root 路径是否正确。")

    for class_dir in class_dirs:
        class_name = class_dir.name
        gt_path = class_dir / f"gt_{class_name}.yaml"
        if not gt_path.exists():
            cands = list(class_dir.glob("gt_*.yaml"))
            if not cands:
                print(f"[跳过] {class_dir} 下没有找到 gt_*.yaml 标注文件")
                continue
            gt_path = cands[0]

        cls_id = get_class_id(class_name)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = yaml.safe_load(f)

        for frame_idx, entries in gt.items():
            if not entries:
                continue
            img_path = find_image(class_dir, frame_idx)
            if img_path is None:
                n_missing_img += 1
                continue
            for entry in entries:  # 一帧可能不止一个目标，逐个处理
                x, y, w, h = entry["obj_bb"]  # 像素坐标：左上角x,y + 宽高
                samples.append((img_path, cls_id, (x, y, w, h)))

    if n_missing_img:
        print(f"[警告] 有 {n_missing_img} 帧未能匹配到图片文件，已跳过。"
              f"如果这个数字很大，说明图片命名规则和脚本假设的不一样，"
              f"请把某个类别文件夹 `ls` 的真实文件名发给我核对。")

    return samples


def write_yolo_sample(img_path: Path, cls_id: int, bb_xywh_px, out_img_dir: Path, out_label_dir: Path, uid: str):
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    x, y, w, h = bb_xywh_px
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h

    ext = img_path.suffix
    dst_img = out_img_dir / f"{uid}{ext}"
    if not dst_img.exists():
        try:
            dst_img.symlink_to(img_path.resolve())
        except OSError:
            import shutil
            shutil.copy(img_path, dst_img)

    (out_label_dir / f"{uid}.txt").write_text(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True,
                         help="RDCS 原始数据根目录，例如 /data_d/WSJ/Images/RDCS/RDCS")
    parser.add_argument("--out_root", type=str, default="datasets/RDCS", help="转换后 YOLO 格式数据根目录")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)  # 剩余为 test
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    samples = collect_samples(raw_root)
    print(f"共解析到 {len(samples)} 个有效标注样本。")
    if not samples:
        return

    random.seed(args.seed)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split, split_samples in splits.items():
        out_img_dir = out_root / "images" / split
        out_label_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_path, cls_id, bbox) in enumerate(split_samples):
            parent_tag = img_path.parent.parent.name if img_path.parent.name == "rgb" else img_path.parent.name
            uid = f"{parent_tag}_{img_path.stem}_{i}"
            write_yolo_sample(img_path, cls_id, bbox, out_img_dir, out_label_dir, uid)
        print(f"[{split}] 写入 {len(split_samples)} 张")

    print("\n最终类别映射（请核对并同步到 configs/rdcs.yaml 的 names）：")
    for name, idx in sorted(CLASS_NAME_TO_ID.items(), key=lambda kv: kv[1]):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
