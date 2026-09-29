# YOLOv8n-Lite for RDCS —— 代码包使用说明

## 0. 数据集背景
RDCS 数据集来自 `github.com/Snowbluerose/RDCS-dataset`，它是航天器 RGB-D 位姿估计数据集，Blender 仿真渲染，包含 RGB 图像、像素对齐深度图、Mask（含带噪声版本）、位姿标签、Bounding Box 标签、`.ply` 三维模型，涵盖 8 种在役/刚退役航天器，分别为General satellite、B330、Dragon 2、Enhanced Cygnus、Nauka、CST-100 Starliner、Dream Chaser、Boeing Airlock。

本代码包只使用其中的RGB 图像 + Bounding Box 标签，做8 类航天器检测任务。

下载地址参见数据集仓库首页，`.7z` 压缩格式，下载后自行解压。

## 目录结构
```
yolov8-lite-rdcs/
├── configs/
│   ├── rdcs.yaml           # 数据集配置
│   ├── yolov8n-lite.yaml   # 改进后的模型结构
│   ├── yolov8n-lite-exp1-backbone-only.yaml  # 消融实验1：只改 backbone
│   └── yolov8n-lite-exp2-neck-only           # 消融实验2：只改 neck
├── tools/
│   └── convert_to_yolo.py  # RDCS原始标注 -> YOLO txt 格式转换脚本模板
├── yolov8_lite_modules/
│   ├── __init__.py
│   ├── lite.py             # 自定义模块实现
│   └── register.py         # 运行时把自定义模块注册进 ultralytics，无需改源码
├── compare.py
├── train.py
├── val_and_bench.py
├── yolov8n.pt
└── README.md
```

## 1. 环境安装
```bash
pip install ultralytics==8.2.0 pillow
```


## 2. 准备数据
```yaml
0:
- cam_R_m2c: [...]      # 位姿旋转矩阵
  cam_t_m2c: [...]      # 位姿平移向量
  obj_bb: [x, y, w, h]  # 像素坐标 bbox：左上角x,y + 宽高
  obj_id: BA            # 内部编码
```
`tools/convert_to_yolo.py` 已按此格式实现，会自动按 7:2:1 划分 train/val/test 并生成 YOLO txt 标注
```bash
pip install pyyaml pillow
python tools/convert_to_yolo.py --raw_root /data_d/RDCS --out_root datasets/RDCS
```

## 3. 训练命令
```bash
python train.py \
    --model configs/yolov8n-lite.yaml \
    --data configs/rdcs.yaml \
    --pretrained yolov8n.pt \
    --epochs 55 \
    --imgsz 640 \
    --batch 16 \
    --name yolov8n-lite
```

```bash
python train.py \
    --model yolov8n.yaml \
    --data configs/rdcs.yaml \
    --pretrained yolov8n.pt \
    --epochs 55 \
    --imgsz 640 \
    --batch 16 \
    --name yolov8n-baseline
```


## 4. 单模型验证
```bash
# 验证 Lite 模型
python val_and_bench.py \
    --weights runs/rdcs/yolov8n-lite/weights/best.pt \
    --data configs/rdcs.yaml \
    --split test

# 验证 Baseline
python val_and_bench.py \
    --weights runs/rdcs/yolov8n-baseline/weights/best.pt \
    --data configs/rdcs.yaml \
    --split test
```

## 5. Baseline vs Lite对比
```bash
python compare.py \
    --baseline_weights runs/rdcs/yolov8n-baseline/weights/best.pt \
    --lite_weights     runs/rdcs/yolov8n-lite/weights/best.pt \
    --data configs/rdcs.yaml
```
输出会打印两份模型各自的 Params/GFLOPs（来自 `model.info()`），以及 mAP50、mAP50-95的对比表格。


## 6. 消融实验
复制 `configs/yolov8n-lite.yaml` 改名，分别只保留一处改动，用 `train.py --model 你的yaml --name expX` 各训练一份即可。
