# -*- coding: utf-8 -*-
"""
train.py —— RoadEye 训练脚本

用法
----
    # 先确认数据就位
    python tools/extract.py --skip-extract
    python tools/probe.py --root data/raw/unified

    # 训练（默认读 configs/train.yaml）
    python train.py
    python train.py --epochs 10 --batch 8      # 临时覆盖
    python train.py --model yolo11s.pt         # 换更大模型

产物
----
    runs/<name>/weights/best.pt       ← 权重
    runs/<name>/results.csv           ← 每轮指标（后面 eval 脚本读它）
    runs/<name>/results.png, PR_curve.png 等图

注意
----
    · 数据集是 tar.gz 解压出来的，data.yaml 由 tools/extract.py 生成
    · 上游 rdd_merged.yaml 里的 path 是作者本机绝对路径，不要用那个
    · degrees=0：道路病害不能旋转增强，否则横裂会被转成纵裂
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="RoadEye 训练")
    ap.add_argument("--config", default=str(HERE / "configs" / "train.yaml"))
    ap.add_argument("--paths", default=str(HERE / "configs" / "paths.yaml"))
    # 以下都可临时覆盖配置文件
    ap.add_argument("--data", default=None, help="data.yaml 路径")
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader worker 数。★ 1024 分辨率下必须降到 2，"
                         "否则 Windows 共享内存/页面文件会不够（torch 报 "
                         "Couldn't open shared file mapping / error 1455）")
    ap.add_argument("--device", default=None, help="如 0 / cpu")
    ap.add_argument("--name", default=None, help="runs/<name>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印配置不训练")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = load_yaml(Path(args.config))
    paths = load_yaml(Path(args.paths)) if Path(args.paths).is_file() else {}

    # data.yaml 定位：命令行 > configs/paths.yaml > 默认
    data = args.data or paths.get("dataset_yaml") or "data/raw/unified/data.yaml"
    data_path = (HERE / data).resolve() if not Path(data).is_absolute() else Path(data)
    if not data_path.is_file():
        print(f"❌ 找不到 {data_path}")
        print("   先解压数据集：python tools\\extract.py")
        print("   或指定：python train.py --data <你的 data.yaml>")
        return 1

    # 校验 data.yaml 里的 train/val 能否解析到真实目录
    dcfg = load_yaml(data_path)
    base = data_path.parent
    ok = True
    for key in ("train", "val"):
        v = dcfg.get(key)
        if not v:
            print(f"❌ data.yaml 缺 {key} 字段")
            ok = False
            continue
        v = v[0] if isinstance(v, list) else v
        cand = [base / v, base / str(v).lstrip("./"), base / str(v).replace("../", "")]
        hit = next((c for c in cand if c.exists()), None)
        n = 0
        if hit:
            n = sum(1 for _ in hit.rglob("*") if _.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".bmp"})
        print(f"  {key:5s} -> {(hit or cand[0])}   图片 {n} 张")
        if not hit or n == 0:
            print(f"        ⚠ 这个 {key} 路径没有图片，检查 data.yaml")
            ok = False
    if not ok:
        print("\n⚠ data.yaml 有问题，仍可继续但很可能训练失败。建议先跑 probe.py 排查")

    # 合并配置
    run_cfg = {
        "data": str(data_path),
        "model": args.model or cfg.get("model", "yolo11n.pt"),
        "epochs": args.epochs if args.epochs is not None else cfg.get("epochs", 50),
        "batch": args.batch if args.batch is not None else cfg.get("batch", 16),
        "imgsz": args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640),
        "patience": cfg.get("patience", 15),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "workers": args.workers if args.workers is not None else cfg.get("workers", 8),
        "amp": cfg.get("amp", True),
        "optimizer": cfg.get("optimizer", "auto"),
        "lr0": cfg.get("lr0", 0.01),
        "lrf": cfg.get("lrf", 0.01),
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "warmup_epochs": cfg.get("warmup_epochs", 3.0),
        "box": cfg.get("box", 7.5),
        "cls": cfg.get("cls", 0.5),
        "dfl": cfg.get("dfl", 1.5),
        # 增强（道路病害专用）
        "degrees": cfg.get("degrees", 0.0),
        "translate": cfg.get("translate", 0.1),
        "scale": cfg.get("scale", 0.5),
        "shear": cfg.get("shear", 0.0),
        "perspective": cfg.get("perspective", 0.0),
        "flipud": cfg.get("flipud", 0.0),
        "fliplr": cfg.get("fliplr", 0.5),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.7),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.0),
        "close_mosaic": cfg.get("close_mosaic", 10),
        "val": True,
        "plots": True,
        "project": str(HERE / cfg.get("project", "runs")),
        "name": args.name or cfg.get("name", "baseline"),
        "exist_ok": False,
        "verbose": True,
    }
    if args.device is not None:
        run_cfg["device"] = args.device
    if args.resume:
        run_cfg["resume"] = True

    print("\n=== 训练配置 ===")
    for k, v in run_cfg.items():
        print(f"  {k:16s} = {v}")

    # 参数冲突自检
    if run_cfg["patience"] >= run_cfg["epochs"]:
        print(f"\n⚠ patience({run_cfg['patience']}) >= epochs({run_cfg['epochs']})，"
              f"早停永远不会触发")
    if run_cfg["degrees"] > 0:
        print("\n⚠ degrees > 0：旋转增强会把横向裂缝转成纵向，标签语义就错了")

    if args.dry_run:
        print("\n--dry-run：不训练")
        return 0

    # 延迟 import，让 --dry-run 不依赖 ultralytics
    from ultralytics import YOLO
    import ultralytics
    print(f"\nultralytics {ultralytics.__version__}  ({ultralytics.__file__})")

    model = YOLO(run_cfg.pop("model"))
    model.train(**run_cfg)

    best = HERE / cfg.get("project", "runs") / run_cfg["name"] / "weights" / "best.pt"
    print(f"\n训练完成。权重：{best}")
    print("下一步：")
    print(f"  python src\\offline\\eval_baseline.py --run {run_cfg['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
