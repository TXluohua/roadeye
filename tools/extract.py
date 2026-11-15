# -*- coding: utf-8 -*-
"""
extract.py —— 解压 Unified_Road_Defect_Dataset 的三个 tar.gz，并修正 data.yaml

为什么需要单独一个脚本：
  上游仓库给的 rdd_merged.yaml 里 `path:` 写的是作者本机的绝对路径
  （/home/eyehearyou143/unified_rdd/data/yolo），直接拿去训练必然报错。
  本脚本解压后按实际位置重写一份 data.yaml。

用法
----
    python tools/extract.py                        # 解压到 data/raw/unified
    python tools/extract.py --src data/_dl --dst data/raw/unified
    python tools/extract.py --skip-extract         # 只重写 yaml（已经解压过了）

产物
----
    data/raw/unified/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
        data.yaml          ← 修好的，path 指向本目录
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

NAMES = [
    "Longitudinal Crack (D00)",
    "Transverse Crack (D10)",
    "Alligator Crack (D20)",
    "Pothole (D40)",
]


def extract_one(tar_path: Path, dst: Path) -> None:
    print(f"  解压 {tar_path.name} ({tar_path.stat().st_size / 1e6:.0f} MB) …", flush=True)
    t0 = time.time()
    with tarfile.open(tar_path, "r:gz") as tf:
        n = len(tf.getmembers())
        tf.extractall(dst)
    print(f"    完成 {n} 个成员，{time.time() - t0:.0f}s", flush=True)


def find_layout(root: Path) -> tuple[Path | None, Path | None, str]:
    """返回 (train_images_dir, val_images_dir, 布局说明)"""
    # 情况 1：images/train, images/val
    a = root / "images" / "train"
    b = root / "images" / "val"
    if a.is_dir():
        return a, (b if b.is_dir() else None), "images/{train,val}"
    # 情况 2：train/images, valid/images
    a = root / "train" / "images"
    if a.is_dir():
        for v in ("val", "valid"):
            if (root / v / "images").is_dir():
                return a, root / v / "images", f"train/images + {v}/images"
    # 情况 3：解压出了套娃目录
    subs = [d for d in root.iterdir() if d.is_dir()]
    for d in subs:
        r = find_layout(d)
        if r[0]:
            return r[0], r[1], f"{d.name}/" + r[2]
    return None, None, "未知"


def count(d: Path | None, ext: str) -> int:
    if d is None or not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob(f"*{ext}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="解压并修正 Unified RDD 数据集")
    ap.add_argument("--src", default="data/_dl", help="tar.gz 所在目录")
    ap.add_argument("--dst", default="data/raw/unified", help="解压目标目录")
    ap.add_argument("--skip-extract", action="store_true", help="跳过解压，只重写 yaml")
    args = ap.parse_args(argv)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        tars = sorted(src.glob("*.tar.gz"))
        if not tars:
            print(f"❌ {src} 下没有 tar.gz")
            print("   先跑下载：python _dl.py   或   huggingface-cli download ...")
            return 1
        print(f"找到 {len(tars)} 个压缩包：{[t.name for t in tars]}")
        for t in tars:
            extract_one(t, dst)

    print("\n检查目录布局 …")
    tr, va, how = find_layout(dst)
    if tr is None:
        print("❌ 没找到图片目录。请把 data/raw/unified 的目录结构贴出来")
        for p in sorted(dst.iterdir())[:20]:
            print("   ", p.name)
        return 1
    print(f"  布局：{how}")
    print(f"  train images: {tr}")

    n_tr_img = count(tr, ".jpg") + count(tr, ".png")

    def labels_of(img_dir: Path | None) -> Path | None:
        """images/{split} -> labels/{split}"""
        if img_dir is None:
            return None
        if img_dir.parent.name == "images":
            return img_dir.parent.parent / "labels" / img_dir.name
        return img_dir / "labels"

    n_tr_lab = count(labels_of(tr), ".txt")
    n_va_img = count(va, ".jpg") + count(va, ".png") if va else 0
    n_va_lab = count(labels_of(va), ".txt")

    print(f"  train: {n_tr_img:,} 张图 / {n_tr_lab:,} 个标签")
    print(f"  val  : {n_va_img:,} 张图 / {n_va_lab:,} 个标签")

    if n_tr_img != n_tr_lab:
        print("  ⚠ 图与标签数量不一致（YOLO 允许空 txt，但完全缺失会报错）—— 用 probe.py 细查")

    # 写 yaml（不写 path 绝对路径，交给 ultralytics 按 yaml 所在目录解析）
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        "# 由 tools/extract.py 自动生成\n"
        "# 故意不写 path：ultralytics 会以本 yaml 所在目录为数据集根目录\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    print(f"\n已写 {yaml_path}")

    print("\n下一步：")
    print(f"  python tools\\probe.py --root {args.dst} --dump-issues --out assets\\results\\probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
