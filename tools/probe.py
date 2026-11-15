# -*- coding: utf-8 -*-
"""
probe.py —— 数据集探查工具（★ 写任何训练代码之前，先跑这个）

解决的问题：拿到一个数据集，还不知道里面长什么样就开始写训练脚本 ——
这是最常见的翻车方式（类别索引错、路径错、坐标格式错、大量空标注）。

本脚本只读文件、不训练、不需要 GPU、不依赖 ultralytics。
支持三种标注格式自动识别：YOLO txt / PASCAL VOC xml / COCO json。

用法
----
    # 最常用：指向数据集根目录，自动找 data.yaml
    python tools/probe.py --root data/raw/my_dataset

    # 或者直接指 data.yaml
    python tools/probe.py --data data/raw/my_dataset/data.yaml

    # 只看某一类问题，并导出有问题的样张（画好框，肉眼核对用）
    python tools/probe.py --root data/raw/my_dataset --dump-issues

    # 顺带做重复图检测（按采样数，避免太慢）
    python tools/probe.py --root data/raw/my_dataset --dup-sample 3000

    # 存成 JSON，供后面文档/README 引用
    python tools/probe.py --root data/raw/my_dataset --out assets/results/probe.json

输出什么
--------
    ① 来源与格式        ② 拆分统计        ③ 类别分布（含 ASCII 条形图）
    ④ 图片尺寸分布      ⑤ 框尺寸/宽高比    ⑥ 标注质量检查
    ⑦ 重复图检测        ⑧ 自动诊断结论（能不能开始训练）

只依赖：opencv-python, numpy, pillow, pyyaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BAR_CHARS = "▏▎▍▌▋▊▉█"


def bar(value: float, vmax: float, width: int = 28) -> str:
    """纯文本条形图。"""
    if vmax <= 0:
        return ""
    n = value / vmax * width
    full = int(n)
    frac = n - full
    s = "█" * full
    if frac > 0.05 and full < width:
        s += BAR_CHARS[min(len(BAR_CHARS) - 1, int(frac * len(BAR_CHARS)))]
    return s


def human(n: int) -> str:
    return f"{n:,}"


def pct(a: float, b: float) -> str:
    return f"{(a / b * 100):.1f}%" if b else "—"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 1. 找 data.yaml 并解析
# --------------------------------------------------------------------------- #
def find_data_yaml(root: Path) -> Path | None:
    """在 root 下递归找 data.yaml / dataset.yaml。"""
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = root / name
        if p.is_file():
            return p
    cands = sorted(list(root.rglob("data.yaml")) + list(root.rglob("data.yml")))
    return cands[0] if cands else None


def load_yaml_cfg(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_splits(cfg: dict, yaml_path: Path) -> dict[str, Path]:
    """把 data.yaml 里的 train/val/test 解析成绝对路径。

    要处理三种写法：
      · 相对 yaml 所在目录（ultralytics 默认）
      · Roboflow 导出的 '../train/images'（多一个 ..）
      · 绝对路径
    """
    base = yaml_path.parent
    out: dict[str, Path] = {}
    for key in ("train", "val", "valid", "test"):
        v = cfg.get(key)
        if not v:
            continue
        v = v[0] if isinstance(v, list) else v
        cand = [
            Path(v) if Path(v).is_absolute() else (base / v),
            base / str(v).lstrip("./"),           # 去掉开头的 ./
            base / str(v).replace("../", ""),     # Roboflow 的 .. 兼容
        ]
        for c in cand:
            c = c.resolve()
            if c.exists():
                out["val" if key == "valid" else key] = c
                break
        else:
            out["val" if key == "valid" else key] = cand[0].resolve()  # 记为不存在
    return out


# --------------------------------------------------------------------------- #
# 2. 列图片、判格式
# --------------------------------------------------------------------------- #
def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS])


def detect_format(split_dir: Path) -> str:
    """判断这个 split 用的哪种标注格式。"""
    if not split_dir.is_dir():
        return "unknown"
    # YOLO：图片旁边有 labels/ 或同名 .txt
    if any(split_dir.rglob("labels")) or any(split_dir.rglob("*.txt")):
        txts = list(split_dir.rglob("*.txt"))
        if txts:
            # 排除 classes.txt / data 相关
            real = [t for t in txts if t.stem.lower() not in ("classes", "notes")]
            if real:
                return "yolo"
    if list(split_dir.rglob("*.xml")):
        return "voc"
    if list(split_dir.rglob("*.json")):
        return "coco"
    # 标注可能挂在兄弟目录：dataset/{images,labels}/
    if split_dir.name == "images" and (split_dir.parent / "labels").is_dir():
        return "yolo"
    return "unknown"


def find_images_dir(split_path: Path) -> Path:
    """把一个 split 路径解析成"图片所在目录"。"""
    if not split_path.exists():
        return split_path
    if any(p.suffix.lower() in IMG_EXTS for p in split_path.rglob("*")):
        # 路径本身就是图片目录（或包含图片）
        if split_path.name == "images" or list(split_path.glob("*.jpg")):
            return split_path
        sub = split_path / "images"
        return sub if sub.is_dir() else split_path
    return split_path


def labels_dir_for(images_dir: Path) -> Path:
    """YOLO：图片目录对应的 labels 目录。

    三种布局都要覆盖：
      root/images/train/*.jpg      -> root/labels/train/*.txt   （本项目）
      root/images/*.jpg            -> root/labels/*.txt
      root/train/images/*.jpg      -> root/train/labels/*.txt
    """
    if images_dir.parent.name == "images":          # .../images/{split}
        return images_dir.parent.parent / "labels" / images_dir.name
    if images_dir.name == "images":                 # .../images
        return images_dir.parent / "labels"
    if (images_dir.parent / "labels").is_dir():      # .../{split}/images
        return images_dir.parent / "labels"
    return images_dir / "labels"


# --------------------------------------------------------------------------- #
# 3. 读标注
# --------------------------------------------------------------------------- #
def read_yolo_labels(lab_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """返回 {图片名(不含扩展): [(cls, cx, cy, w, h), ...]}，坐标为归一化值。"""
    out: dict[str, list] = {}
    if not lab_dir.is_dir():
        return out
    for t in lab_dir.rglob("*.txt"):
        if t.stem.lower() in ("classes", "notes", "readme"):
            continue
        rows = []
        for ln, line in enumerate(t.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                rows.append((-1, ln, 0, 0, 0))       # 标记为格式错
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:])
                rows.append((c, cx, cy, w, h))
            except ValueError:
                rows.append((-1, ln, 0, 0, 0))
        out[t.stem] = rows
    return out


def read_voc_labels(split_dir: Path) -> dict[str, list]:
    """VOC: 返回 {图片名: [(cls_name, xmin, ymin, xmax, ymax) 像素]}"""
    out: dict[str, list] = {}
    for x in split_dir.rglob("*.xml"):
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        fn = root.findtext("filename") or x.stem
        sz = root.find("size")
        W = int(float(sz.findtext("width"))) if sz is not None and sz.findtext("width") else 0
        H = int(float(sz.findtext("height"))) if sz is not None and sz.findtext("height") else 0
        rows = []
        for obj in root.findall("object"):
            name = obj.findtext("name") or "?"
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            rows.append((name, x1, y1, x2, y2, W, H))
        out[Path(fn).stem] = rows
    return out


def read_coco_labels(split_dir: Path) -> tuple[dict[str, list], list[str]]:
    """COCO: 返回 ({图片名: [(cls, x1,y1,x2,y2)]}, 类别名列表)"""
    out: dict[str, list] = {}
    names: list[str] = []
    js = list(split_dir.rglob("*.json"))
    if not js:
        return out, names
    j = max(js, key=lambda p: p.stat().st_size)
    try:
        data = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return out, names
    if "categories" in data:
        cats = sorted(data["categories"], key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        cid2idx = {c["id"]: i for i, c in enumerate(cats)}
    else:
        cid2idx = {}
    id2file = {im["id"]: Path(im.get("file_name", "")).stem for im in data.get("images", [])}
    for a in data.get("annotations", []):
        stem = id2file.get(a.get("image_id"))
        if stem is None:
            continue
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        out.setdefault(stem, []).append(
            (cid2idx.get(a.get("category_id"), -1), x, y, x + w, y + h))
    return out, names


# --------------------------------------------------------------------------- #
# 4. 图像统计
# --------------------------------------------------------------------------- #
def image_sizes(paths: list[Path]) -> Counter:
    c = Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                c[im.size] += 1
        except Exception:
            c[("BROKEN", 0)] += 1
    return c


def dhash(path: Path, size: int = 8) -> np.uint64 | None:
    """差分哈希（比平均哈希稳，比 pHash 快，无需 imagehash 库）。"""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    diff = a[:, 1:] > a[:, :-1]
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# 5. 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="数据集探查（训练前必跑）")
    ap.add_argument("--root", default=None, help="数据集根目录（会自动找 data.yaml）")
    ap.add_argument("--data", default=None, help="直接指定 data.yaml")
    ap.add_argument("--format", default="auto",
                    choices=["auto", "yolo", "voc", "coco"])
    ap.add_argument("--dump-issues", action="store_true",
                    help="把有问题的样张画框存到 assets/probe_issues/")
    ap.add_argument("--dup-sample", type=int, default=0,
                    help=">0 时抽这么多样本做重复图检测")
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="差分哈希汉明距离 <= 此值判为重复（默认 6）")
    ap.add_argument("--out", default=None, help="结果另存为 JSON")
    ap.add_argument("--size-scan-max", type=int, default=8000,
                    help="最多扫描多少张图取尺寸（0=全部）")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---------- 定位 ----------
    if args.data:
        yaml_path = Path(args.data).expanduser().resolve()
        root = yaml_path.parent
    elif args.root:
        root = Path(args.root).expanduser().resolve()
        yaml_path = find_data_yaml(root)
    else:
        print("请给 --root 或 --data")
        return 1

    if not root.exists():
        print(f"路径不存在：{root}")
        return 1

    section("① 来源与格式")
    print(f"根目录   : {root}")
    cfg: dict = {}
    if yaml_path and yaml_path.is_file():
        cfg = load_yaml_cfg(yaml_path)
        print(f"data.yaml: {yaml_path.relative_to(root) if root in yaml_path.parents else yaml_path}")
        rf = cfg.get("roboflow")
        if isinstance(rf, dict):
            print(f"Roboflow : workspace={rf.get('workspace')}  project={rf.get('project')}  "
                  f"version={rf.get('version')}  license={rf.get('license')}")
            print(f"            {rf.get('url', '')}")
    else:
        print("⚠ 没找到 data.yaml —— 将按目录结构自动猜类别")

    # 类别
    names = cfg.get("names")
    if isinstance(names, dict):                       # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    if not names:
        # 从 classes.txt 猜
        ct = list(root.rglob("classes.txt"))
        if ct:
            names = [l.strip() for l in ct[0].read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"类别({len(names) if names else '?'}) : {names}")

    # ---------- 拆分 ----------
    splits = resolve_splits(cfg, yaml_path) if cfg else {}
    if not splits:
        # 猜目录
        for cand in ("train", "valid", "val", "test"):
            d = root / cand
            if d.is_dir():
                splits["val" if cand == "valid" else cand] = d
    if not splits:
        print("\n⚠ 没有解析出任何 split —— 请检查 data.yaml 的 train/val 字段，或手动传路径")
        return 1

    section("② 拆分统计")
    print(f"{'split':<8}{'路径存在':<10}{'格式':<10}{'图片':>8}{'标注文件':>10}"
          f"{'实例':>8}{'空标签':>8}{'缺标注':>8}")
    print("-" * 78)

    all_info: dict[str, dict] = {}
    for name, sp in splits.items():
        img_dir = find_images_dir(sp)
        imgs = list_images(img_dir)
        fmt = args.format if args.format != "auto" else detect_format(sp)
        if fmt == "unknown" and imgs:
            fmt = detect_format(img_dir.parent) or "unknown"

        lab_dir = labels_dir_for(img_dir)
        n_lab = 0
        n_inst = 0
        n_empty = 0
        n_missing = 0
        lab_map: dict[str, list] = {}

        if fmt == "yolo":
            lab_map = read_yolo_labels(lab_dir)
            n_lab = len(lab_map)
            n_inst = sum(len(v) for v in lab_map.values())
            for v in lab_map.values():
                if not v:
                    n_empty += 1
            stems = {p.stem for p in imgs}
            n_missing = len(stems - set(lab_map)) + 0
        elif fmt == "voc":
            lab_map = read_voc_labels(sp)
            n_lab = len(lab_map)
            n_inst = sum(len(v) for v in lab_map.values())
            stems = {p.stem for p in imgs}
            n_missing = len(stems - set(lab_map))
        elif fmt == "coco":
            lab_map, coco_names = read_coco_labels(sp)
            if not names and coco_names:
                names = coco_names
            n_lab = len(lab_map)
            n_inst = sum(len(v) for v in lab_map.values())
            stems = {p.stem for p in imgs}
            n_missing = len(stems - set(lab_map))

        print(f"{name:<8}{'✅' if sp.exists() else '❌':<10}{fmt:<10}"
              f"{human(len(imgs)):>8}{human(n_lab):>10}{human(n_inst):>8}"
              f"{human(n_empty):>8}{human(n_missing):>8}")

        all_info[name] = dict(path=str(sp), img_dir=str(img_dir), fmt=fmt,
                              n_img=len(imgs), n_lab=n_lab, n_inst=n_inst,
                              n_empty=n_empty, n_missing=n_missing,
                              imgs=imgs, lab_map=lab_map)

    if not names:
        print("\n⚠ 类别名未知：请检查 data.yaml 的 names 字段")

    # ---------- 类别分布 ----------
    section("③ 类别分布")
    dist_all: Counter = Counter()
    per_split: dict[str, Counter] = {}
    for name, info in all_info.items():
        c = Counter()
        if info["fmt"] == "yolo":
            for rows in info["lab_map"].values():
                for r in rows:
                    if r[0] >= 0:
                        c[r[0]] += 1
        elif info["fmt"] == "voc":
            for rows in info["lab_map"].values():
                for r in rows:
                    c[r[0]] += 1
        elif info["fmt"] == "coco":
            for rows in info["lab_map"].values():
                for r in rows:
                    if r[0] >= 0:
                        c[r[0]] += 1
        per_split[name] = c
        dist_all.update(c)

    total = sum(dist_all.values())
    tiny_cnt = 0
    if total:
        entries = []
        for k, v in dist_all.items():
            if isinstance(k, int) and names and 0 <= k < len(names):
                label = names[k]
            else:
                label = f"<id {k} ⚠越界>"
            entries.append((label, v))
        entries.sort(key=lambda x: -x[1])
        vmax = entries[0][1]
        print(f"{'类别':<22}{'实例数':>9}{'占比':>9}  分布")
        print("-" * 78)
        for label, v in entries:
            print(f"{label:<22}{human(v):>9}{pct(v, total):>9}  {bar(v, vmax)}")
        print("-" * 78)
        print(f"{'合计':<22}{human(total):>9}")
        if len(entries) >= 2:
            print(f"\n最不平衡比 : {entries[0][1] / max(1, entries[-1][1]):.1f} : 1"
                  f"   （{entries[0][0]} 最多，{entries[-1][0]} 最少）")

    # ---------- 图片尺寸 ----------
    section("④ 图片尺寸分布")
    all_imgs = [p for i in all_info.values() for p in i["imgs"]]
    scan = all_imgs if args.size_scan_max == 0 else all_imgs[:args.size_scan_max]
    sizes = image_sizes(scan)
    for (w, h), cnt in sizes.most_common(12):
        print(f"  {str(w) + '×' + str(h):<16}{human(cnt):>8}  {pct(cnt, len(scan))}")
    if len(sizes) > 12:
        print(f"  …… 另有 {len(sizes) - 12} 种尺寸")
    if len(sizes) > 3:
        print(f"\n⚠ 尺寸不统一（{len(sizes)} 种）—— 训练前确认 imgsz 与 letterbox 策略")
    broken = sizes.get(("BROKEN", 0), 0)
    if broken:
        print(f"\n❌ 有 {broken} 张图打不开（损坏）")

    # ---------- 框统计与质量检查 ----------
    section("⑤ 框尺寸 / 宽高比  +  ⑥ 标注质量检查")
    issues = defaultdict(list)
    areas, ratios = [], []
    for name, info in all_info.items():
        for stem, rows in info["lab_map"].items():
            for r in rows:
                if info["fmt"] == "yolo":
                    cls, cx, cy, w, h = r
                    if cls < 0 or w <= 0 or h <= 0:
                        issues["格式错"].append(f"{name}/{stem}")
                        continue
                    if names and cls >= len(names):
                        issues["类别id越界"].append(f"{name}/{stem} id={cls}")
                    if cx - w / 2 < -0.01 or cy - h / 2 < -0.01 or \
                       cx + w / 2 > 1.01 or cy + h / 2 > 1.01:
                        issues["框越界"].append(f"{name}/{stem}")
                    areas.append(w * h)
                    if h > 0:
                        ratios.append(w / h)
                elif info["fmt"] == "voc":
                    cname, x1, y1, x2, y2, W, H = r
                    if W and H:
                        bw, bh = (x2 - x1) / W, (y2 - y1) / H
                        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                            issues["框越界"].append(f"{name}/{stem}")
                        areas.append(bw * bh)
                        if bh > 0:
                            ratios.append(bw / bh)

    if areas:
        areas = np.array(areas)
        ratios = np.array(ratios) if ratios else np.array([1.0])
        print(f"框面积占比 :  中位 {np.median(areas):.4%}   "
              f"最小 {areas.min():.4%}   最大 {areas.max():.4%}")
        tiny = areas < max(1e-5, np.median(areas) / 100)
        tiny_cnt = int(tiny.sum())
        print(f"  极小框（< 中位面积的 1/100）: {human(tiny_cnt)} 个 "
              f"({pct(tiny_cnt, len(areas))})  —— 远距离病害可能天然很小，抽查即可")
        flat = (ratios > 12) | (ratios < 1 / 12)
        print(f"  极扁框（宽高比 >12 或 <1/12）: {human(int(flat.sum()))} 个 —— "
              f"⚠ 裂缝天然细长，此项仅供参考，别当异常")

    print()
    if issues:
        for k, v in issues.items():
            print(f"  ❌ {k:<12} {human(len(v)):>7} 个   例：{v[:2]}")
    else:
        print("  ✅ 未发现明显的标注格式问题")

    # ---------- 重复图 ----------
    if args.dup_sample > 0 and all_imgs:
        section("⑦ 重复图检测（差分哈希）")
        sample = all_imgs if args.dup_sample == 0 else \
            [all_imgs[i] for i in
             np.linspace(0, len(all_imgs) - 1, min(args.dup_sample, len(all_imgs))).astype(int)]
        hashes = []
        for p in sample:
            h = dhash(p)
            if h is not None:
                hashes.append((p, h))
        print(f"参与检测 {human(len(hashes))} 张（采样）……")
        groups, seen = [], set()
        for i in range(len(hashes)):
            if i in seen:
                continue
            grp = [hashes[i][0]]
            for j in range(i + 1, len(hashes)):
                if j in seen:
                    continue
                if hamming(hashes[i][1], hashes[j][1]) <= args.dup_threshold:
                    grp.append(hashes[j][0])
                    seen.add(j)
            if len(grp) > 1:
                seen.add(i)
                groups.append(grp)
        if groups:
            print(f"⚠ 检出 {len(groups)} 组近重复，共 {sum(len(g) for g in groups)} 张")
            for g in groups[:5]:
                print(f"   例：{[str(x.relative_to(root)) for x in g[:3]]}")
        else:
            print("✅ 未检出近重复")

    # ---------- 诊断 ----------
    section("⑧ 自动诊断")
    diag = []
    if not cfg:
        diag.append("⚠ 没有 data.yaml：类别名可能是猜的，训练前必须补上")
    if not names:
        diag.append("❌ 类别名未知：检查 data.yaml 的 names 字段")
    tot_img = sum(i["n_img"] for i in all_info.values())
    tot_inst = sum(i["n_inst"] for i in all_info.values())
    if "train" not in all_info:
        diag.append("❌ 没有 train split")
    if "val" not in all_info:
        diag.append("❌ 没有 val split —— 无法评估")
    if tot_img and tot_inst / tot_img < 1.0:
        diag.append(f"⚠ 每图平均只有 {tot_inst/tot_img:.2f} 个框，可能是病害稀疏场景")
    n_missing = sum(i["n_missing"] for i in all_info.values())
    if n_missing:
        diag.append(f"⚠ {n_missing} 张图没有对应标注文件（YOLO 里应写空 txt，否则会报错）")
    if issues.get("类别id越界"):
        diag.append("❌ 类别 id 越界 —— 与 data.yaml 的 names 不匹配")
    if issues.get("框越界"):
        diag.append("⚠ 有框越界（>1 或 <0）：可能坐标没归一化，或 VOC→YOLO 转换时漏减 xmin")
    if sizes.get(("BROKEN", 0)):
        diag.append("❌ 有损坏图片")
    if not diag:
        diag.append("✅ 一切正常，可以开始训练")

    for d in diag:
        print("  " + d)

    print()
    print("下一步建议：")
    print("  1) 先用 --dump-issues 导出有问题的样张，把框画回原图肉眼核对 20 张")
    print("  2) 确认类别顺序后，把 names 写进 configs/split.yaml（全项目只在这一处定义）")
    print("  3) 再跑 train.py —— 不要跳过上面两步")

    # ---------- JSON ----------
    if args.out:
        out = {
            "root": str(root),
            "data_yaml": str(yaml_path) if yaml_path else None,
            "classes": names,
            "splits": {k: {kk: vv for kk, vv in v.items() if kk not in ("imgs", "lab_map")}
                       for k, v in all_info.items()},
            "class_distribution": {str(k): v for k, v in dist_all.items()},
            "image_sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(30)},
            "issues": {k: len(v) for k, v in issues.items()},
            "tiny_boxes": tiny_cnt,
            "diagnosis": diag,
        }
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已存 JSON → {p}")

    # ---------- 导出问题样张 ----------
    if args.dump_issues:
        import cv2
        out_dir = Path("assets/probe_issues")
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for name, info in all_info.items():
            cls_names = names or []
            for stem, rows in info["lab_map"].items():
                if not rows:
                    continue
                cand = [p for p in info["imgs"] if p.stem == stem]
                if not cand:
                    continue
                img = cv2.imread(str(cand[0]))
                if img is None:
                    continue
                H, W = img.shape[:2]
                for r in rows:
                    if info["fmt"] == "yolo":
                        cls, cx, cy, w, h = r
                        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                        label = cls_names[cls] if 0 <= cls < len(cls_names) else str(cls)
                    elif info["fmt"] == "voc":
                        label, x1, y1, x2, y2 = r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4])
                    else:
                        continue
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, str(label), (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                dst = out_dir / f"{name}_{stem}.jpg"
                cv2.imwrite(str(dst), img)
                n += 1
                if n >= 40:
                    break
            if n >= 40:
                break
        print(f"\n已导出 {n} 张画框图 → {out_dir}  （★ 请肉眼核对框位置是否正确）")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
