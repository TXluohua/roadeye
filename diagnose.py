# -*- coding: utf-8 -*-
"""
diagnose.py —— 找出「哪一类为什么差」

思路：在验证集上逐张推理，把每个 GT 框按面积分桶（小/中/大），
统计每个 (类别 × 尺寸桶) 的检出率。这样能区分：

  · 如果所有类在小目标桶都很差 → 是分辨率问题（提高 imgsz / 切片推理）
  · 如果只有 D40 在所有桶都差   → 是类别定义/标注歧义问题（重定义或重标）
  · 如果 D40 只在"大目标"桶好   → 是样本尺度分布问题（补近距离坑槽样本）

用法：
    python diagnose.py                                  # 用 runs/baseline/weights/best.pt
    python diagnose.py --weights runs/x/weights/best.pt
    python diagnose.py --conf 0.25 --iou 0.5 --max-images 2000
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["Longitudinal Crack (D00)", "Transverse Crack (D10)",
         "Alligator Crack (D20)", "Pothole (D40)"]
SHORT = ["D00", "D10", "D20", "D40"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/baseline/weights/best.pt")
    ap.add_argument("--data", default="data/raw/unified/data.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="判定命中用的 IoU 阈值")
    ap.add_argument("--max-images", type=int, default=0, help="0=全部")
    ap.add_argument("--device", default="0")
    return ap.parse_args(argv)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv=None) -> int:
    args = parse_args(argv)
    import yaml
    from ultralytics import YOLO

    wp = (HERE / args.weights) if not Path(args.weights).is_absolute() else Path(args.weights)
    if not wp.is_file():
        print(f"找不到权重 {wp}")
        return 1

    # 读 val 图片
    dcfg = yaml.safe_load(open(HERE / args.data, encoding="utf-8"))
    base = (HERE / args.data).parent
    val = dcfg["val"]
    val = val[0] if isinstance(val, list) else val
    img_dir = base / val
    if not img_dir.is_dir():
        img_dir = base / str(val).replace("../", "")
    imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    lab_dir = img_dir.parent.parent / "labels" / img_dir.name
    if args.max_images > 0:
        imgs = imgs[:args.max_images]
    print(f"验证图片 {len(imgs):,} 张   标签目录 {lab_dir}")
    print(f"权重 {wp.name}   conf={args.conf}  IoU 阈值={args.iou}\n")

    model = YOLO(str(wp))

    # 尺寸桶：按 GT 框面积占整图的比例
    BUCKETS = [("极小 <0.3%", 0.0, 0.003),
               ("小 0.3~1%", 0.003, 0.01),
               ("中 1~3%", 0.01, 0.03),
               ("大 >3%", 0.03, 1.01)]
    # stats[cls][bucket] = [命中, 总数]
    stats = {c: {b[0]: [0, 0] for b in BUCKETS} for c in range(len(SHORT))}
    all_gt = 0
    all_hit = 0
    miss_by_cls = defaultdict(int)

    for n, p in enumerate(imgs, 1):
        lab = lab_dir / (p.stem + ".txt")
        if not lab.is_file():
            continue
        gts = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:])
            gts.append((c, cx, cy, w, h))
        if not gts:
            continue

        r = model.predict(str(p), conf=args.conf, verbose=False, device=args.device)[0]
        dh, dw = r.orig_shape
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                preds.append((int(c), b, cf))

        for (c, cx, cy, w, h) in gts:
            area = w * h
            bucket = next((bn for bn, lo, hi in BUCKETS if lo <= area < hi), BUCKETS[-1][0])
            # 转成像素框
            gx1, gy1 = (cx - w / 2) * dw, (cy - h / 2) * dh
            gx2, gy2 = (cx + w / 2) * dw, (cy + h / 2) * dh
            best = 0.0
            for (pc, pb, _cf) in preds:
                if pc != c:
                    continue
                best = max(best, iou_xyxy((gx1, gy1, gx2, gy2), pb))
            hit = best >= args.iou
            stats[c][bucket][1] += 1
            all_gt += 1
            if hit:
                stats[c][bucket][0] += 1
                all_hit += 1
            else:
                miss_by_cls[c] += 1
        if n % 500 == 0:
            print(f"  已处理 {n}/{len(imgs)} …", flush=True)

    # ---------- 输出 ----------
    print("\n" + "=" * 92)
    print("检出率（Recall @ IoU>=%.2f, conf>=%.2f）按 目标尺寸 分桶" % (args.iou, args.conf))
    print("=" * 92)
    hdr = f"{'类别':<24}" + "".join(f"{b[0]:>14}" for b in BUCKETS) + f"{'合计':>12}"
    print(hdr)
    print("-" * 92)
    for c in range(len(SHORT)):
        line = f"{SHORT[c]:<24}"
        tot_hit = tot_all = 0
        for b in BUCKETS:
            hit, tot = stats[c][b[0]]
            tot_hit += hit
            tot_all += tot
            line += f"{f'{hit}/{tot} {hit/tot*100:4.0f}%' if tot else '      -':>14}"
        line += f"{f'{tot_hit}/{tot_all} {tot_hit/tot_all*100:4.0f}%' if tot_all else '-':>12}"
        print(line)
    print("-" * 92)
    print(f"\n整体检出率 {all_hit}/{all_gt} = {all_hit/all_gt*100:.1f}%")
    print(f"漏检数：", {SHORT[c]: v for c, v in sorted(miss_by_cls.items())})

    # ---------- 自动结论 ----------
    print("\n" + "=" * 92)
    print("自动结论")
    print("=" * 92)
    small_key = BUCKETS[0][0]
    big_key = BUCKETS[-1][0]
    # 1) 是否所有类都在小目标上崩
    small_recalls = [stats[c][small_key][0] / stats[c][small_key][1]
                     for c in range(len(SHORT)) if stats[c][small_key][1] >= 20]
    big_recalls = [stats[c][big_key][0] / stats[c][big_key][1]
                   for c in range(len(SHORT)) if stats[c][big_key][1] >= 20]
    if small_recalls and big_recalls:
        ms, mb = np.mean(small_recalls), np.mean(big_recalls)
        print(f"· 平均：小目标桶 {ms*100:.0f}%  vs  大目标桶 {mb*100:.0f}%")
        if mb - ms > 0.30:
            print("  → 尺寸效应显著：小目标严重吃亏")
            print("    首选 **融合切片推理**（实测 +8.0 点，见 docs/experiments.md E5）")
            print("    单纯提高 imgsz 收益有限（E4 实测等轮次仅 +2.75 点，成本翻倍）")
        else:
            print("  → 尺寸效应不显著：问题不在分辨率，别急着调 imgsz")
    # 2) D40 是否在所有桶都差
    d40 = stats[3]
    d40_rec = [d40[b[0]][0] / d40[b[0]][1] for b in BUCKETS if d40[b[0]][1] >= 20]
    others = [stats[c][b[0]][0] / stats[c][b[0]][1]
              for c in range(3) for b in BUCKETS if stats[c][b[0]][1] >= 20]
    if d40_rec and others:
        r40, ro = np.mean(d40_rec), np.mean(others)
        print(f"· D40 各桶平均 {r40*100:.0f}%  vs  其他三类平均 {ro*100:.0f}%")
        if ro - r40 > 0.25:
            print("  → D40 在所有尺寸上普遍差，不只是小目标问题")
            print("    首选排查：① D40 标注是否混入了修补痕迹/阴影  ② 是否过采样 D40")
            print("    可跑：python diagnose.py --max-images 200 （先快速看一轮）")

    print("\n结论已按上表自动判定。各尺寸桶的完整实验数据见 docs/experiments.md 的 E2 与 E3。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
