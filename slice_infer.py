# -*- coding: utf-8 -*-
"""
slice_infer.py —— 切片推理 vs 整图推理 的对照实验

要解决的问题
------------
诊断发现：极小目标（面积 <0.3%）recall 只有 6%，大目标有 60%，差 10 倍。
D40 坑槽有 47% 是极小目标，所以它整体 recall 只有 21%。

切片推理的原理（这是 SAHI 的核心思想，不是简单"裁图"）
------------------------------------------------------
模型是在 imgsz=640 下训练的，它"见过"的目标尺度是固定的。
如果直接裁剪 320×320 的小块再喂给模型，块里的目标会显得很大，
但那需要把块缩放到 640 才会回到训练尺度 —— 这一步才是关键：

    裁剪 320×320  →  缩放到 640  →  目标在画面里放大 2 倍
    而模型仍然工作在自己熟悉的 imgsz=640 上（无分布偏移）

对比：
    整图 512×512  →  letterbox 到 640  →  放大 1.25 倍
    切片 320×320  →  缩放   到 640  →  放大 2 倍   ← 小目标更容易被发现

代价：大目标会被切片切断，且推理耗时按片数倍增。这个权衡就是实验要量化的。

用法
----
    # 快速验证（前 800 张）
    python slice_infer.py --max-images 800

    # 全量对照（A 整图 / B 切片 320 / C 整图 1024）
    python slice_infer.py --modes A,B,C

    # 自定义切片尺寸
    python slice_infer.py --modes A,B --tile 384 --imgsz 640 --overlap 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.detect import pick_weak_regions  # noqa: E402  E10 按需切片的判据

SHORT = ["D00", "D10", "D20", "D40"]
BUCKETS = [("极小<0.3%", 0.0, 0.003),
           ("小0.3~1%", 0.003, 0.01),
           ("中1~3%", 0.01, 0.03),
           ("大>3%", 0.03, 1.01)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/baseline/weights/best.pt")
    ap.add_argument("--data", default="data/raw/unified/data.yaml")
    ap.add_argument("--tile", type=int, default=448,
                    help="切片裁剪尺寸（像素）。★ 448 是实测最优（配合 overlap=0.25）")
    ap.add_argument("--imgsz", type=int, default=640, help="送进模型的尺寸")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="切片重叠比例。★ 0.25 实测最优：耗时是 0.5 的 1/6，精度只掉 0.5 点")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="评估命中阈值")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="跨片合并 NMS 阈值")
    ap.add_argument("--modes", default="A,B,C,D",
                    help="A=整图@imgsz  B=切片  C=整图@1024  D=整图+切片融合  "
                         "E=按需切片（整图→只切低置信框+空白网格）")
    ap.add_argument("--low-conf", type=float, default=0.45,
                    help="E 模式：整图置信度低于此值的框，其周边会被切片复查")
    ap.add_argument("--adaptive-grid", type=int, default=2,
                    help="E 模式：把图切成 grid×grid 大格，无检测的格子会被切片复查")
    ap.add_argument("--no-empty-fallback", action="store_true",
                    help="E 模式：不切「空白网格」，只切低置信框（更快但收益可能接近 0）")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 时随机抽这么多张（固定 seed）。★ 注意：直接取前 N 张不具代表性，"
                         "因为按文件名排序前段全是大目标图")
    ap.add_argument("--device", default="0")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_per_class(boxes, scores, clss, iou_thr=0.5):
    """按类别做贪心 NMS。boxes: list[[x1,y1,x2,y2]]"""
    keep_all = []
    for c in set(clss):
        idx = [i for i, v in enumerate(clss) if v == c]
        idx.sort(key=lambda i: -scores[i])
        keep = []
        while idx:
            i = idx.pop(0)
            keep.append(i)
            idx = [j for j in idx if iou_xyxy(boxes[i], boxes[j]) <= iou_thr]
        keep_all.extend(keep)
    return keep_all


def make_tiles(h, w, tile, overlap):
    """生成覆盖全图的切片坐标 (x1, y1, x2, y2)，含边界补齐。"""
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    out = []
    for y in ys:
        for x in xs:
            out.append((x, y, min(x + tile, w), min(y + tile, h)))
    return out


def load_val_list(data_yaml: Path):
    import yaml
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    base = data_yaml.parent
    v = cfg["val"]
    v = v[0] if isinstance(v, list) else v
    img_dir = base / v
    if not img_dir.is_dir():
        img_dir = base / str(v).replace("../", "")
    lab_dir = img_dir.parent.parent / "labels" / img_dir.name
    imgs = sorted([p for p in img_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    return imgs, lab_dir


def load_gt(lab: Path):
    gts = []
    if not lab.is_file():
        return gts
    for line in lab.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        gts.append((int(float(p[0])), *[float(x) for x in p[1:]]))
    return gts


# --------------------------------------------------------------------------- #
def run_mode(mode, model, imgs, lab_dir, args):
    """跑一种推理模式，返回 (统计, 耗时秒, 总检测框数)"""
    stats = {c: {b[0]: [0, 0] for b in BUCKETS} for c in range(4)}
    total_det = 0
    n_tiles_used = 0          # E 模式：累计切了多少块（用于算平均开销）
    t0 = time.time()

    infer_imgsz = 1024 if mode == "C" else args.imgsz
    use_slice = mode in ("B", "D")
    use_whole = mode in ("A", "C", "D")      # ★ D 模式：整图 + 切片都跑
    use_adaptive = mode == "E"               # ★ E 模式：整图 → 只切薄弱区域

    for n, p in enumerate(imgs, 1):
        gts = load_gt(lab_dir / (p.stem + ".txt"))
        if not gts:
            continue
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes, scores, clss = [], [], []

        if use_adaptive:
            # ① 整图先跑
            rw0 = model.predict(img, imgsz=args.imgsz, conf=args.conf,
                                verbose=False, device=args.device)[0]
            whole = []
            if rw0.boxes is not None and len(rw0.boxes):
                for b, cf, c in zip(rw0.boxes.xyxy.tolist(), rw0.boxes.conf.tolist(),
                                    rw0.boxes.cls.tolist()):
                    whole.append({"cls": int(c), "conf": float(cf),
                                  "xyxy": [float(x) for x in b]})
                    boxes.append([float(x) for x in b])
                    scores.append(cf)
                    clss.append(int(c))
            # ② 只切薄弱区域（与 core/detect.py 用同一套判据）
            tiles = pick_weak_regions(whole, h, w, args.tile,
                                      grid=args.adaptive_grid,
                                      low_conf=args.low_conf,
                                      empty_fallback=not args.no_empty_fallback)
            n_tiles_used += len(tiles)
        elif use_slice:
            tiles = make_tiles(h, w, args.tile, args.overlap)
        else:
            tiles = [(0, 0, w, h)]

        # 批量推理所有切片
        # ★ 必须防空列表：E 模式下若整图已检满、又没有空白网格，
        #   tiles 会是空的，而 ultralytics 对空输入直接抛
        #   FileNotFoundError("No images found in source")。
        #   这种情况应当退化为「只用整图结果」，而不是判错。
        if tiles:
            crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]
            results = model.predict(crops, imgsz=infer_imgsz, conf=args.conf,
                                    verbose=False, device=args.device)
        else:
            results = []

        for (x1, y1, _x2, _y2), r in zip(tiles, results):
            if r.boxes is None or not len(r.boxes):
                continue
            for b, cf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                r.boxes.cls.tolist()):
                boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
                scores.append(cf)
                clss.append(int(c))

        # ★ D 模式：再跑一遍整图，把整图结果和切片结果融合
        if use_whole and use_slice:
            rw = model.predict(img, imgsz=args.imgsz, conf=args.conf,
                               verbose=False, device=args.device)[0]
            if rw.boxes is not None and len(rw.boxes):
                for b, cf, c in zip(rw.boxes.xyxy.tolist(), rw.boxes.conf.tolist(),
                                    rw.boxes.cls.tolist()):
                    boxes.append([float(x) for x in b])
                    scores.append(cf)
                    clss.append(int(c))

        total_det += len(boxes)

        # 跨片/跨源合并
        if len(boxes) > 1:
            keep = nms_per_class(boxes, scores, clss, args.nms_iou)
            boxes = [boxes[i] for i in keep]
            clss = [clss[i] for i in keep]

        # 评估
        for (c, cx, cy, bw, bh) in gts:
            area = bw * bh
            bucket = next((bn for bn, lo, hi in BUCKETS if lo <= area < hi), BUCKETS[-1][0])
            gx1, gy1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            gx2, gy2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            best = 0.0
            for b, pc in zip(boxes, clss):
                if pc != c:
                    continue
                best = max(best, iou_xyxy((gx1, gy1, gx2, gy2), b))
            hit = best >= args.iou
            stats[c][bucket][1] += 1
            if hit:
                stats[c][bucket][0] += 1

        if n % 200 == 0:
            print(f"    [{mode}] {n}/{len(imgs)} …", flush=True)

    return stats, time.time() - t0, total_det, n_tiles_used

def report(mode, stats, secs, ndet, n_img, title):
    all_hit = sum(v[0] for c in stats for v in stats[c].values())
    all_gt = sum(v[1] for c in stats for v in stats[c].values())
    print(f"\n{'='*94}")
    print(f"  {mode}  {title}")
    print(f"{'='*94}")
    print(f"{'类别':<8}" + "".join(f"{b[0]:>15}" for b in BUCKETS) + f"{'合计':>13}")
    print("-" * 94)
    for c in range(4):
        line = f"{SHORT[c]:<8}"
        th = ta = 0
        for b in BUCKETS:
            hit, tot = stats[c][b[0]]
            th += hit
            ta += tot
            line += f"{(f'{hit}/{tot} {hit/tot*100:3.0f}%' if tot else '      -'):>15}"
        line += f"{(f'{th}/{ta} {th/ta*100:3.0f}%' if ta else '-'):>13}"
        print(line)
    print("-" * 94)
    print(f"{'全部':<8}" + " " * 15 * 4 + f"{all_hit}/{all_gt} {all_hit/all_gt*100:.1f}%")
    print(f"耗时 {secs/60:.1f} 分钟   每张 {secs/n_img*1000:.0f} ms   "
          f"平均每张检测框 {ndet/n_img:.1f} 个")
    # 小目标桶单独报
    sh = sum(stats[c][BUCKETS[0][0]][0] for c in range(4))
    st = sum(stats[c][BUCKETS[0][0]][1] for c in range(4))
    bh = sum(stats[c][BUCKETS[-1][0]][0] for c in range(4))
    bt = sum(stats[c][BUCKETS[-1][0]][1] for c in range(4))
    print(f"★ 极小目标 recall {sh}/{st} = {sh/st*100:.1f}%   "
          f"| 大目标 recall {bh}/{bt} = {bh/bt*100:.1f}%   "
          f"| 尺寸比 {(bh/bt)/(sh/st) if sh and st else 0:.1f}×")
    return all_hit / all_gt


def main(argv=None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    wp = (HERE / args.weights) if not Path(args.weights).is_absolute() else Path(args.weights)
    if not wp.is_file():
        print(f"找不到权重 {wp}")
        return 1
    imgs, lab_dir = load_val_list(HERE / args.data)
    if args.sample > 0:
        import random
        rnd = random.Random(42)
        imgs = rnd.sample(imgs, min(args.sample, len(imgs)))
        imgs.sort()
        print(f"随机采样 {len(imgs):,} 张（seed=42）")
    elif args.max_images:
        imgs = imgs[:args.max_images]
        print(f"⚠ 取前 {len(imgs)} 张 —— 前段文件全是大目标图，结果不可代表全量")
    print(f"权重 {wp}")
    print(f"验证图片 {len(imgs):,}   切片 {args.tile}×{args.tile}  重叠 {args.overlap}  "
          f"送模型 imgsz={args.imgsz}")

    model = YOLO(str(wp))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    titles = {"A": f"整图推理 @ imgsz={args.imgsz}（基线）",
              "B": f"切片推理  {args.tile}×{args.tile} → {args.imgsz}（放大 {args.imgsz/args.tile:.1f}×）",
              "C": "整图推理 @ imgsz=1024（直接提高分辨率）",
              "D": f"整图 + 切片融合（{args.tile}×{args.tile} → {args.imgsz}）★",
              "E": f"按需切片：整图 → 只切低置信框+空白网格（tile {args.tile}，"
                   f"low_conf {args.low_conf}，grid {args.adaptive_grid}）★"}
    summary = {}
    tile_info = {}
    for m in modes:
        print(f"\n--- 跑模式 {m} ---", flush=True)
        st, secs, ndet, ntl = run_mode(m, model, imgs, lab_dir, args)
        tile_info[m] = ntl
        summary[m] = report(m, st, secs, ndet, len(imgs), titles.get(m, m))

    print(f"\n{'='*94}")
    print("  横向对比")
    print(f"{'='*94}")
    base = summary.get(modes[0], 0)
    for m in modes:
        v = summary[m]
        extra = ""
        if tile_info.get(m):
            extra = f"   平均切块 {tile_info[m] / max(1, len(imgs)):.1f} 块/张"
        print(f"  {m}  recall {v*100:5.1f}%   {'（基线）' if m == modes[0] else f'相对基线 {(v-base)*100:+.1f} 个点'}{extra}")
    print("\n把这张表保存进 docs/experiments.md，它就是 M2 模块的实验数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
