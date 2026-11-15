# -*- coding: utf-8 -*-
"""
detect.py —— 模型单例 + 三种推理模式（M1 / M2 模块的底座）

三种模式（实验数据见 docs/experiments.md E3/E5）
------------------------------------------------
    whole    整图推理                    基线，25 ms/张
    slice    切片推理（切小再放大）       小目标强，大目标弱
    fusion   整图 + 切片 按类 NMS 合并   ★ 最优，+8 个点 @ 123 ms/张

融合为什么有效
--------------
切片把目标放大，小目标更容易检出，但大目标会被切片切断；
整图对大目标好。两者结果**合并**（而不是替换），各取所长。

注意：必须按类别做 NMS，否则不同类别的框会互相压制。
"""
from __future__ import annotations

import numpy as np

_models: dict[str, object] = {}
CLASS_ZH = ["纵向裂缝", "横向裂缝", "龟裂", "坑槽"]
CLASS_SHORT = ["D00", "D10", "D20", "D40"]
DEFAULT_TILE = 448
DEFAULT_OVERLAP = 0.25


def load(weights: str) -> object:
    """幂等加载模型并预热。

    按权重路径分别缓存（不再只留一个单例）——演示台要现场对比
    「640 基线」和「1024 重训」两个权重，若只缓存一个，每切换一次就
    要重新加载 5 MB 权重 + 预热，两次调用会互相当对方是缓存失效。
    """
    import os
    w = os.path.abspath(weights)
    hit = _models.get(w)
    if hit is not None:
        return hit
    if not os.path.isfile(w):
        raise FileNotFoundError(f"找不到权重 {w}")
    from ultralytics import YOLO
    m = YOLO(w)
    # 预热：形状/dtype 与真实输入一致
    m.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    _models[w] = m
    return m


def make_tiles(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y, min(x + tile, w), min(y + tile, h))
            for y in sorted(set(ys)) for x in sorted(set(xs))]


def _nms_per_class(boxes, scores, clss, iou_thr=0.5) -> list[int]:
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    keep_all = []
    for c in set(clss):
        idx = [i for i, v in enumerate(clss) if v == c]
        idx.sort(key=lambda i: -scores[i])
        keep = []
        while idx:
            i = idx.pop(0)
            keep.append(i)
            idx = [j for j in idx if _iou(boxes[i], boxes[j]) <= iou_thr]
        keep_all.extend(keep)
    return keep_all


def _to_dets(r) -> list[dict]:
    out = []
    if r.boxes is None or not len(r.boxes):
        return out
    for b, cf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        out.append({"cls": int(c), "conf": float(cf), "xyxy": [float(x) for x in b]})
    return out


def pick_weak_regions(dets: list[dict], h: int, w: int, tile: int,
                      grid: int = 2, low_conf: float = 0.45,
                      empty_fallback: bool = True) -> list[tuple[int, int, int, int]]:
    """从整图结果里挑出「值得再放大看一遍」的区域（E10 按需切片的核心）。

    判据（两条，任一命中即可）
    ------------------------
    ① 低置信框：`conf < low_conf` —— 有嫌疑但没把握，放大确认
    ② 空白网格：把图切成 `grid×grid` 大格，某些格里**一个框都没有** ——
       整图看不见不等于没有，极小目标往往就藏在这些安静区域里

    ⚠️ 设计取舍（实测后的结论写在这里，避免被问倒）
    -------------------------------------------------
    判据②是**必要**的：E3 的尺寸分桶显示极小目标 recall 只有 6%，
    说明绝大多数小目标压根没被检出 —— 只切「低置信框」等于只处理那 6% 之外的噪音，
    真正的漏检区域（完全无框）反而被跳过，测出来收益会接近 0。

    但判据②也很贵：路面图大部分格子是空的，全切就退化成 fusion。
    所以用 `empty_fallback` 控制：
      True  —— 有空网格就切（保召回，可能接近 fusion 的开销）
      False —— 只切低置信框（保速度，可能几乎没收益）

    returns: 去重合并后的切片框列表 [(x1,y1,x2,y2), ...]
    """
    regions: list[tuple[int, int, int, int]] = []

    # ① 低置信框 → 往外扩一圈（目标常在框边缘被切断）
    margin = max(8, tile // 6)
    for d in dets:
        if d["conf"] >= low_conf:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        regions.append((max(0, int(x1) - margin), max(0, int(y1) - margin),
                        min(w, int(x2) + margin), min(h, int(y2) + margin)))

    # ② 空白网格
    if empty_fallback:
        gw, gh = w / grid, h / grid
        occupied = [[False] * grid for _ in range(grid)]
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gx = min(grid - 1, max(0, int(cx / gw)))
            gy = min(grid - 1, max(0, int(cy / gh)))
            occupied[gy][gx] = True
        for gy in range(grid):
            for gx in range(grid):
                if occupied[gy][gx]:
                    continue
                regions.append((int(gx * gw), int(gy * gh),
                                int((gx + 1) * gw), int((gy + 1) * gh)))

    # 合并严重重叠的区域（避免同一块切两次），再把小区域撑到 tile 大小
    merged: list[list[int]] = []
    for r in regions:
        hit = None
        for mreg in merged:
            ix = min(mreg[2], r[2]) - max(mreg[0], r[0])
            iy = min(mreg[3], r[3]) - max(mreg[1], r[1])
            if ix > 0 and iy > 0:
                inter = ix * iy
                small = min((mreg[2] - mreg[0]) * (mreg[3] - mreg[1]),
                            (r[2] - r[0]) * (r[3] - r[1]))
                if small > 0 and inter / small > 0.5:
                    hit = mreg
                    break
        if hit is None:
            merged.append(list(r))
        else:
            hit[0], hit[1] = min(hit[0], r[0]), min(hit[1], r[1])
            hit[2], hit[3] = max(hit[2], r[2]), max(hit[3], r[3])

    # 撑到 tile 尺寸并夹在图像边界内（太小的 patch 放大后信息量不够）
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in merged:
        bw, bh = x2 - x1, y2 - y1
        if bw < tile:
            cx = (x1 + x2) // 2
            x1, x2 = cx - tile // 2, cx + tile // 2
        if bh < tile:
            cy = (y1 + y2) // 2
            y1, y2 = cy - tile // 2, cy + tile // 2
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = min(w, max(x2, x1 + 1))
        y2 = min(h, max(y2, y1 + 1))
        out.append((x1, y1, x2, y2))
    return out


def infer(image_bgr: np.ndarray, weights: str, mode: str = "whole",
          conf: float = 0.25, imgsz: int = 640,
          tile: int = DEFAULT_TILE, overlap: float = DEFAULT_OVERLAP,
          nms_iou: float = 0.5, low_conf: float = 0.45,
          adaptive_grid: int = 2, empty_fallback: bool = True) -> dict:
    """对单张图推理。

    mode
    ----
    whole     整图推理（基线）
    slice     全量切片
    fusion    整图 + 全量切片，按类 NMS 合并
    adaptive  ★ 按需切片（E10）：整图先跑 → 只对「低置信框 + 空白网格」再切

    returns:
        {"mode", "dets": [...], "n_dets", "elapsed_ms", "n_tiles", "img_shape"}
    """
    import time
    m = load(weights)
    h, w = image_bgr.shape[:2]
    t0 = time.time()

    dets: list[dict] = []
    n_tiles = 0

    def _run_tiles(tiles):
        nonlocal n_tiles
        if not tiles:
            return []
        crops = [image_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]
        rs = m.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
        n_tiles += len(tiles)
        got = []
        for (x1, y1, _x2, _y2), r in zip(tiles, rs):
            for d in _to_dets(r):
                d["xyxy"] = [d["xyxy"][0] + x1, d["xyxy"][1] + y1,
                             d["xyxy"][2] + x1, d["xyxy"][3] + y1]
                got.append(d)
        return got

    if mode == "adaptive":
        # 第一阶段：整图（快）
        r0 = m.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
        whole = _to_dets(r0)
        dets.extend(whole)
        # 第二阶段：只切薄弱区域
        tiles = pick_weak_regions(whole, h, w, tile, grid=adaptive_grid,
                                  low_conf=low_conf, empty_fallback=empty_fallback)
        dets.extend(_run_tiles(tiles))
    else:
        if mode in ("slice", "fusion"):
            dets.extend(_run_tiles(make_tiles(h, w, tile, overlap)))
        if mode in ("whole", "fusion"):
            r = m.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
            if mode == "whole":
                dets = _to_dets(r)
            else:
                dets.extend(_to_dets(r))

    if len(dets) > 1:
        keep = _nms_per_class([d["xyxy"] for d in dets],
                              [d["conf"] for d in dets],
                              [d["cls"] for d in dets], nms_iou)
        dets = [dets[i] for i in keep]
        dets.sort(key=lambda d: -d["conf"])

    return {"mode": mode, "dets": dets, "n_dets": len(dets),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "n_tiles": n_tiles, "img_shape": [int(h), int(w)],
            "tile": tile, "overlap": overlap, "imgsz": imgsz}


def draw(image_bgr: np.ndarray, dets: list[dict], show_conf: bool = True) -> np.ndarray:
    """画框（不修改入参）。按类别配色。"""
    import cv2
    colors = [(0, 200, 255), (0, 255, 120), (255, 180, 0), (80, 80, 255)]  # BGR
    out = image_bgr.copy()
    for d in dets:
        c = d["cls"]
        col = colors[c % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{CLASS_SHORT[c % 4]}"
        if show_conf:
            label += f" {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), col, -1)
        cv2.putText(out, label, (x1 + 1, max(th, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out
