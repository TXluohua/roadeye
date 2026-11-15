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

_model = None
_model_path: str | None = None

CLASS_ZH = ["纵向裂缝", "横向裂缝", "龟裂", "坑槽"]
CLASS_SHORT = ["D00", "D10", "D20", "D40"]
DEFAULT_TILE = 448
DEFAULT_OVERLAP = 0.25


def load(weights: str) -> object:
    """幂等加载模型并预热。"""
    global _model, _model_path
    import os
    w = os.path.abspath(weights)
    if _model is not None and _model_path == w:
        return _model
    if not os.path.isfile(w):
        raise FileNotFoundError(f"找不到权重 {w}")
    from ultralytics import YOLO
    _model = YOLO(w)
    # 预热：形状/dtype 与真实输入一致
    _model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    _model_path = w
    return _model


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


def infer(image_bgr: np.ndarray, weights: str, mode: str = "whole",
          conf: float = 0.25, imgsz: int = 640,
          tile: int = DEFAULT_TILE, overlap: float = DEFAULT_OVERLAP,
          nms_iou: float = 0.5) -> dict:
    """对单张图推理。

    returns:
        {"mode", "dets": [...], "n_dets", "elapsed_ms", "n_tiles", "img_shape"}
    """
    import time
    m = load(weights)
    h, w = image_bgr.shape[:2]
    t0 = time.time()

    dets: list[dict] = []
    n_tiles = 0
    if mode in ("slice", "fusion"):
        tiles = make_tiles(h, w, tile, overlap)
        crops = [image_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]
        rs = m.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
        n_tiles = len(tiles)
        for (x1, y1, _x2, _y2), r in zip(tiles, rs):
            for d in _to_dets(r):
                d["xyxy"] = [d["xyxy"][0] + x1, d["xyxy"][1] + y1,
                             d["xyxy"][2] + x1, d["xyxy"][3] + y1]
                dets.append(d)

    if mode in ("whole", "fusion"):
        if mode == "whole":
            r = m.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
            dets = _to_dets(r)
        else:
            r = m.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
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
