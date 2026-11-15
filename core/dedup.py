# -*- coding: utf-8 -*-
"""
dedup.py —— 连续帧去重（M5 模块）

为什么需要它
------------
行车记录仪是**视频流**。路上的一个坑，车开过去会被连续拍到 30~50 帧。
如果逐帧输出检测结果，一份病害清单里同一个坑会出现几十次 —— 这在上报时
完全不可用。

要交出去的是一份「**唯一点位清单**」，不是「帧清单」。

怎么判断"两帧里是同一个病害"
----------------------------
两个条件**同时**满足才算同一组，缺一不可：

    ① 整帧相似：差分哈希（dHash）的汉明距离 <= 阈值
       —— 说明是同一段路
    ② 框重合：两个框的 IoU >= 阈值，且类别相同
       —— 说明是同一个病害

**只判 pHash 会出错**：同一段路上前后两个不同的坑，整帧很像，但它们是两个病害。
所以必须叠加框位置判断。

参数（configs/dedup.yaml）
-------------------------
    phash_hamming_max  0~64，越小越严。实测：同段路相邻帧 0~6，不同路段 >20
    iou_min            框重合度下限
    max_gap_frames     两帧间隔超过这么多帧就断开（防止把远处的同款病害并进来）
    same_class_only    只合并同类别的框
    keep_strategy      组内保留哪一帧：conf / area / middle
"""
from __future__ import annotations

import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)


def dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    """差分哈希。比平均哈希稳，比 pHash 快，不需要额外依赖。"""
    import cv2

    g = _gray(img_bgr)
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]     # 相邻像素的亮度关系
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def group_frames(images: list[np.ndarray],
                 detections: list[list[dict]],
                 cfg: dict | None = None) -> dict:
    """把连续帧的检测框归并成「唯一点位」。

    parameters:
        images      每帧的 BGR 图
        detections  每帧的检测结果 [{"cls": int, "conf": float, "xyxy": [...]}, ...]
        cfg         configs/dedup.yaml

    returns:
        {
          "n_raw_boxes": int, "n_unique_points": int, "reduction_ratio": float,
          "groups": [{"group_id", "class", "frames", "kept_frame", "box", "max_conf"}]
        }
    """
    cfg = cfg or {}
    ham_max = cfg.get("phash_hamming_max", 8)
    iou_min = cfg.get("iou_min", 0.3)
    max_gap = cfg.get("max_gap_frames", 45)
    same_cls = cfg.get("same_class_only", True)
    keep = cfg.get("keep_strategy", "conf")

    n = len(images)
    hashes = [dhash(im) for im in images]
    n_raw = sum(len(d) for d in detections)

    groups: list[dict] = []

    def find_or_make(frame_i: int, det: dict) -> dict:
        """在某帧的某个检测框上，找它属于哪个已有组；没有就新建。"""
        best, best_score = None, -1.0
        for g in groups:
            last = g["frames"][-1]
            if frame_i - last > max_gap:
                continue                                  # 太远，不算同一段
            if same_cls and g["class"] != det["cls"]:
                continue
            if hamming(hashes[last], hashes[frame_i]) > ham_max:
                continue                                  # 整帧不像，不是同一段路
            ov = iou(g["box"], det["xyxy"])
            if ov < iou_min:
                continue
            if ov > best_score:
                best, best_score = g, ov
        if best is None:
            g = {"group_id": len(groups) + 1, "class": det["cls"],
                 "frames": [frame_i], "kept_frame": frame_i,
                 "box": list(det["xyxy"]), "max_conf": det["conf"]}
            groups.append(g)
            return g
        best["frames"].append(frame_i)
        if det["conf"] > best["max_conf"]:
            best["max_conf"] = det["conf"]
            if keep == "conf":
                best["kept_frame"] = frame_i
                best["box"] = list(det["xyxy"])
        return best

    for fi, dets in enumerate(detections):
        for det in dets:
            find_or_make(fi, det)

    n_uniq = len(groups)
    return {
        "n_frames": n,
        "n_raw_boxes": n_raw,
        "n_unique_points": n_uniq,
        "reduction_ratio": round(n_raw / n_uniq, 2) if n_uniq else 0.0,
        "groups": sorted(groups, key=lambda g: (g["class"], g["group_id"]))[:200],
    }
