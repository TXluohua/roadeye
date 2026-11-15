# -*- coding: utf-8 -*-
"""
quality.py —— 图像质量前置（M4 模块）

为什么需要它
------------
行车记录仪拍的图里，总有一部分是"不可判定"的：夜里过曝、雨天反光、
高速运动糊掉、隧道口明暗对比炸掉。把这些图送进模型，只会得到一堆误检，
而且会污染病害清单。

正确做法：**先判"这张图能不能用"，不能用的直接跳过并记录**，
而不是硬给一个结果。

四个指标
--------
两个来自相机物理特性，两个来自图像统计：

    sharpness   Laplacian 方差 —— 越糊越小
    brightness  灰度均值      —— 太暗/太亮都不行
    overexposure 过曝像素占比  —— 高光处路面细节全部丢失
    contrast    灰度标准差    —— 太低说明一片灰，没有可用纹理

阈值由 `tools/calibrate_quality.py` 在验证集上实测标定后回填到
configs/quality.yaml，标定产物见 assets/results/quality_calib.json。

★ 一个实测教训：过曝阈值最初定 0.05，在 1200 张验证图上把 28.3% 的
**正常图**误判为"过曝"——因为路面高光、天空、白线本身就是超白像素。
放宽到 0.20 后误杀降到 8.3%。**前置过滤过严比不过滤更糟**：它会把
能用的图丢掉，而丢掉的图不会报错，只会静默消失。
"""
from __future__ import annotations

import numpy as np


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    # BGR -> 灰度（OpenCV 权重）
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)


def assess(img_bgr: np.ndarray, cfg: dict | None = None) -> dict:
    """评估单张图的质量。

    parameters:
        img_bgr  OpenCV 读出的 BGR 图（或灰度图）
        cfg      configs/quality.yaml 的内容

    returns:
        {
          "sharpness": float, "brightness": float,
          "overexposure": float, "underexposure": float, "contrast": float,
          "usable": bool, "reason": str, "fails": [str]
        }
    """
    import cv2

    cfg = cfg or {}
    gray = _to_gray(img_bgr)

    # ① 清晰度：Laplacian 响应的方差。糊的图高频成分少，方差小
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # ② 亮度均值
    brightness = float(gray.mean())
    # ③ 过曝 / 欠曝像素占比
    overexposure = float((gray > 250).mean())
    underexposure = float((gray < 5).mean())
    # ④ 对比度
    contrast = float(gray.std())

    fails: list[str] = []
    if sharpness < cfg.get("sharpness_min", 60.0):
        fails.append(f"模糊（清晰度 {sharpness:.0f} < {cfg.get('sharpness_min', 60.0)}）")
    lo, hi = cfg.get("brightness_range", [30, 225])
    if not (lo <= brightness <= hi):
        fails.append(f"亮度异常（{brightness:.0f} 不在 {lo}~{hi}）")
    if overexposure > cfg.get("overexposure_max", 0.05):
        fails.append(f"过曝（{overexposure * 100:.1f}% > {cfg.get('overexposure_max', 0.05) * 100:.0f}%）")
    if underexposure > cfg.get("underexposure_max", 0.30):
        fails.append(f"欠曝（{underexposure * 100:.1f}%）")
    if contrast < cfg.get("contrast_min", 18.0):
        fails.append(f"对比度低（{contrast:.1f} < {cfg.get('contrast_min', 18.0)}）")

    usable = len(fails) < cfg.get("fail_votes", 1)
    return {
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "overexposure": round(overexposure, 4),
        "underexposure": round(underexposure, 4),
        "contrast": round(contrast, 1),
        "usable": usable,
        "reason": "；".join(fails) if fails else "正常",
        "fails": fails,
    }


def inject_degradation(img_bgr: np.ndarray, kind: str, level: float) -> np.ndarray:
    """生成退化图，用于标定阈值和演示。

    kind: blur / dark / overexpose / noise / jpeg
    level: 0~1
    """
    import cv2

    out = img_bgr.copy()
    if kind == "blur":
        k = int(3 + level * 26) | 1        # 奇数核
        return cv2.GaussianBlur(out, (k, k), 0)
    if kind == "dark":
        return np.clip(out.astype(np.float32) * (1 - 0.9 * level), 0, 255).astype(np.uint8)
    if kind == "overexpose":
        return np.clip(out.astype(np.float32) + 255 * level, 0, 255).astype(np.uint8)
    if kind == "noise":
        noise = np.random.normal(0, 40 * level, out.shape)
        return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        q = int(95 - 85 * level)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out
    return out
