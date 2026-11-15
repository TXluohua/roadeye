# -*- coding: utf-8 -*-
"""
make_m5_samples.py —— 为 M5（连续帧去重）生成一组"连续帧"

两种模式
--------
    python tools/make_m5_samples.py             # 默认：合成连续帧（推荐）
    python tools/make_m5_samples.py --real      # 从数据集里找真实近重复簇

为什么默认用合成
----------------
数据集里能直接找到的"连续帧"其实不连续（rdv 的帧号是稀疏的：268, 274, 281...），
差分哈希距离 10~20，超出 dedup 阈值，演示效果差。

合成的好处是**真值已知**：从一张真实图出发，模拟车开过去时的
微小位移 + 轻微缩放 + 光照抖动，生成 N 帧。**每帧里都是同一批病害**，
所以理想情况下去重后点位数 = 源图里的病害数。
这样就可以直接验证去重算法的准确率，而不是"看着像就行"。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SAMP = PROJ / "assets" / "samples" / "m5"
VAL = PROJ / "data" / "raw" / "unified" / "images" / "val"
LAB = PROJ / "data" / "raw" / "unified" / "labels" / "val"


def dhash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def count_labels(p: Path) -> int:
    lab = LAB / (p.stem + ".txt")
    if not lab.is_file():
        return 0
    return sum(1 for line in lab.read_text().splitlines() if len(line.split()) == 5)


def synth(n_frames: int = 12) -> int:
    """找一张病害较多的真实图，生成一组模拟"车驶过"的连续帧。"""
    cands = []
    for p in sorted(VAL.glob("*.jpg"))[:1500]:
        c = count_labels(p)
        if c >= 4:
            cands.append((c, p))
    if not cands:
        print("找不到病害 >=4 的验证图")
        return 1
    cands.sort(key=lambda x: -x[0])
    n_src, src = cands[0]
    img = cv2.imread(str(src))
    if img is None:
        return 1
    h, w = img.shape[:2]
    print(f"源图 {src.name}   真实病害 {n_src} 个   尺寸 {w}×{h}")

    SAMP.mkdir(parents=True, exist_ok=True)
    for old in SAMP.glob("*.jpg"):
        old.unlink()

    rng = np.random.default_rng(42)
    # 模拟车辆前进：每帧向右上微弱平移 + 轻微放大 + 亮度抖动
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        # 平移（像素）：沿一个方向累计约 12 px
        dx = int(round(12 * t))
        dy = int(round(-6 * t))
        # 轻微缩放 1.00 → 1.04
        scale = 1.0 + 0.04 * t
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # 亮度/噪声抖动
        alpha = 1.0 + rng.uniform(-0.06, 0.06)
        beta = rng.uniform(-8, 8)
        frame = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        noise = rng.normal(0, 3.0, frame.shape).astype(np.float32)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(SAMP / f"frame{i:02d}.jpg"), frame)

    print(f"已生成 {n_frames} 帧 → {SAMP}")
    print(f"★ 真值：这 {n_frames} 帧里是**同一批 {n_src} 个病害**，"
          f"理想情况下去重后应得到 {n_src} 个唯一点位")

    hs = [dhash(cv2.imread(str(SAMP / f"frame{i:02d}.jpg"))) for i in range(n_frames)]
    ds = [hamming(hs[0], x) for x in hs]
    print(f"与首帧的汉明距离: {ds}   最大 {max(ds)}（dedup 阈值 8）")
    return 0


def real() -> int:
    """从数据集里找真实的近重复簇（帧号可能稀疏，效果次之）。"""
    imgs = sorted(VAL.glob("rdv_*.jpg")) or sorted(VAL.glob("*Drone*.jpg"))
    if not imgs:
        print("找不到 rdv/Drone 帧")
        return 1
    print(f"候选 {len(imgs)} 帧，计算差分哈希…")
    hashes = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is not None:
            hashes.append((p, dhash(im)))
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, min(n, i + 300)):
            if hamming(hashes[i][1], hashes[j][1]) <= 8:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    best = max(groups.values(), key=len)
    seq = sorted(best)[:12]
    SAMP.mkdir(parents=True, exist_ok=True)
    for old in SAMP.glob("*.jpg"):
        old.unlink()
    for i, idx in enumerate(seq):
        shutil.copy(hashes[idx][0], SAMP / f"frame{i:02d}_{hashes[idx][0].name}")
    sd = hashes[seq[0]][1]
    print(f"采用 {len(seq)} 帧，与首帧距离 {[hamming(sd, hashes[k][1]) for k in seq]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用数据集里的真实近重复簇")
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    return real() if args.real else synth(args.frames)


if __name__ == "__main__":
    sys.exit(main())
