# -*- coding: utf-8 -*-
"""
make_samples.py —— 从验证集挑预置样例（演示台用）

原则
----
1. **样例必须全部来自验证集**，不能用训练集（否则面试官一眼看出没泛化）
2. 覆盖面要全：小目标多的、大目标多的、各主导类别的、D40 坑槽的
3. M5 模块需要一组**连续帧**（rdv 来源），单独放一个目录

产物
----
    assets/samples/m1/*.jpg     检测演示（4~6 张，覆盖不同难度）
    assets/samples/m4/*.jpg     质量演示（3 张，含一张较暗的）
    assets/samples/m5/*.jpg     连续帧（12 张同段路）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SAMP = PROJ / "assets" / "samples"


def load_val():
    base = PROJ / "data" / "raw" / "unified"
    img_dir = base / "images" / "val"
    lab_dir = base / "labels" / "val"
    imgs = sorted(img_dir.glob("*.jpg"))
    return imgs, lab_dir


def stats_of(img_p: Path, lab_dir: Path):
    lab = lab_dir / (img_p.stem + ".txt")
    rows = []
    if lab.is_file():
        for line in lab.read_text().splitlines():
            q = line.split()
            if len(q) == 5:
                rows.append((int(float(q[0])), *[float(x) for x in q[1:]]))
    areas = [r[3] * r[4] for r in rows]
    return {
        "n": len(rows),
        "cls": [r[0] for r in rows],
        "n_tiny": sum(1 for a in areas if a < 0.003),
        "max_area": max(areas) if areas else 0.0,
    }


def main() -> int:
    imgs, lab_dir = load_val()
    if not imgs:
        print("找不到验证集，先跑 tools/extract.py")
        return 1
    print(f"验证集 {len(imgs):,} 张")

    # 只看前 2000 张，控制耗时
    cand = imgs[:2000]
    info = [(p, stats_of(p, lab_dir)) for p in cand]
    info = [(p, s) for p, s in info if s["n"] > 0]
    print(f"有标注的 {len(info)} 张")

    picks = {}

    # ① 小目标最多的一张（演示整图漏检 vs 融合检出）
    picks["m1_small"] = max(info, key=lambda x: x[1]["n_tiny"])[0]
    # ② 目标最多的一张
    picks["m1_many"] = max(info, key=lambda x: x[1]["n"])[0]
    # ③ 以 D40 坑槽为主的一张
    d40 = [x for x in info if x[1]["cls"].count(3) >= max(1, x[1]["n"] // 2)]
    if d40:
        picks["m1_d40"] = d40[len(d40) // 2][0]
    # ④ 以 D20 龟裂为主的一张
    d20 = [x for x in info if x[1]["cls"].count(2) >= max(1, x[1]["n"] // 2)]
    if d20:
        picks["m1_d20"] = d20[len(d20) // 2][0]
    # ⑤ 大目标一张（对比：融合对大小目标都有效）
    picks["m1_large"] = max(info, key=lambda x: x[1]["max_area"])[0]

    for sub in ("m1", "m4", "m5"):
        (SAMP / sub).mkdir(parents=True, exist_ok=True)

    n = 0
    for tag, p in picks.items():
        dst = SAMP / "m1" / f"{tag}_{p.name}"
        shutil.copy(p, dst)
        # m4 共用 m1 的样例
        shutil.copy(p, SAMP / "m4" / f"{tag}_{p.name}")
        n += 1
        print(f"  m1/{tag:10s} <- {p.name}  框={stats_of(p, lab_dir)['n']}")

    # ⑥ M5：同一段路的连续帧（rdv_* 是视频帧，天然连续）
    rdv = [p for p in imgs if p.name.startswith("rdv_")]
    if not rdv:
        rdv = [p for p in imgs if any(k in p.name for k in ("Drone", "MotorBike"))]
    # 找一段编号连续的
    def num_of(p):
        parts = p.stem.split("_")
        for x in reversed(parts):
            if x.isdigit() and len(x) >= 4:
                return int(x)
        return -1
    rdv = sorted(rdv, key=num_of)
    seq = rdv[:12] if len(rdv) >= 12 else rdv
    for i, p in enumerate(seq):
        shutil.copy(p, SAMP / "m5" / f"frame{i:02d}_{p.name}")
    print(f"  m5/ 连续帧 {len(seq)} 张（rdv 来源）")

    print(f"\n样例已生成：{SAMP}")
    for sub in ("m1", "m4", "m5"):
        c = len(list((SAMP / sub).glob("*.jpg")))
        print(f"  {sub}/  {c} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
