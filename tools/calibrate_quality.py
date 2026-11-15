# -*- coding: utf-8 -*-
"""
calibrate_quality.py —— 图像质量阈值标定（补 E8 的前置工作）

干什么
------
configs/quality.yaml 里的 5 个阈值原来是拍脑袋写的。这个脚本用验证集
真实图像的分布把它们标定出来，并回答一个关键问题：

    "当前的阈值到底过滤掉了多少张图？"

如果几乎过滤不掉任何一张，说明阈值过松（等于没做前置）；
如果过滤掉太多，说明阈值过严（会丢掉可用图）。

用法
----
    python tools/calibrate_quality.py --sample 2000
    python tools/calibrate_quality.py --sample 2000 --out assets/results/quality_calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VAL = PROJ / "data" / "raw" / "unified" / "images" / "val"

sys.path.insert(0, str(PROJ))
from core.quality import assess  # noqa: E402

METRICS = ("sharpness", "brightness", "overexposure", "underexposure", "contrast")
PCTS = (1, 5, 10, 25, 50, 75, 90, 99)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000, help="采样张数（0=全量）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = PROJ / "configs" / "quality.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    imgs = sorted(VAL.rglob("*.jpg"))
    if not imgs:
        print(f"找不到验证图：{VAL}")
        return 1
    if args.sample and args.sample < len(imgs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(imgs), args.sample, replace=False)
        imgs = [imgs[i] for i in sorted(idx)]

    print(f"标定样本：{len(imgs)} 张验证图（全量 {len(sorted(VAL.rglob('*.jpg')))} 张）")

    vals = {m: [] for m in METRICS}
    n_fail = {m: 0 for m in METRICS}
    n_unusable = 0
    reasons: dict[str, int] = {}

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = assess(img, cfg)
        for m in METRICS:
            vals[m].append(r[m])
        for f in r["fails"]:
            key = f.split("（")[0]
            n_fail[key] = n_fail.get(key, 0) + 1
            reasons[key] = reasons.get(key, 0) + 1
        if not r["usable"]:
            n_unusable += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(imgs)} …")

    n = len(vals["sharpness"])
    print(f"\n实际评估 {n} 张\n")

    print(f"{'指标':<14}{'P1':>10}{'P5':>10}{'P10':>10}{'P50':>10}{'P90':>10}{'P99':>10}{'当前阈值':>12}")
    print("-" * 86)
    for m in METRICS:
        a = np.array(vals[m])
        q = {p: float(np.percentile(a, p)) for p in PCTS}
        cur = {
            "sharpness": cfg.get("sharpness_min"),
            "brightness": cfg.get("brightness_range"),
            "overexposure": cfg.get("overexposure_max"),
            "underexposure": cfg.get("underexposure_max"),
            "contrast": cfg.get("contrast_min"),
        }[m]
        cur_s = str(cur)
        print(f"{m:<14}{q[1]:>10.1f}{q[5]:>10.1f}{q[10]:>10.1f}"
              f"{q[50]:>10.1f}{q[90]:>10.1f}{q[99]:>10.1f}{cur_s:>12}")

    print(f"\n按当前阈值判「不可用」：{n_unusable}/{n} = {n_unusable / n * 100:.2f}%")
    if reasons:
        print("各项不达标次数：")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {k:<10} {v:>6} ({v / n * 100:.2f}%)")
    else:
        print("★ 没有任何一张被任何一项判为不达标 —— 阈值过松，等于没做前置。")

    # 建议阈值：用 P5 / P99 这类分位点，而不是拍脑袋
    sug = {
        "sharpness_min": round(float(np.percentile(vals["sharpness"], 5)), 1),
        "brightness_range": [round(float(np.percentile(vals["brightness"], 1)), 1),
                             round(float(np.percentile(vals["brightness"], 99)), 1)],
        "overexposure_max": round(float(np.percentile(vals["overexposure"], 99)), 4),
        "underexposure_max": round(float(np.percentile(vals["underexposure"], 99)), 4),
        "contrast_min": round(float(np.percentile(vals["contrast"], 5)), 1),
    }
    print("\n若按「P5 为下限 / P99 为上限」重新标定，建议阈值：")
    for k, v in sug.items():
        print(f"  {k} = {v}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = PROJ / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_images": n,
            "seed": args.seed,
            "current": cfg,
            "percentiles": {m: {str(p): float(np.percentile(vals[m], p)) for p in PCTS}
                            for m in METRICS},
            "n_unusable": n_unusable,
            "unusable_ratio": round(n_unusable / n, 6),
            "fail_counts": reasons,
            "suggested": sug,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
