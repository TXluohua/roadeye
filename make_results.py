# -*- coding: utf-8 -*-
"""
make_results.py —— 把已完成实验的数字固化成 assets/results/*.json

为什么要有这个脚本
------------------
演示台的原则：**重指标离线算好，页面只读 JSON**。
否则点一下就要等 10 分钟评估，面试官直接走了。

这里的数据来源全部是真实跑出来的（见 docs/experiments.md），
脚本只是把它们转成结构化 JSON，并附上来源说明。

用法：
    python make_results.py            # 生成全部
    python make_results.py --verify   # 只检查 JSON schema
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "results"

NAMES_ZH = ["纵向裂缝", "横向裂缝", "龟裂", "坑槽"]
NAMES_EN = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
CODES = ["D00", "D10", "D20", "D40"]


# --------------------------------------------------------------------------- #
def m1_baseline() -> dict:
    """E1 基线指标。per-class 来自训练结束时 ultralytics 的 val 输出。"""
    per_class = [
        {"code": "D00", "zh": "纵向裂缝", "en": "Longitudinal Crack",
         "images": 2500, "instances": 4807,
         "P": 0.589, "R": 0.415, "mAP50": 0.422, "mAP50_95": 0.215},
        {"code": "D10", "zh": "横向裂缝", "en": "Transverse Crack",
         "images": 1757, "instances": 2992,
         "P": 0.553, "R": 0.427, "mAP50": 0.413, "mAP50_95": 0.173},
        {"code": "D20", "zh": "龟裂", "en": "Alligator Crack",
         "images": 1400, "instances": 1769,
         "P": 0.627, "R": 0.600, "mAP50": 0.612, "mAP50_95": 0.325},
        {"code": "D40", "zh": "坑槽", "en": "Pothole",
         "images": 699, "instances": 1902,
         "P": 0.512, "R": 0.221, "mAP50": 0.228, "mAP50_95": 0.100},
    ]
    curve = []
    p = HERE / "runs" / "baseline" / "results.csv"
    if p.is_file():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            r = {k.strip(): v for k, v in r.items()}
            curve.append({
                "epoch": int(r["epoch"]),
                "train_box": round(float(r["train/box_loss"]), 4),
                "val_box": round(float(r["val/box_loss"]), 4),
                "mAP50": round(float(r["metrics/mAP50(B)"]), 4),
                "mAP50_95": round(float(r["metrics/mAP50-95(B)"]), 4),
            })
    return {
        "meta": {
            "title": "E1 基线训练",
            "model": "yolo11n.pt", "imgsz": 640, "epochs": 50, "batch": 16,
            "dataset": "Unified_Road_Defect_Dataset (RDD2022 合并版)",
            "train_images": 12838, "val_images": 4509,
            "train_minutes": 125,
            "conclusion": "第 40 轮起 val/box_loss 由 1.914 升到 1.943 而 train 继续降 → 已过拟合，加轮无用",
            "inference_ms": {"preprocess": 0.2, "inference": 1.6, "postprocess": 1.6},
        },
        "overall": {"P": 0.570, "R": 0.416, "mAP50": 0.418, "mAP50_95": 0.203,
                    "instances": 11470, "images": 4509},
        "per_class": per_class,
        "curve": curve,
    }


def m2_slice() -> dict:
    """E3 + E5 四组推理方案对照。"""
    return {
        "meta": {
            "title": "E2 整图 / 切片 / 融合 四组对照",
            "sample_images": 1200, "sample_seed": 42,
            "eval": "recall @ IoU>=0.5, conf>=0.25",
            "note": "B 行来自 1200 张采样；其余调参扫描来自 500 张采样（基线 39.1%）",
        },
        "buckets": ["极小<0.3%", "小0.3~1%", "中1~3%", "大>3%"],
        "modes": [
            {"key": "A", "name": "整图推理 @640（基线）",
             "recall": 39.1, "ms": 25, "boxes_per_img": 1.6,
             "buckets": {"极小<0.3%": 4.0, "小0.3~1%": 35, "中1~3%": 51, "大>3%": 64.5},
             "note": "基线"},
            {"key": "B", "name": "切片推理 320→640（放大 2×）",
             "recall": 38.5, "ms": 533, "boxes_per_img": 14.3,
             "buckets": {"极小<0.3%": 19.6, "小0.3~1%": 40, "中1~3%": 41, "大>3%": 54.5},
             "note": "小目标 +15.6，大目标 −10.0，净收益≈0（零和）"},
            {"key": "C", "name": "整图推理 @1024",
             "recall": 38.3, "ms": 28, "boxes_per_img": 2.0,
             "buckets": {"极小<0.3%": 9.4, "小0.3~1%": 38, "中1~3%": 48, "大>3%": 58.4},
             "note": "源图仅 512~720px，上采样不产生信息"},
            {"key": "D", "name": "★ 整图 + 切片 融合（tile448 ov0.25）",
             "recall": 47.1, "ms": 123, "boxes_per_img": 8.2,
             "buckets": {"极小<0.3%": 16.3, "小0.3~1%": 44, "中1~3%": 56, "大>3%": 67.6},
             "note": "+8.0 个点，代价 4.9 倍延迟"},
        ],
        "sweep": [
            {"tile": 320, "overlap": 0.50, "recall": 47.3, "ms": 1065},
            {"tile": 320, "overlap": 0.25, "recall": 46.8, "ms": 172},
            {"tile": 384, "overlap": 0.25, "recall": 46.6, "ms": 127},
            {"tile": 448, "overlap": 0.25, "recall": 47.1, "ms": 123, "best": True},
            {"tile": 384, "overlap": 0.40, "recall": 47.5, "ms": 203},
        ],
        "conclusion": [
            "单纯切片是零和博弈：小目标赚、大目标亏，净收益≈0",
            "单纯提高 imgsz 无效：源图分辨率不足，插值不产生新信息",
            "融合才让两边都赢：小目标由切片提供、大目标由整图兜底，整体 +8.0 点",
        ],
    }


def m3_scale() -> dict:
    """E2 尺寸分桶诊断（全量 4509 图）。"""
    return {
        "meta": {
            "title": "E3 尺度诊断：为什么 D40 只有 0.221",
            "images": 4509, "instances": 11471,
            "eval": "recall @ IoU>=0.5, conf>=0.25",
            "conclusion": "极小目标平均 6%，大目标平均 60%（差 10 倍）。"
                          "D40 有 47% 样本落在极小桶，而 D20 只有 1.5% —— 根因是尺寸不是类别",
        },
        "buckets": ["极小<0.3%", "小0.3~1%", "中1~3%", "大>3%"],
        "rows": [
            {"code": "D00", "zh": "纵向裂缝",
             "hit": [67, 337, 721, 774], "tot": [1071, 1126, 1381, 1229]},
            {"code": "D10", "zh": "横向裂缝",
             "hit": [38, 378, 504, 339], "tot": [534, 988, 936, 534]},
            {"code": "D20", "zh": "龟裂",
             "hit": [0, 14, 100, 902], "tot": [26, 126, 336, 1281]},
            {"code": "D40", "zh": "坑槽",
             "hit": [96, 168, 89, 50], "tot": [890, 625, 269, 119]},
        ],
        "size_stats": [
            {"code": "D00", "area_pct_median": 2.54, "long_side_px": 144, "aspect_median": 2.6},
            {"code": "D10", "area_pct_median": 2.07, "long_side_px": 178, "aspect_median": 5.1},
            {"code": "D20", "area_pct_median": 9.40, "long_side_px": 229, "aspect_median": 1.6},
            {"code": "D40", "area_pct_median": 0.57, "long_side_px": 63, "aspect_median": 1.6},
        ],
    }


def m4_quality() -> dict:
    """M4 图像质量前置的配置与阈值来源。"""
    return {
        "meta": {
            "title": "M4 图像质量前置",
            "purpose": "先判「这张图能不能用」，不能用的直接跳过，而不是硬给结果",
            "metrics": ["sharpness", "brightness", "overexposure", "contrast"],
            "method": "对清晰图注入 3 档 blur / dark / overexpose / noise / jpeg 退化，"
                      "绘制「退化强度 → 质量分」曲线，按漏检代价选定阈值",
        },
        "thresholds": {
            "sharpness_min": 60.0,
            "brightness_range": [30, 225],
            "overexposure_max": 0.05,
            "underexposure_max": 0.30,
            "contrast_min": 18.0,
            "fail_votes": 1,
        },
        "degradations": ["blur", "dark", "overexpose", "noise", "jpeg"],
        "note": "阈值已在 configs/quality.yaml，可由 calibrate 脚本重新标定",
    }


def m5_dedup() -> dict:
    """M5 连续帧去重（参数 + 实测结果）。"""
    return {
        "meta": {
            "title": "M5 连续帧去重",
            "purpose": "行车记录仪是视频流，同一个坑会被连续拍 30~50 帧；"
                       "要交出去的是「唯一点位清单」而不是「帧清单」",
            "rule": "整帧 dHash 汉明距离 + 框 IoU 双条件（只判 pHash 会把同段路上不同的坑误合并）",
            "dataset_evidence": "probe.py 在训练集里检出 134 组近重复共 368 张，"
                                "全部是 rdv（视频帧）来源",
            "benchmark": "12 帧模拟车驶过序列（源图有 11 个真实病害，"
                         "各帧汉明距离 0~6，全部在阈值 8 内）",
        },
        "params": {
            "phash_hamming_max": 8,
            "iou_min": 0.3,
            "max_gap_frames": 45,
            "same_class_only": True,
            "keep_strategy": "conf",
        },
        "measured": {
            "n_frames": 12,
            "n_raw_boxes": 147,
            "n_unique_points": 16,
            "reduction_ratio": 9.19,
            "ground_truth_points": 11,
            "stable_rule": "出现帧数 >= 总帧数的一半（12 帧里 >=6 帧）",
            "stable_points": 11,
            "note": "12 帧共 147 个原始框，归并成 16 个点位（压缩 9.19 倍）；"
                    "但 16 个里有 5 个只出现 1~5 帧，是 warpAffine 边缘拉伸产生的假阳性。"
                    "真实病害应在一半以上的帧里都可见，按「出现 >=6 帧」过滤后剩 11 个，"
                    "与源图真值 11 个完全一致。",
            "conf_sweep": [
                {"conf": 0.25, "n_raw_boxes": 147, "n_points": 16,
                 "n_ge3_frames": 13, "n_ge6_frames": 11},
                {"conf": 0.30, "n_raw_boxes": 140, "n_points": 15,
                 "n_ge3_frames": 13, "n_ge6_frames": 11},
                {"conf": 0.35, "n_raw_boxes": 127, "n_points": 14,
                 "n_ge3_frames": 12, "n_ge6_frames": 11},
                {"conf": 0.40, "n_raw_boxes": 118, "n_points": 13,
                 "n_ge3_frames": 12, "n_ge6_frames": 11},
                {"conf": 0.50, "n_raw_boxes": 95, "n_points": 12,
                 "n_ge3_frames": 9, "n_ge6_frames": 8},
            ],
            "sweep_conclusion": "「出现 >=6 帧」在 conf 0.25~0.40 全区间都恰好得到 11 个，"
                                "对置信度阈值不敏感 —— 说明这个过滤规则是稳的，"
                                "不是靠调参凑出来的；conf 提到 0.50 才开始漏掉真病害。",
        },
    }


# --------------------------------------------------------------------------- #
def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ {name:16s} {p.stat().st_size / 1024:6.1f} KB")


REQUIRED = {
    "m1_baseline.json": ["meta", "overall", "per_class"],
    "m2_slice.json": ["meta", "buckets", "modes", "sweep"],
    "m3_scale.json": ["meta", "buckets", "rows"],
    "m4_quality.json": ["meta", "thresholds"],
    "m5_dedup.json": ["meta", "params"],
}


def verify() -> int:
    ok = True
    for fn, keys in REQUIRED.items():
        p = OUT / fn
        if not p.is_file():
            print(f"  ✗ 缺文件 {fn}")
            ok = False
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        miss = [k for k in keys if k not in d]
        if miss:
            print(f"  ✗ {fn} 缺字段 {miss}")
            ok = False
        else:
            print(f"  ✓ {fn} schema OK")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        return verify()

    print("生成 assets/results/*.json …")
    write("m1_baseline.json", m1_baseline())
    write("m2_slice.json", m2_slice())
    write("m3_scale.json", m3_scale())
    write("m4_quality.json", m4_quality())
    write("m5_dedup.json", m5_dedup())
    print("\n校验 schema …")
    return verify()


if __name__ == "__main__":
    sys.exit(main())
