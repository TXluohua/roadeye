# -*- coding: utf-8 -*-
"""
app.py —— 演示台后端（FastAPI 单文件）

设计原则
--------
**重指标离线预计算，页面只读 JSON**。所以点开就是秒开，不会出现
"点了之后等 10 分钟评估"的情况。单图推理是现场跑的（1~120 ms）。

接口一览
--------
    GET  /                          单页前端
    GET  /api/meta                  模型 / 数据 / 各实验的元信息
    GET  /api/samples               预置样例列表（按模块分组）
    GET  /api/sample?path=xxx       取样例图
    GET  /api/m/{m1..m5}            各模块的预计算指标（读 assets/results/*.json）
    POST /api/detect                单图检测（mode=whole|slice|fusion）
    POST /api/compare               同一张图跑多种模式，返回对比
    POST /api/quality               图像质量评估
    POST /api/quality/degrade       生成退化图 + 评估（M4 演示）
    POST /api/dedup                 一组帧 → 连续帧去重（M5）

启动
----
    python run.py            # 推荐：自动开浏览器
    uvicorn app:app --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS = HERE / "assets" / "results"
SAMPLES = HERE / "assets" / "samples"
WEIGHTS_DEFAULT = HERE / "runs" / "baseline" / "weights" / "best.pt"

app = FastAPI(title="RoadEye 演示台")

_cache: dict[str, tuple[float, dict]] = {}


# --------------------------------------------------------------------------- #
def load_yaml(rel: str) -> dict:
    p = HERE / rel
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def read_result(name: str) -> dict:
    """读 assets/results/<name>.json（按 mtime 缓存，重跑 make_results.py 后无需重启）。"""
    p = RESULTS / f"{name}.json"
    if not p.is_file():
        raise HTTPException(404, f"{p.name} 还没生成，先跑 python make_results.py")
    mtime = p.stat().st_mtime
    hit = _cache.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    d = json.loads(p.read_text(encoding="utf-8"))
    _cache[name] = (mtime, d)
    return d


def b64_of(img_bgr) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(500, "编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def decode_upload(data: bytes):
    import cv2
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "不是有效图片")
    return img


def resolve_sample(path: str) -> Path:
    p = (SAMPLES / path).resolve()
    if not str(p).startswith(str(SAMPLES.resolve())) or not p.is_file():
        raise HTTPException(404, "样例不存在")
    return p


# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    p = HERE / "index.html"
    if not p.is_file():
        return HTMLResponse("<h1>缺 index.html</h1>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/meta")
def api_meta():
    cfg = load_yaml("configs/split.yaml")
    w = WEIGHTS_DEFAULT
    out = {
        "weights": str(w.relative_to(HERE)) if w.is_file() else None,
        "weights_mb": round(w.stat().st_size / 1e6, 2) if w.is_file() else None,
        "classes": [{"id": i, "code": c, "zh": cfg.get("zh", {}).get(i, ""),
                     "en": cfg.get("names", {}).get(i, "")}
                    for i, c in enumerate(cfg.get("short", {}).values() or ["D00", "D10", "D20", "D40"])],
        "inference": {"whole_ms": 25, "slice_ms": 533, "fusion_ms": 123},
        "dataset": {"train": 12838, "val": 4509, "instances": 37614},
    }
    try:
        out["experiments"] = {k: read_result(k)["meta"].get("title", k)
                              for k in ("m1_baseline", "m2_slice", "m3_scale",
                                        "m4_quality", "m5_dedup")}
    except HTTPException:
        out["experiments"] = {}
    return JSONResponse(out)


@app.get("/api/samples")
def api_samples():
    """列出 assets/samples 下的样例，按子目录分组。"""
    groups: dict[str, list[str]] = {}
    if SAMPLES.is_dir():
        for sub in sorted(SAMPLES.iterdir()):
            if sub.is_dir():
                files = [f"{sub.name}/{p.name}" for p in sorted(sub.glob("*"))
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                if files:
                    groups[sub.name] = files
        root_files = [p.name for p in sorted(SAMPLES.glob("*"))
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if root_files:
            groups["_root"] = root_files
    return JSONResponse(groups)


@app.get("/api/sample")
def api_sample(path: str):
    p = resolve_sample(path)
    return Response(p.read_bytes(), media_type="image/jpeg")


# ---------------- 各模块的预计算指标 ----------------
@app.get("/api/m/m1")
def m1():
    return JSONResponse(read_result("m1_baseline"))


@app.get("/api/m/m2")
def m2():
    return JSONResponse(read_result("m2_slice"))


@app.get("/api/m/m3")
def m3():
    return JSONResponse(read_result("m3_scale"))


@app.get("/api/m/m4")
def m4():
    d = read_result("m4_quality")
    d["thresholds_live"] = load_yaml("configs/quality.yaml")
    return JSONResponse(d)


@app.get("/api/m/m5")
def m5():
    d = read_result("m5_dedup")
    d["params_live"] = load_yaml("configs/dedup.yaml")
    return JSONResponse(d)


# ---------------- 单图检测 ----------------
class DetectReq(BaseModel):
    sample: str | None = None
    image_b64: str | None = None
    mode: str = "fusion"
    conf: float = 0.25
    tile: int = 448
    overlap: float = 0.25


def _get_image(req: DetectReq):
    if req.image_b64:
        raw = req.image_b64.split(",")[-1]
        return decode_upload(base64.b64decode(raw))
    if req.sample:
        import cv2
        p = resolve_sample(req.sample)
        img = cv2.imread(str(p))
        if img is None:
            raise HTTPException(400, "样例读取失败")
        return img
    raise HTTPException(400, "需要 sample 或 image_b64")


@app.post("/api/detect")
def api_detect(req: DetectReq):
    from core import detect as D
    if not WEIGHTS_DEFAULT.is_file():
        raise HTTPException(404, "找不到权重，先训练：python train.py")
    img = _get_image(req)
    r = D.infer(img, str(WEIGHTS_DEFAULT), mode=req.mode, conf=req.conf,
                tile=req.tile, overlap=req.overlap)
    vis = D.draw(img, r["dets"])
    r["image_b64"] = b64_of(vis)
    r["counts"] = {D.CLASS_SHORT[i]: sum(1 for d in r["dets"] if d["cls"] == i)
                   for i in range(4)}
    return JSONResponse(r)


@app.post("/api/compare")
def api_compare(req: DetectReq):
    """同一张图跑整图 vs 融合，返回并排结果 —— 这是 M2 模块的现场演示。"""
    from core import detect as D
    if not WEIGHTS_DEFAULT.is_file():
        raise HTTPException(404, "找不到权重")
    img = _get_image(req)
    out = {}
    for mode in ("whole", "fusion"):
        r = D.infer(img, str(WEIGHTS_DEFAULT), mode=mode, conf=req.conf,
                    tile=req.tile, overlap=req.overlap)
        r["image_b64"] = b64_of(D.draw(img, r["dets"]))
        r["counts"] = {D.CLASS_SHORT[i]: sum(1 for d in r["dets"] if d["cls"] == i)
                       for i in range(4)}
        out[mode] = r
    return JSONResponse(out)


# ---------------- 图像质量 ----------------
class QualityReq(BaseModel):
    sample: str | None = None
    image_b64: str | None = None
    degrade: str | None = None       # blur/dark/overexpose/noise/jpeg
    level: float = 0.5


@app.post("/api/quality")
def api_quality(req: QualityReq):
    from core import quality as Q
    img = _get_image(DetectReq(sample=req.sample, image_b64=req.image_b64))
    cfg = load_yaml("configs/quality.yaml")
    base = Q.assess(img, cfg)
    out = {"original": base, "image_b64": b64_of(img)}
    if req.degrade:
        deg = Q.inject_degradation(img, req.degrade, req.level)
        out["degraded"] = Q.assess(deg, cfg)
        out["degraded_image_b64"] = b64_of(deg)
        out["degrade"] = req.degrade
        out["level"] = req.level
    return JSONResponse(out)


# ---------------- 连续帧去重 ----------------
class DedupReq(BaseModel):
    samples: list[str]
    conf: float = 0.25


@app.post("/api/dedup")
def api_dedup(req: DedupReq):
    from core import detect as D
    from core import dedup as Dd
    if not req.samples:
        raise HTTPException(400, "需要一组帧")
    import cv2
    imgs, dets, names = [], [], []
    for s in req.samples[:60]:
        p = resolve_sample(s)
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(img)
        names.append(s)
        if WEIGHTS_DEFAULT.is_file():
            r = D.infer(img, str(WEIGHTS_DEFAULT), mode="whole", conf=req.conf)
            dets.append(r["dets"])
        else:
            dets.append([])
    cfg = load_yaml("configs/dedup.yaml")
    res = Dd.group_frames(imgs, dets, cfg)
    res["frames"] = names
    # 把类别 index 翻译成名字：绝不能在前端用 cls*10 推算——D40 的 index 是 3
    sp = load_yaml("configs/split.yaml")
    short, zh = sp.get("short", {}), sp.get("zh", {})
    for g in res["groups"]:
        c = g["class"]
        g["cls_short"] = short.get(c, f"cls{c}")
        g["cls_zh"] = zh.get(c, "")
    return JSONResponse(res)


@app.get("/api/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
