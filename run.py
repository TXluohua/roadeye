# -*- coding: utf-8 -*-
"""
run.py —— 一键启动演示台

做三件事：
  1. 检查依赖（ultralytics / fastapi / uvicorn / opencv）
  2. 检查 assets/results/*.json 是否齐全，缺了自动生成
  3. 起服务并自动打开浏览器

    python run.py
    python run.py --port 8000 --no-browser
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REQUIRED = ["fastapi", "uvicorn", "ultralytics", "cv2", "yaml", "numpy", "PIL"]


def check_deps() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError:
            print(f"  ✗ 缺依赖：{name}")
            ok = False
    if not ok:
        print("\n装一下：pip install fastapi uvicorn ultralytics opencv-python pyyaml pillow numpy")
    return ok


def ensure_results() -> bool:
    results = HERE / "assets" / "results"
    need = ["m1_baseline", "m2_slice", "m3_scale", "m4_quality", "m5_dedup"]
    missing = [n for n in need if not (results / f"{n}.json").is_file()]
    if not missing:
        print(f"  ✓ 指标 JSON 齐全（{len(need)} 个）")
        return True
    print(f"  · 缺 {len(missing)} 个指标 JSON，自动生成…")
    r = subprocess.run([sys.executable, str(HERE / "make_results.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode == 0


def ensure_samples() -> None:
    samp = HERE / "assets" / "samples"
    if samp.is_dir() and any(samp.rglob("*.jpg")):
        n = len(list(samp.rglob("*.jpg")))
        print(f"  ✓ 预置样例 {n} 张")
        return
    print("  · 还没有预置样例，尝试从验证集生成…")
    r = subprocess.run([sys.executable, str(HERE / "tools" / "make_samples.py")],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-800:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print("  RoadEye 演示台")
    print("=" * 62)
    print("[1/3] 检查依赖")
    if not check_deps():
        return 1
    print("[2/3] 检查数据与指标")
    ensure_results()
    ensure_samples()
    w = HERE / "runs" / "baseline" / "weights" / "best.pt"
    if w.is_file():
        print(f"  ✓ 权重 {w.name}（{w.stat().st_size / 1e6:.1f} MB）")
    else:
        print("  ⚠ 没有训练好的权重，M1/M2/M5 的现场推理会失败（先跑 python train.py）")

    url = f"http://{args.host}:{args.port}"
    print(f"[3/3] 启动 {url}")
    print()
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)),
                         daemon=True).start()
    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
