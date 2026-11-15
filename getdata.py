# -*- coding: utf-8 -*-
"""
getdata.py —— 可靠下载器：下载 → 校验 gzip → 坏则重下

关键设计：
  1. **不做断点续传**（本项目实测：被打断后续传会写错位导致 gzip 损坏）
  2. 下载完立刻用 tarfile 校验整包
  3. 校验失败就删掉重下，最多 4 轮
  4. 下载中不要 kill 它（kill 会留下写了一半的块）

用法：
    python getdata.py val           # 只下 val
    python getdata.py train_a       # 只下 train_a
    python getdata.py all           # 两个都下
    python getdata.py verify        # 只校验已有文件
"""
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

HERE = Path(__file__).resolve().parent
BASE = "https://hf-mirror.com/datasets/TamAko783/Unified_Road_Defect_Dataset/resolve/main/data"
# 下载缓存目录：跟着本文件走，不写死绝对路径（换机器/换盘不用改）
DL = str(HERE / "data" / "_dl")
# 上游发布时的字节数，用于校验下载完整。若上游更新了数据集，这几个数会失配，
# 届时按实际大小改这里（脚本会把"大小对但 gzip 坏"和"大小不符"分开报）。
SIZES = {
    "train_a.tar.gz": 928438317,
    "val.tar.gz": 1067597682,
    "train_b.tar.gz": 5141618844,
}


def path_of(name):
    return os.path.join(DL, name)


def size(name):
    p = path_of(name)
    return os.path.getsize(p) if os.path.exists(p) else 0


def verify(name, quick=True):
    """整包校验：能完整遍历 tar 成员就算好。"""
    p = path_of(name)
    if not os.path.exists(p):
        return False, "文件不存在"
    try:
        with tarfile.open(p, "r:gz") as tf:
            n_img = n_txt = total = 0
            for m in tf:
                total += 1
                if m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    n_img += 1
                elif m.name.lower().endswith(".txt"):
                    n_txt += 1
                if quick and total >= 200000:
                    break
        return True, f"OK  成员 {total}  图片 {n_img}  标签 {n_txt}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def download(name, rounds=4):
    p = path_of(name)
    target = SIZES[name]
    os.makedirs(DL, exist_ok=True)
    for rd in range(1, rounds + 1):
        if os.path.exists(p):
            s = size(name)
            if s == target:
                ok, msg = verify(name)
                if ok:
                    print(f"[{name}] 已有且完整：{msg}", flush=True)
                    return True
                print(f"[{name}] 大小对但校验失败（{msg}），删除重下", flush=True)
            else:
                print(f"[{name}] 不完整 {s}/{target}，删除重下（不做续传）", flush=True)
            os.remove(p)

        print(f"[{name}] 第 {rd}/{rounds} 轮下载开始  {time.strftime('%H:%M:%S')}"
              f"  目标 {target/1e6:.0f} MB", flush=True)
        t0 = time.time()
        r = subprocess.run(
            ["curl.exe", "-L", "--ssl-no-revoke", "--noproxy", "*",
             "--max-time", "3600", "--connect-timeout", "30",
             "--retry", "5", "--retry-delay", "3", "--retry-all-errors",
             "--no-progress-meter", "-o", p, f"{BASE}/{name}"],
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        s = size(name)
        print(f"[{name}] 下载结束 {s/1e6:.1f}/{target/1e6:.0f} MB  "
              f"{s/target*100:.1f}%  {dt/60:.1f} 分钟  "
              f"{s/1e6/max(1,dt):.2f} MB/s  rc={r.returncode}", flush=True)
        if s != target:
            print(f"[{name}] 大小不符，重试", flush=True)
            continue
        ok, msg = verify(name)
        print(f"[{name}] 校验：{msg}", flush=True)
        if ok:
            return True
    return False


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if what == "verify":
        for n in ("train_a.tar.gz", "val.tar.gz"):
            if os.path.exists(path_of(n)):
                s = size(n)
                ok, msg = verify(n)
                flag = "✅" if ok else "❌"
                print(f"{flag} {n:18s} {s:>13,} 字节  {msg}", flush=True)
            else:
                print(f"— {n:18s} 未下载", flush=True)
        return 0

    names = ["val.tar.gz", "train_a.tar.gz"] if what == "all" else [what if what.endswith(".tar.gz") else what + ".tar.gz"]
    allok = True
    for n in names:
        if n not in SIZES:
            print(f"未知文件 {n}", flush=True)
            return 1
        if not download(n):
            allok = False
    print("\n=== 汇总 ===", flush=True)
    for n in SIZES:
        if os.path.exists(path_of(n)):
            print(f"  {n:18s} {size(n)/1e6:8.1f} MB", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
