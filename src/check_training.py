"""Training health check: progress, loss trend, overfitting signal, GPU status.

Reads the TensorBoard event files (accurate step numbers) plus the console log,
and prints a short report with explicit warnings.
"""

import glob
import os
import re
import subprocess
import sys
import time

import paths
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 用法: python check_training.py [数据集名] [总步数]
_dataset = sys.argv[1] if len(sys.argv) > 1 else paths.DEFAULT_DATASET
RUN_DIR = paths.run_dir(_dataset)
LOG = paths.log_file(_dataset)
TOTAL_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 249


def gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, free, util, temp = [x.strip() for x in out.split(",")]
        return f"{used} MiB used / {free} MiB free | util {util}% | {temp} C"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def process_status():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"
    for line in out.splitlines():
        if "train_sft.py" in line and "grep" not in line:
            parts = line.split(None, 2)
            return f"RUNNING  pid={parts[0]}  elapsed={parts[1]}"
    return "NOT RUNNING"


def load_scalars():
    files = glob.glob(os.path.join(RUN_DIR, "**", "events.out.tfevents*"), recursive=True)
    if not files:
        return None, {}
    merged = {}
    for f in files:
        acc = EventAccumulator(os.path.dirname(f))
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            merged.setdefault(tag, [])
            merged[tag].extend((e.step, e.value) for e in acc.Scalars(tag))
    for tag in merged:
        # dedupe by step (multiple event files can overlap) and sort
        merged[tag] = sorted({s: v for s, v in merged[tag]}.items())
    return files, merged


def fmt(x):
    return f"{x:.4g}"


def main():
    print("=" * 72)
    print("训练健康检查  " + time.strftime("%H:%M:%S"))
    print("=" * 72)

    print("\n[进程]      ", process_status())
    print("[GPU]       ", gpu_status())

    files, sc = load_scalars()
    if not sc:
        print("\n还没有 TensorBoard 事件文件，训练可能刚启动。")
        return

    loss = sc.get("train/loss", [])
    if loss:
        step = loss[-1][0]
        pct = step / TOTAL_STEPS * 100
        epoch = sc.get("train/epoch", [(0, 0)])[-1][1]
        # seconds per step from the console log's progress bar
        per_step = None
        if os.path.exists(LOG):
            raw = open(LOG, encoding="utf-8", errors="replace").read()
            # 进度条形如 " 12/2020 [02:11<5:32:10, 9.87s/it]"，步数上限不能写死
            m = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:]+), ([\d.]+)s/it\]", raw)
            if m:
                per_step = float(m[-1][4])
        eta = f"{per_step * (TOTAL_STEPS - step) / 60:.1f} 分钟" if per_step else "?"

        print(f"\n[进度]       step {step}/{TOTAL_STEPS} ({pct:.1f}%)  "
              f"epoch {epoch:.2f}  {per_step or '?'} s/step  剩余约 {eta}")

    print("\n[train 曲线]")
    keys = ["train/loss", "train/grad_norm", "train/learning_rate",
            "train/mean_token_accuracy", "train/entropy"]
    present = [k for k in keys if k in sc]
    print("  " + f"{'step':>6}" + "".join(f"{k.split('/')[1][:13]:>15}" for k in present))
    steps = [s for s, _ in sc.get("train/loss", [])]
    lookup = {k: dict(sc[k]) for k in present}
    for s in steps:
        row = f"  {s:>6}"
        for k in present:
            v = lookup[k].get(s)
            row += f"{fmt(v):>15}" if v is not None else f"{'-':>15}"
        print(row)

    if len(loss) >= 3:
        recent = [v for _, v in loss[-3:]]
        trend = "下降" if recent[-1] < recent[0] else "持平或上升"
        print(f"\n[loss 趋势]  最近 3 个点: "
              f"{' -> '.join(fmt(v) for v in recent)}  ({trend})")

    ev = sc.get("eval/loss", [])
    if ev:
        print("\n[eval/loss]")
        for s, v in ev:
            print(f"  step {s:>4}   loss {fmt(v)}")

        train_at = {s: v for s, v in loss}
        close = [(s, v) for s, v in train_at.items() if s in dict(ev)]
        if close:
            print("\n[过拟合检查]")
            for s, t in close:
                e = dict(ev)[s]
                gap = e - t
                verdict = "正常" if gap < 0.35 else "偏大，开始过拟合"
                print(f"  step {s:>4}  train {fmt(t)}  eval {fmt(e)}  "
                      f"差 {fmt(gap)}  -> {verdict}")
            if len(ev) >= 2:
                if ev[-1][1] > ev[-2][1]:
                    print(f"  !! eval/loss 上一次 {fmt(ev[-2][1])} -> 这次 {fmt(ev[-1][1])}，"
                          f"在回升，考虑停在这里")
                else:
                    print(f"  eval/loss 仍在下降 ({fmt(ev[-2][1])} -> {fmt(ev[-1][1])})")
    else:
        print("\n[eval/loss]   还没有评测点（每 50 步一次，第一次在第 50 步）")

    if os.path.exists(LOG):
        raw = open(LOG, encoding="utf-8", errors="replace").read()
        bad = [l for l in raw.replace("\r", "\n").splitlines()
               if re.search(r"Traceback|CUDA out of memory|Error|nan|inf", l, re.I)]
        bad = [b for b in bad if "error_message" not in b]
        print(f"\n[日志异常]    {len(bad)} 条")
        for b in bad[-5:]:
            print("   ", b[:150])


if __name__ == "__main__":
    main()
