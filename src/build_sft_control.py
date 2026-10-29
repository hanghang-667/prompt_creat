"""把 SPIN 的 pairs 文件转成 SFT 对照组用的训练集。

对照组必须用与 SPIN 完全相同的 (prompt, real) 样本，否则对比不成立。
生成阶段产出的 pairs_iter{t}.jsonl 里已经含 prompt 和 real，直接取用。

输出：data/spin_sft/{train,val,test}.jsonl（messages 格式，供 train_sft.py 使用）
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
from spin_train import SYSTEM


def main():
    iter_idx = sys.argv[1] if len(sys.argv) > 1 else "0"
    # 默认用全部样本训练，保证与 SPIN 的数据量对等
    # （v1 版切了 160/20/20，导致 SFT 只训 160 条而 SPIN 用满 200 条，对比不公平）
    pairs = os.path.join(paths.RUNS_DIR, "spin", f"pairs_iter{iter_idx}.jsonl")
    rows = [json.loads(l) for l in open(pairs, encoding="utf-8")]

    records = [{"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["real"]},
    ]} for r in rows]

    random.seed(42)
    random.shuffle(records)
    n = len(records)
    # 训练集 = 全部样本（与 SPIN 对等）；val/test 只用于监控，不参与训练
    n_val = max(int(n * 0.05), 8)
    n_test = max(int(n * 0.05), 8)

    out = paths.data_dir("spin_sft")
    os.makedirs(out, exist_ok=True)
    for name, data in (("train", records),                 # ← 全部用于训练
                       ("val", records[:n_val]),
                       ("test", records[n_val:n_val + n_test])):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in data:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"对照组数据: {out}  train {len(records)} / "
          f"val {n_val} / test {n_test}  （源自 {os.path.basename(pairs)}）")


if __name__ == "__main__":
    main()
