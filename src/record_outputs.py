"""记录不同 adapter 在固定评测集上的生成输出与量化指标。

用途：SPIN 前后对比、以及实验矩阵里各组方法的横向对比。

设计要点：
  - 评测集固定（从三个测试集按固定步长抽取），保证任何适配器看到的输入完全相同
  - 输出按 tag 存成 json，最后能汇总成对照表
  - 同时算定量指标（目标 loss / 保持 loss / 中文字符占比 / 输出长度）

用法：
    python record_outputs.py --adapter runs/refinement/final --tag baseline
    python record_outputs.py --compare          # 汇总成 markdown
    python record_outputs.py --list             # 看已记录了哪些
"""

import argparse
import json
import os
import re
import sys
import time

import paths
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spin_train import SYSTEM, batch_logprob  # noqa: E402

REC_DIR = os.path.join(paths.RUNS_DIR, "spin", "records")
CJK = re.compile(r"[\u4e00-\u9fff]")
ROLE = re.compile(r"\b(?:assume the role|imagine you are|acting as|you are a|"
                  r"take on the role|as a (?:seasoned|skilled|professional|senior))\b", re.I)
CONSTRAINT = re.compile(r"\b(?:including|at least|no more than|limit(?:ed)? to|word|words|"
                        r"ensure|must|should|avoid)\b", re.I)
FORMAT = re.compile(r"\b(?:format|bullet|list|table|step[- ]by[- ]step|numbered|outline|sections?)\b", re.I)
REFUSAL = re.compile(r"\b(?:i\s+cannot|i\s+can'?t|i\s+am\s+unable|as\s+an\s+ai\b)\b", re.I)


def build_eval_set():
    """固定评测集：目标能力 10 条 / 保持能力 5 条 / 中文 5 条。"""
    items = []

    def sample(dataset, split, n, kind):
        p = os.path.join(paths.data_dir(dataset), f"{split}.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        step = max(len(rows) // n, 1)
        for i, r in enumerate(rows[::step][:n]):
            m = r["messages"]
            items.append({"kind": kind, "id": f"{kind}-{i}", "brief": m[1]["content"],
                          "reference": m[2]["content"]})

    sample("mepo", "test", 10, "target")
    sample("refinement", "test", 5, "retain")

    # 中文非程序员简述，取自手工种子
    from chinese_seed import PAIRS

    step = max(len(PAIRS) // 5, 1)
    for i, (b, ref) in enumerate(PAIRS[::step][:5]):
        items.append({"kind": "zh", "id": f"zh-{i}", "brief": b, "reference": ref})
    return items


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(paths.find_model())
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        paths.find_model(),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map="auto",
    )
    model = base if adapter in (None, "", "base") else PeftModel.from_pretrained(base, adapter)
    model.eval()
    return tok, model


def metrics(model, tok, max_len=384, n=80):
    """目标/保持两套 loss + 生成侧指标。"""
    out = {}

    # 生成侧：用固定评测集
    eval_set = build_eval_set()
    gens = {}
    for item in eval_set:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["brief"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=128, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        gens[item["id"]] = tok.decode(g[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
        del g, ids
    return gens


def holdout_loss(model, tok, dataset, max_len=384, n=80):
    rows = []
    for split in ("test", "val"):
        p = os.path.join(paths.data_dir(dataset), f"{split}.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l)["messages"] for l in open(p, encoding="utf-8")]
    step = max(len(rows) // n, 1)
    rows = rows[::step][:n]
    vals = []
    for m in rows:
        with torch.no_grad():
            lp = batch_logprob(model, tok, [m[1]["content"]], [m[2]["content"]], max_len)
        ntok = len(tok(m[2]["content"], add_special_tokens=False)["input_ids"]) + 1
        vals.append(-lp.item() / max(ntok, 1))
        del lp
    return sum(vals) / len(vals)


def cmd_record(args):
    os.makedirs(REC_DIR, exist_ok=True)
    print(f"[record] adapter = {args.adapter}  tag = {args.tag}")
    tok, model = load_model(args.adapter)

    print("[record] 计算目标集 / 保持集 loss ...", flush=True)
    target_loss = holdout_loss(model, tok, "mepo", n=args.n)
    retain_loss = holdout_loss(model, tok, "refinement", n=args.n)

    print("[record] 生成固定评测集输出 ...", flush=True)
    t0 = time.time()
    gens = metrics(model, tok)

    rec = {
        "tag": args.tag,
        "adapter": args.adapter,
        "target_loss": round(target_loss, 5),
        "retain_loss": round(retain_loss, 5),
        "generations": gens,
        "meta": {"n_loss": args.n, "gen_seconds": round(time.time() - t0, 1)},
    }
    path = os.path.join(REC_DIR, f"{args.tag}.json")
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[record] 写入 {path}")
    print(f"         target_loss={target_loss:.4f}  retain_loss={retain_loss:.4f}")
    return 0


def cmd_compare(args):
    files = sorted(f for f in os.listdir(REC_DIR) if f.endswith(".json")) if os.path.isdir(REC_DIR) else []
    if not files:
        print("还没有记录，先跑 --adapter")
        return 1

    recs = [json.load(open(os.path.join(REC_DIR, f), encoding="utf-8")) for f in files]
    base = next((r for r in recs if r["tag"] == args.baseline), recs[0])
    eval_set = build_eval_set()

    # ---- 汇总指标 ----
    lines = ["# SPIN 前后生成对比", "",
             f"基线 tag = `{base['tag']}`；评测集固定 {len(eval_set)} 条"
             "（目标 10 / 保持 5 / 中文 5）。", "",
             "## 量化指标", "",
             "| tag | target_loss ↓ | retain_loss ↓ | Δretain vs 基线 | 中文输出占比 | 平均输出长度 |",
             "|---|---|---|---|---|---|"]
    for r in recs:
        gens = r["generations"]
        zh_items = [g for k, g in gens.items() if k.startswith("zh-")]
        zh_ratio = (sum(1 for g in zh_items if len(CJK.findall(g)) / max(len(g), 1) > 0.5)
                    / max(len(zh_items), 1))
        avg_len = sum(len(g) for g in gens.values()) / max(len(gens), 1)
        d = r["retain_loss"] - base["retain_loss"]
        lines.append(f"| {r['tag']} | {r['target_loss']:.4f} | {r['retain_loss']:.4f} | "
                     f"{d:+.4f} | {zh_ratio * 100:.0f}% | {avg_len:.0f} |")

    # ---- 逐条对比 ----
    lines += ["", "## 逐条输出对比", ""]
    for item in eval_set:
        lines += [f"### [{item['kind']}] {item['brief']}", "",
                  f"*参考答案*：{item['reference'][:200]}", ""]
        for r in recs:
            g = r["generations"].get(item["id"], "（未记录）")
            lines += [f"**{r['tag']}**（{len(g)} 字符）", "", "> " + g.replace("\n", "\n> "), ""]
        lines += ["---", ""]

    out = os.path.join(paths.RUNS_DIR, "spin", "对照结果.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[: 16]))
    print(f"\n完整对比写入 {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=80, help="loss 评测抽样条数")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        if os.path.isdir(REC_DIR):
            for f in sorted(os.listdir(REC_DIR)):
                rec = json.load(open(os.path.join(REC_DIR, f), encoding="utf-8"))
                print(f"  {rec['tag']:<20} target={rec['target_loss']:.4f} "
                      f"retain={rec['retain_loss']:.4f}  ({rec['adapter']})")
        return 0
    if args.compare:
        return cmd_compare(args)
    if not args.adapter or not args.tag:
        ap.error("需要 --adapter 和 --tag")
    return cmd_record(args)


if __name__ == "__main__":
    sys.exit(main())
