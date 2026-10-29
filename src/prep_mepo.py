"""MePO 数据清洗：产出文本任务提示词的 SFT 训练集，以及后续 DPO 用的偏好对。

体检结论（决定了下面每条规则）：

1. **输入用 `raw_prompt`，不用 `rejected`。**
   40,151 行里有 4,178 行两者不同，且差异方向一致：`rejected` 是故意写差的版本
   （"Brain some ways to get people interested in your stuff."），
   `raw_prompt` 才是干净的原始简述（"Brainstorm three strategies to engage potential customers."）。
   推理时用户给的是正常简述，所以 raw_prompt 才是对的输入。

2. **不能用「输出必须比输入长」的过滤规则。**
   581 行 chosen 比 rejected 短、1,287 行不足 1.2 倍，但逐条核对后全是**合法**样本 ——
   它们的任务本身就是"缩短"：
       rejected(115): Rewrite the following sentence in a shorter and simpler way: "The girl who is..."
       chosen  (108): Rewrite the sentence "The girl standing in the corner is my sister." using fewer words...
   （上次 refinement 也是栽在这条规则上，误删了 98 条。）

3. **拒答检测必须收紧。**
   宽泛的 `^I (can't|cannot|am unable to)` 会误杀 4 条合法数据，例如
   "I am unable to work today. Could you please consider a fictional illness..." ——
   这是一条正常的提示词改写，不是拒答。改成必须命中"拒绝执行"的动词。

4. 非 ASCII 字符（破折号、法语重音、°²π 等）是正常内容，保留。

5. **有 260 条（0.75%）的输出把"答案"也写进了提示词**，必须删掉。
   典型形态是「冒号紧跟引号」，引号里是改写后的结果：
       IN : Rewrite the sentence: "I'm thrilled about going for a vacation"
       OUT: Rewrite the sentence: "I am delighted to embark on a vacation."  Guidance: ...
   这类样本会让模型学会直接答题而不是生成提示词。用高精度规则（冒号+引号且引号内容
   不在输入中出现）识别，260 条全部命中，人工抽查未发现误判。
   另有 153 条弱信号（引号内容不在输入中但前面没有冒号）真假混杂，精度不足，
   未自动删除，作为已知限制记录在报告里。

输出：
    data/mepo/{train,val,test}.jsonl   SFT 训练文件（messages 格式）
    data/mepo/dpo_pairs.jsonl          后续 DPO 用的偏好对
    data/mepo/dropped.jsonl            丢弃的样本及原因
    data/mepo/清洗报告.md               统计报告
"""

import json
import os
import random
import re
import statistics as st

import paths
from transformers import AutoTokenizer

SRC = os.path.join(paths.dataset_path("MePO"), "MePO.jsonl")
REFINEMENT = paths.data_dir("refinement")
OUT_DIR = paths.data_dir("mepo")
TOK_PATH = paths.find_model()

SEED = 42
MAX_TOKENS = 2048        # 8GB 显存下的长度上限
MIN_OUT_CHARS = 60       # chosen 短于这个长度不可用
LONG_INPUT_CHARS = 400   # 超过这个长度的输入打标签

SYSTEM = (
    "You are a prompt engineer. Rewrite the user's request into a clear, "
    "well-structured, directly executable prompt with explicit constraints. "
    "Output only the rewritten prompt itself -- no explanation, no restating "
    "the request, no quotation marks, no markdown code fence."
)

# 开场白：实测只有 3 条
PREAMBLE = re.compile(
    r"^\**\s*(?:here(?:'s| is| are)?|below\s+is)\b[^\n:：]{0,80}?"
    r"\b(?:prompt|version|request|instruction|rewrite)\b\s*[:：]?\s*",
    re.I,
)
# 真正的拒答：必须是"拒绝执行任务"，而不是碰巧以 I am unable to 开头的正文
REFUSAL = re.compile(
    r"^\s*i\s+(?:can'?t|cannot|won'?t|am unable to|'?m unable to)\s+"
    r"(?:fulfill|assist|help|comply|provide|answer|do that|engage|complete)\b",
    re.I,
)
# 「冒号紧跟引号」且引号内容不在输入中 —— 这是把改写后的答案写进提示词的典型形态
COLON_QUOTE = re.compile(r"[:：]\s*[\"\u201c]([^\"\u201c\u201d]{15,})[\"\u201d]")


def has_answer_leak(inp, out):
    ni = re.sub(r"\W+", "", inp.lower())
    for span in COLON_QUOTE.findall(out):
        if re.sub(r"\W+", "", span.lower()) not in ni:
            return True
    return False


def normalize(text):
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_quotes(text):
    t = text.strip()
    for left, right in (('"', '"'), ("\u201c", "\u201d"), ("\u300c", "\u300d")):
        if len(t) > 2 and t.startswith(left) and t.endswith(right):
            inner = t[1:-1].strip()
            if inner:
                return inner
    return t


def strip_wrappers(text):
    t, original = text.strip(), text.strip()
    for _ in range(4):
        before = t
        t = PREAMBLE.sub("", t).strip()
        t = strip_quotes(t)
        if t == before:
            break
    return t, t != original


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH)

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print(f"原始行数: {len(rows)}")

    # 已训练过的 refinement 数据，用于跨数据集去重，避免重复训练
    trained = set()
    ref_path = os.path.join(REFINEMENT, "train.jsonl")
    if os.path.exists(ref_path):
        for line in open(ref_path, encoding="utf-8"):
            msg = json.loads(line)["messages"]
            trained.add(re.sub(r"\W+", "", msg[1]["content"].lower()))
    print(f"已训练过的 refinement 输入: {len(trained)} 条（用于跨集去重）")

    kept, dropped, dpo = [], [], []
    stats = {}

    def drop(reason, inp, out):
        dropped.append({"reason": reason, "in_chars": len(inp), "out_chars": len(out),
                        "input": inp[:400], "output": out[:400]})

    for r in rows:
        inp = normalize(r.get("raw_prompt") or "")
        out = normalize(r.get("chosen") or "")

        if not inp or not out:
            drop("empty", inp, out)
            continue

        out, changed = strip_wrappers(out)
        if changed:
            stats["output_unwrapped"] = stats.get("output_unwrapped", 0) + 1

        if REFUSAL.match(out):
            drop("refusal", inp, out)
            continue

        if has_answer_leak(inp, out):
            drop("answer_leaked_in_output", inp, out)
            continue

        # 绝对下限；不做比值过滤（见文件头的说明）
        if len(out) < MIN_OUT_CHARS:
            drop("output_too_short", inp, out)
            continue

        total = ntok(inp) + ntok(out)
        if total > MAX_TOKENS:
            drop("too_long", inp, out)
            continue

        key = re.sub(r"\W+", "", inp.lower())
        if key in trained:
            drop("already_trained_in_refinement", inp, out)
            continue

        kept.append({"input": inp, "output": out, "tokens": total,
                     "long_input": len(inp) > LONG_INPUT_CHARS})

        # 偏好对留给后续 DPO：好提示词 vs 差提示词
        rej = normalize(r.get("rejected") or "")
        if rej and rej.lower() != inp.lower():
            dpo.append({"input": inp, "chosen": out, "rejected": rej})

    print(f"过滤后保留: {len(kept)}")

    # 去重：(输入) 和 (输入, 输出) 各一层
    seen_in, seen_pair, deduped = set(), set(), []
    for r in kept:
        ki = re.sub(r"\W+", "", r["input"].lower())
        kp = (ki, re.sub(r"\W+", "", r["output"].lower()))
        if ki in seen_in or kp in seen_pair:
            dropped.append({"reason": "duplicate", "in_chars": len(r["input"]),
                            "out_chars": len(r["output"]),
                            "input": r["input"][:400], "output": r["output"][:400]})
            continue
        seen_in.add(ki)
        seen_pair.add(kp)
        deduped.append(r)

    print(f"去重后保留: {len(deduped)}")

    records = [{"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["input"]},
        {"role": "assistant", "content": r["output"]},
    ]} for r in deduped]

    random.seed(SEED)
    random.shuffle(records)
    random.seed(SEED)
    random.shuffle(dpo)
    n = len(records)
    n_val = max(int(n * 0.03), 100)
    n_test = max(int(n * 0.03), 100)
    val, test, train = records[:n_val], records[n_val:n_val + n_test], records[n_val + n_test:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in data:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "dropped.jsonl"), "w", encoding="utf-8") as fh:
        for rec in dropped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "dpo_pairs.jsonl"), "w", encoding="utf-8") as fh:
        for rec in dpo:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    from collections import Counter

    reasons = Counter(d["reason"] for d in dropped)
    tok_counts = sorted(r["tokens"] for r in deduped)
    lines = ["# MePO 清洗报告", "",
             "| 阶段 | 行数 |", "|---|---|",
             f"| 原始加载 | {len(rows)} |",
             f"| 过滤后保留 | {len(kept)} |",
             f"| 去重后保留 | {len(deduped)} |",
             f"| **最终训练用** | **{n}** |", "",
             "## 丢弃原因", "", "| 原因 | 条数 |", "|---|---|"]
    for reason, cnt in reasons.most_common():
        lines.append(f"| `{reason}` | {cnt} |")
    lines += ["", f"另有 {stats.get('output_unwrapped', 0)} 条输出去掉了外围引号或开场白。", "",
              "## 长度分布", ""]
    if tok_counts:
        m = len(tok_counts)
        lines += [f"- user+assistant token: 中位 {tok_counts[m // 2]}，"
                  f"p90 {tok_counts[int(m * 0.9)]}，p99 {tok_counts[min(int(m * 0.99), m - 1)]}，"
                  f"最大 {tok_counts[-1]}", "",
                  "| max_length | 覆盖率 |", "|---|---|"]
        for cap in (256, 384, 512, 768, 1024, 2048):
            cov = sum(1 for x in tok_counts if x <= cap)
            lines.append(f"| {cap} | {cov}/{m} = {cov / m * 100:.1f}% |")
    lines += ["", "## 切分", "",
              f"- train {len(train)} / val {len(val)} / test {len(test)}",
              f"- 输入中位长度 {int(st.median([len(r['input']) for r in deduped]))} 字符，"
              f"输出中位长度 {int(st.median([len(r['output']) for r in deduped]))} 字符",
              f"- 长输入（>{LONG_INPUT_CHARS} 字符）{sum(1 for r in deduped if r['long_input'])} 条",
              f"- DPO 偏好对 {len(dpo)} 条（存在 dpo_pairs.jsonl）", ""]

    with open(os.path.join(OUT_DIR, "清洗报告.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print("files written to", OUT_DIR)


if __name__ == "__main__":
    main()
