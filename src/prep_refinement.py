"""Clean prompt-refinement-dataset into SFT training files.

What this dataset actually is (verified by inspection, not assumed):
the `input` side is often NOT a short brief. It is a messy source blob --
a 9k-char PHP file, a passage plus a question, a rambling weekend story,
a dump of engineering parameters. The `output` side is the tidy, structured
prompt. So "output must be longer than input" is the WRONG rule here, and an
earlier version of this script wrongly dropped 98 valid rows by applying it.

Pipeline: load -> normalize -> strip output wrappers -> drop RAG rows ->
drop refusals -> drop language-inconsistent rows -> length floor/ceiling ->
to chat messages -> dedupe -> split.

Outputs (all under OUT_DIR):
    train.jsonl / val.jsonl / test.jsonl   training files (messages format)
    dropped.jsonl                          everything removed, with a reason
    清洗报告.md                             before/after stats
"""

import csv
import json
import os
import random
import re

import paths
from transformers import AutoTokenizer

csv.field_size_limit(10**9)

SRC = os.path.join(paths.dataset_path("prompt-refinement-dataset"),
                   "prompt-refinement-dataset.csv")
OUT_DIR = paths.data_dir("refinement")
TOK_PATH = paths.find_model()

SEED = 42
MAX_TOKENS = 2048          # drop pairs longer than this (OOM guard on 8GB)
MIN_OUT_CHARS = 100        # below this the "refined prompt" is not usable
LONG_SOURCE_CHARS = 1500   # above this the row is tagged, but kept

# 数据集是纯英文的，system prompt 也用英文，与训练数据保持一致的语言信号。
# 想改成中文输出，需要同时改这里并补中文训练对，否则模型会随机切换语言。
SYSTEM = (
    "You are a prompt engineer. Rewrite the user's request into a clear, "
    "well-structured, directly executable prompt with explicit constraints. "
    "Output only the rewritten prompt itself -- no explanation, no restating "
    "the request, no quotation marks, no markdown code fence."
)

# 开场白：实测共约 185 条，形如
#   "Here's an improved version of the prompt:"   (107 条)
#   "Here's the improved prompt:"                 (34 条)
#   "Here's a high-quality, detailed, and effective prompt:"  (22 条)
#   "Here is the improved ..." / "Below is ..." 等
# 注意不能要求出现 "is"，因为绝大多数是 "Here's"。
PREAMBLE = re.compile(
    r"^\**\s*(?:here(?:'s| is| are)?|below\s+is)\b[^\n:：]{0,80}?"
    r"\b(?:prompt|version|request|instruction|rewrite|response)\b\s*[:：]?\s*",
    re.I,
)
PREAMBLE_ZH = re.compile(r"^\**\s*(?:改进后的|改写后的|优化后的|改進後的)[^\n:：]{0,14}[:：]\s*")

# "Improved prompt:" / "**Improved Prompt:**" / "**Improved Prompt (in English)**" 标记行
MARKER_LINE = re.compile(
    r"^\**\s*(?:improved|enhanced|rewritten|optimized)\s+prompt"
    r"(?:\s*\([^)]{0,20}\))?\s*[:：]?\s*\**\s*$",
    re.I,
)
# 被截断的开场白残片，实测形如 "of the prompt:" / "of the original prompt:"
# 中间可能夹着 original / given / following 之类的形容词
TRUNCATED_PREAMBLE = re.compile(
    r"^(?:of|to|for)\s+the\s+(?:\w+\s+){0,2}(?:prompt|request|rewrite)\s*[:：]\s*",
    re.I,
)

# 标记行出现在正文中间：输出里往往先给了几个候选版本，最后才标出「真正的那个」。
# 训练目标应该只有最后那个，所以取最后一个标记之后的内容。
BODY_MARKER = re.compile(
    r"^\**\s*(?:here'?s?\s+the\s+|here\s+is\s+the\s+)?"
    r"(?:improved|enhanced|rewritten|optimized)\s+prompt"
    r"(?:\s*\([^)]{0,20}\))?\s*[:：]?\s*\**\s*$",
    re.I | re.M,
)

# 结尾的自我点评，如 "This improved prompt provides a clear context, ..."
# 它不属于提示词本身，留在目标里会教模型输出废话。
TRAILING_NOTE = re.compile(
    r"\n\s*\n\s*(?:this|the)\s+(?:improved|rewritten|enhanced)\s+prompt\b"
    r"|\n\s*\n\s*please\s+note\s+that\s+this\s+(?:improved|rewritten)\s+prompt\b"
    r"|\n\s*\n\s*write\s+the\s+improved\s+prompt\b",
    re.I,
)
IMPROVED_LINE = re.compile(
    r"^\**\s*(?:improved|enhanced|rewritten|optimized)\s+prompt\s*[:：]?\s*\**\s*$", re.I
)

# 输出首段是在讲“如何改进提示词”，而不是改进后的提示词本身
META_FIRST_PARA = re.compile(
    r"^(?:please\s+)?(?:improve|enhance|rewrite|refine|transform|optimize)\b"
    r"|^i'?d\s+be\s+happy\s+to\s+help"
    r"|^\**\s*improve\s+the\s+prompt\b",
    re.I,
)

# 清洗后仍残留的“在讲怎么改提示词”的输出。每条都对应实测中真实出现的样本。
META_FINAL = re.compile(
    r"^(?:transform\s+the\s+given\s+basic\s+prompt"
    r"|given\s+the\s+context\s+of.{0,60}?\bimprove\b"
    r"|i\s+can\s+improve\b"
    r"|as\s+a\s+classic\s+riddle"
    r"|here\s+are\s+the\s+improved\s+prompts?\b"
    r"|(?:please\s+)?improve\s+the\s+following\s+prompts?\b)",
    re.I,
)

RAG = re.compile(r"web\s+search\s+results", re.I)
REFUSAL = re.compile(r"^\s*i\s+(?:can'?t|cannot|won'?t|am unable to)\b", re.I)
CJK = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\u3040-\u30FF]")


def strip_quotes(text):
    """Remove one layer of wrapping quotes.

    An earlier version refused to strip when the inner text also contained the
    quote character. Inspection of the 11 rows where that fired showed they are
    all genuine wrappers (the inner quotes belong to code, case names, or
    quoted phrases inside the prompt). So the rule is now simply "starts and
    ends with the same quote".
    """
    t = text.strip()
    pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u300c", "\u300d")]
    for left, right in pairs:
        if len(t) > 2 and t.startswith(left) and t.endswith(right):
            inner = t[1:-1].strip()
            if inner:
                return inner
    return t


def strip_wrappers(text):
    """Preamble and quotes nest (`Here's an improved prompt:\\n\\n"..."`), so loop.

    Returns (cleaned_text, changed).
    """
    t = text.strip()
    original = t
    for _ in range(5):
        before = t
        # a marker line such as "Improved prompt:" on its own
        lines = t.split("\n")
        while lines and (MARKER_LINE.match(lines[0].strip()) or not lines[0].strip()):
            lines.pop(0)
        t = "\n".join(lines).strip()
        t = TRUNCATED_PREAMBLE.sub("", t).strip()
        t = PREAMBLE.sub("", t).strip()
        t = PREAMBLE_ZH.sub("", t).strip()
        t = strip_quotes(t)
        if t == before:
            break
    return t, t != original


def strip_meta_first_paragraph(text):
    """Drop a leading 'how to improve a prompt' paragraph when the real prompt follows.

    Returns (text, action) where action is 'unchanged' / 'extracted' / 'drop'.
    """
    t = text.strip()
    if "\n\n" not in t:
        return t, "drop" if META_FIRST_PARA.match(t) else "unchanged"

    first, rest = t.split("\n\n", 1)
    if not META_FIRST_PARA.match(first.strip()):
        return t, "unchanged"

    # keep only the part after an explicit "Improved Prompt:" style marker line
    lines = t.split("\n")
    for i, line in enumerate(lines):
        if IMPROVED_LINE.match(line.strip()):
            tail = "\n".join(lines[i + 1:]).strip()
            if tail:
                return tail, "extracted"
    # the whole first paragraph ends with a colon -> the prompt is what follows
    if first.strip().endswith(":") and rest.strip():
        return rest.strip(), "extracted"
    return t, "drop"


def normalize(text):
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH)

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    stats = {"loaded": len(rows)}

    kept, dropped = [], []

    def drop(rec, reason):
        dropped.append({"reason": reason,
                        "in_chars": len(rec["input"]), "out_chars": len(rec["output"]),
                        "input": rec["input"][:400], "output": rec["output"][:400]})

    for rec in rows:
        inp = normalize(rec["input"])
        out = normalize(rec["output"])

        if not inp or not out:
            drop(rec, "empty")
            continue

        # 1) RAG rows: the real request is buried after a wall of search results
        if RAG.search(inp):
            drop(rec, "rag_web_search")
            continue

        # 2) output wrappers: "Here's an improved version of the prompt:" + quotes
        out, changed = strip_wrappers(out)
        if changed:
            stats["output_unwrapped"] = stats.get("output_unwrapped", 0) + 1

        # 2b) a marker line in the middle means earlier text was alternative
        #     candidates; keep only what follows the LAST marker.
        marks = list(BODY_MARKER.finditer(out))
        if marks:
            out = out[marks[-1].end():].strip()
            stats["body_marker_trimmed"] = stats.get("body_marker_trimmed", 0) + 1
            out, changed3 = strip_wrappers(out)
            if changed3:
                stats["output_unwrapped"] = stats.get("output_unwrapped", 0) + 1

        # 2c) trailing self-commentary is not part of the prompt
        note = TRAILING_NOTE.search(out)
        if note:
            out = out[:note.start()].strip()
            stats["trailing_note_trimmed"] = stats.get("trailing_note_trimmed", 0) + 1

        # 3) a leading paragraph about *how* to improve a prompt is not the target
        out, action = strip_meta_first_paragraph(out)
        if action == "extracted":
            stats["meta_extracted"] = stats.get("meta_extracted", 0) + 1
            out, changed2 = strip_wrappers(out)
            if changed2:
                stats["output_unwrapped"] = stats.get("output_unwrapped", 0) + 1
        elif action == "drop":
            drop(rec, "output_is_meta_or_chatty")
            continue

        # 3b) final cleanup pass. The trims above run *after* the first cleanup,
        #     so a row like  '"text..."\n\nThis improved prompt adds ...'  can end
        #     up with the opening quote restored but the closing one cut away.
        #     Re-run the wrapper stripper once more to settle it.
        out, changed_final = strip_wrappers(out)
        if changed_final:
            stats["output_unwrapped"] = stats.get("output_unwrapped", 0) + 1

        # 4) refusals and other non-answers
        if REFUSAL.match(out):
            drop(rec, "refusal_or_non_answer")
            continue

        # 4b) still talking about how to improve a prompt, instead of being one
        if META_FINAL.match(out) or "improved prompt" in out[:120].lower():
            drop(rec, "output_is_meta_or_chatty")
            continue

        # 5) language consistency. All 8 CJK-output rows were inspected by hand:
        # 6 of them are the model *executing* the task (a Chinese answer, a
        # translation, a Japanese explanation) instead of rewriting the prompt,
        # and the other 2 are ambiguous. They teach random language switching.
        if CJK.search(out):
            drop(rec, "output_language_flip")
            continue

        # 6) absolute floor only. NOTE: no ratio rule -- the source side is
        # legitimately allowed to be much longer than the target prompt.
        if len(out) < MIN_OUT_CHARS:
            drop(rec, "output_too_short")
            continue

        # 7) length guard
        total = ntok(inp) + ntok(out)
        if total > MAX_TOKENS:
            drop(rec, "too_long")
            continue

        kept.append({"input": inp, "output": out, "tokens": total,
                     "long_source": len(inp) > LONG_SOURCE_CHARS})

    stats["after_filters"] = len(kept)

    # 6) dedupe on normalized input (and on the full pair)
    seen_in, seen_pair, deduped = set(), set(), []
    for r in kept:
        key_in = re.sub(r"\W+", "", r["input"].lower())
        key_pair = (key_in, re.sub(r"\W+", "", r["output"].lower()))
        if key_in in seen_in or key_pair in seen_pair:
            dropped.append({"reason": "duplicate", "input": r["input"][:400],
                            "output": r["output"][:400]})
            continue
        seen_in.add(key_in)
        seen_pair.add(key_pair)
        deduped.append(r)
    stats["after_dedupe"] = len(deduped)

    # 7) to chat messages
    records, n_long_source = [], 0
    for r in deduped:
        if r["long_source"]:
            n_long_source += 1
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]},
            ]
        })
    stats["long_source"] = n_long_source

    # 8) split 90/5/5
    random.seed(SEED)
    random.shuffle(records)
    n = len(records)
    n_val = max(int(n * 0.05), 20)
    n_test = max(int(n * 0.05), 20)
    val = records[:n_val]
    test = records[n_val:n_val + n_test]
    train = records[n_val + n_test:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in data:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "dropped.jsonl"), "w", encoding="utf-8") as fh:
        for rec in dropped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 9) report
    from collections import Counter

    reasons = Counter(d["reason"] for d in dropped)
    lines = ["# prompt-refinement-dataset 清洗报告", ""]
    lines += ["| 阶段 | 行数 |", "|---|---|",
              f"| 原始加载 | {stats['loaded']} |",
              f"| 过滤后保留 | {stats['after_filters']} |",
              f"| 去重后保留 | {stats['after_dedupe']} |",
              f"| **最终训练用** | **{n}** |", ""]
    lines += ["## 丢弃原因明细", "", "| 原因 | 条数 |", "|---|---|"]
    for reason, cnt in reasons.most_common():
        lines.append(f"| `{reason}` | {cnt} |")
    lines += ["", f"另外有 {stats.get('output_unwrapped', 0)} 条输出被去掉了外围引号或开场白。", ""]

    lines += ["## 输出侧清洗动作明细（这些行是修好的，不是丢掉的）", "",
              "| 动作 | 条数 |", "|---|---|",
              f"| 去掉外围引号 / `Here's an improved version of the prompt:` 一类开场白 | "
              f"{stats.get('output_unwrapped', 0)} |",
              f"| 正文中间出现 `Improved prompt:` 标记，只保留最后一个标记之后的内容 | "
              f"{stats.get('body_marker_trimmed', 0)} |",
              f"| 剪掉结尾的自我点评（`This improved prompt provides ...`） | "
              f"{stats.get('trailing_note_trimmed', 0)} |",
              f"| 首段是「怎么改提示词」的说明，抽出后面的真正提示词 | "
              f"{stats.get('meta_extracted', 0)} |", ""]

    lines += ["## 需要知道的分布特征", "",
              f"- **长来源样本**：{stats['long_source']} 条的 input 超过 {LONG_SOURCE_CHARS} 字符"
              "（里面是代码、论文段落、流水账等原始材料）。这些**已保留**，"
              "因为它们的输出才是整理好的提示词，正是这个数据集的语义。",
              "- 这个数据集**不是**「短简述 → 提示词」，而是「杂乱原文 → 结构化提示词」。"
              "如果只想训「短简述」场景，需要额外按 input 长度做筛选。", ""]

    tok_counts = [r["tokens"] for r in deduped]
    if tok_counts:
        tok_counts.sort()
        n_tok = len(tok_counts)
        lines += ["## token 长度分布（user+assistant，Qwen3 分词器）", "",
                  f"- 中位数 {tok_counts[n_tok // 2]}",
                  f"- p90 {tok_counts[int(n_tok * 0.9)]}",
                  f"- p99 {tok_counts[min(int(n_tok * 0.99), n_tok - 1)]}",
                  f"- 最大 {tok_counts[-1]}", "",
                  "| max_length | 覆盖率 | 说明 |", "|---|---|---|"]
        notes = {
            512: "显存最省，适合先冒烟跑通",
            768: "★ 推荐：8GB 卡上余量充足，几乎不丢样本",
            1024: "更宽裕，但短样本居多，收益有限",
            1280: "接近全覆盖",
            2048: f"覆盖全部，但对本数据集严重浪费（实际最大 {tok_counts[-1]}）",
        }
        for cap in (512, 768, 1024, 1280, 2048):
            cov = sum(1 for x in tok_counts if x <= cap)
            lines.append(f"| {cap} | {cov}/{n_tok} = {cov / n_tok * 100:.1f}% | {notes[cap]} |")
        lines.append("")

    lines += ["## 切分", "", f"- train {len(train)} / val {len(val)} / test {len(test)}", ""]

    with open(os.path.join(OUT_DIR, "清洗报告.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print("files written to", OUT_DIR)


if __name__ == "__main__":
    main()
