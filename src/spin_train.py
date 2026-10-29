"""SPIN 自博弈微调：把 SPIN 用到提示词生成任务上。

论文：Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
      (ICML 2024, arXiv 2401.01335)

核心 loss（论文式 4.7）：
    L = E[ ℓ( λ·log(p_θ(y|x)/p_θt(y|x)) − λ·log(p_θ(y′|x)/p_θt(y′|x)) ) ]
    ℓ(t) = log(1+exp(−t))，y ~ p_data（真答案），y′ ~ p_θt（上一轮模型生成）
    λ（论文）= beta（代码），默认 0.1

工程方案：只加载一份 4-bit 基座，挂两个 LoRA adapter
    opp    = θt，冻结，用于算参考 log-概率 + 生成 y′
    policy = θ，  可训练，初始权重与 opp 完全相同
这样显存约 4–5GB，8GB 单卡可跑。

子命令：
    explain  验证实现正确性：θ=θt 时 loss 必须恰为 ln2 ≈ 0.6931
    generate 用指定 adapter 生成 y′，产出 pairs 文件
    train    跑一轮 SPIN 训练
    eval     在目标集 / 保持集上评测
"""

import argparse
import json
import math
import os
import random
import sys
import time

import paths
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = (
    "You are a prompt engineer. Rewrite the user's request into a clear, "
    "well-structured, directly executable prompt with explicit constraints. "
    "Output only the rewritten prompt itself -- no explanation, no restating "
    "the request, no quotation marks, no markdown code fence. "
    "IMPORTANT: The output language MUST match the user's input language. "
    "If the user writes in Chinese, the entire output must be in Chinese."
)

LN2 = math.log(2.0)


def gpu_mem(tag=""):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  [mem] {tag:<22} 已分配 {alloc:.2f} GiB / 可用 {free / 1024**3:.2f} GiB",
              flush=True)


# ---------------------------------------------------------------- 模型加载

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(paths.find_model())
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        paths.find_model(),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )


def load_dual(base_path, opp_adapter):
    """加载基座 + 两个 adapter（opp 冻结 / policy 可训练，初始权重相同）。"""
    model = load_base()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = PeftModel.from_pretrained(model, opp_adapter, adapter_name="opp", is_trainable=False)
    model.load_adapter(opp_adapter, adapter_name="policy", is_trainable=True)

    # PEFT 的参数名里带 adapter 名，靠这个区分 opp / policy
    n_policy = n_opp = 0
    for name, param in model.named_parameters():
        is_policy = "policy" in name
        param.requires_grad = is_policy
        if is_policy:
            n_policy += 1
        elif "opp" in name:
            n_opp += 1

    if n_policy == 0:
        sample = [n for n, _ in model.named_parameters()][:3]
        raise SystemExit(f"没有找到 policy adapter 的参数，样例名: {sample}")
    print(f"[model] policy 张量 {n_policy} 个（可训练）/ opp 张量 {n_opp} 个（冻结）")

    model.eval()
    return model


# ------------------------------------------------------- 序列 log-概率

def encode_pair(tok, prompt, response, max_len):
    """返回 (input_ids, labels)。

    要点（对齐官方实现）：
      - prompt 与 response 分别编码后再拼接，避免用长度去切分导致的错位
      - 关键：截断时保留 response（从左侧截 prompt）
      - 只对 response 段算 loss
    """
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    if len(r_ids) >= max_len:
        r_ids = r_ids[: max_len - 1] + [tok.eos_token_id]
        p_ids = []
    elif len(p_ids) + len(r_ids) > max_len:
        keep = max_len - len(r_ids)
        p_ids = p_ids[-keep:]

    ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids
    return ids, labels


def batch_logprob(model, tok, prompts, responses, max_len):
    """返回每条样本 response 段的 log-概率【之和】（sum，不是 mean）。"""
    encoded = [encode_pair(tok, p, r, max_len) for p, r in zip(prompts, responses)]
    maxlen = max(len(ids) for ids, _ in encoded)

    input_ids, labels, attn = [], [], []
    for ids, labs in encoded:
        pad = maxlen - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(labs + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(input_ids, device=model.device)
    labels = torch.tensor(labels, device=model.device)
    attn = torch.tensor(attn, device=model.device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = tgt != -100
    if mask.sum() == 0:
        return torch.zeros(input_ids.size(0), device=model.device)

    # 内存友好：按序列分块做 float32 归一化。
    # 直接写 F.log_softmax(logits.float(), dim=-1) 会为整个 [B, T, 151936] 张量
    # 复制一份 float32（实测单次就吃掉 1.4 GiB，8GB 卡直接打满）。
    # 分块后峰值只有 ~120 MiB。
    chunk = 64
    B, T, _ = logits.shape
    parts = []
    for i in range(0, T, chunk):
        lg = logits[:, i:i + chunk, :].float()
        t = tgt[:, i:i + chunk].clamp(min=0).unsqueeze(2)
        picked = lg.gather(2, t).squeeze(2)          # [B, c]
        parts.append(picked - torch.logsumexp(lg, dim=-1))
        del lg
    logp = torch.cat(parts, dim=1)                   # [B, T]
    return (logp * mask).sum(-1)          # ★ sum，不是 mean


def spin_loss(model, tok, prompts, real, generated, beta, max_len):
    """论文式 4.7 的直接实现。"""
    model.set_adapter("policy")
    pi_real = batch_logprob(model, tok, prompts, real, max_len)
    pi_gen = batch_logprob(model, tok, prompts, generated, max_len)

    model.set_adapter("opp")
    with torch.no_grad():
        o_real = batch_logprob(model, tok, prompts, real, max_len)
        o_gen = batch_logprob(model, tok, prompts, generated, max_len)

    logits = (pi_real - pi_gen) - (o_real - o_gen)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss, logits.detach()


# ---------------------------------------------------------------- 数据

def pick_briefs(n, seed=42):
    """从 MePO 训练集里均匀抽取简述（x）。"""
    path = os.path.join(paths.data_dir("mepo"), "train.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    step = max(len(rows) // n, 1)
    return [(r["messages"][1]["content"], r["messages"][2]["content"])
            for r in rows[::step][:n]]


# ---------------------------------------------------------------- explain

def cmd_explain(args):
    """实现正确性验证：θ=θt 时 logits 必须为 0、loss 必须为 log(2)。"""
    tok = load_tokenizer()
    model = load_dual(paths.find_model(), args.init)
    gpu_mem("加载完模型")

    samples = pick_briefs(args.n, seed=0)
    prompts = [p for p, _ in samples]
    real = [r for _, r in samples]
    # 用真答案冒充生成答案：此时 policy 与 opp 也完全相同，
    # logits 必然为 0，可以单独检验 loss 的量纲
    logits_same_all = []
    for i in range(len(prompts)):
        _, lg = spin_loss(model, tok, prompts[i:i + 1], real[i:i + 1], real[i:i + 1],
                          args.beta, args.max_len)
        logits_same_all.append(lg.mean().item())
        del lg
    torch.cuda.empty_cache()
    gpu_mem("第一次 spin_loss")

    # 真正的起步检查：用 policy 生成 y′，此时 θ=θt，logits 也应为 0
    model.set_adapter("opp")
    generated = []
    for p in prompts[: args.n]:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True)
        ids = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        generated.append(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        del out, ids
    torch.cuda.empty_cache()
    gpu_mem("生成完 y'")

    losses, logit_means, logit_maxabs = [], [], []
    for i in range(len(prompts)):
        l, lg = spin_loss(model, tok, prompts[i:i + 1], real[i:i + 1], generated[i:i + 1],
                          args.beta, args.max_len)
        losses.append(l.item())
        logit_means.append(lg.mean().item())
        logit_maxabs.append(lg.abs().max().item())
        del l, lg
    torch.cuda.empty_cache()

    loss_val = sum(losses) / len(losses)
    logit_mean = sum(logit_means) / len(logit_means)
    logit_max = max(logit_maxabs)

    print("=" * 62)
    print("SPIN 实现正确性检查")
    print("=" * 62)
    print(f"  样本数            : {len(prompts)}")
    print(f"  beta              : {args.beta}")
    print()
    import statistics as _st

    print(f"  [对照] 生成答案=真答案时  logits 均值 = {_st.mean(logits_same_all):+.6f}  "
          f"（应为 0）")
    print(f"  [正式] θ=θt 起步时        logits 均值 = {logit_mean:+.6f}  "
          f"最大绝对值 = {logit_max:.6f}")
    print(f"                            loss      = {loss_val:.6f}")
    print(f"                            理论值 ln2 = {LN2:.6f}")
    print()
    ok = abs(logit_mean) < 1e-3 and abs(loss_val - LN2) < 1e-3
    print("  结论:", "✅ 通过，实现与论文一致" if ok else
          "❌ 不通过 —— logits 应为 0、loss 应为 ln2，请检查符号与对齐方式")

    print("\n  生成答案样例（对手模型输出 y′）:")
    if generated:
        print("   ", generated[0][:200].replace("\n", " "))
    return 0 if ok else 1


# ---------------------------------------------------------------- generate

def cmd_generate(args):
    """用指定 adapter（= θt）生成 y′，产出训练用的 pairs 文件。

    同时缓存对手模型的 log-概率：θt 在整轮迭代中冻结不变，
    o_real / o_gen 是常量。提前算好存进文件，训练阶段就不用每步重算，
    直接把每样本的前向次数从 4 次降到 2 次（实测约 2× 加速）。
    """
    tok = load_tokenizer()
    model = load_dual(paths.find_model(), args.model)
    model.set_adapter("opp")

    samples = pick_briefs(args.n, seed=args.seed)
    out_path = os.path.join(paths.RUNS_DIR, "spin", f"pairs_iter{args.iter}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    started = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, (brief, ref) in enumerate(samples, 1):
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": brief}],
                tokenize=False, add_generation_prompt=True)
            ids = tok([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # 缓存对手 log-概率
            with torch.no_grad():
                o_real = batch_logprob(model, tok, [brief], [ref], args.max_len).item()
                o_gen = batch_logprob(model, tok, [brief], [gen], args.max_len).item()

            fh.write(json.dumps({
                "prompt": brief, "real": ref, "generated": gen,
                "o_real": o_real, "o_gen": o_gen,
            }, ensure_ascii=False) + "\n")
            # 生成阶段同样会受显存缓存池饱和影响（实测速度从 9.7 掉到 14.4 秒/条），
            # 定期归还缓存块可以拉回速度。
            if i % 20 == 0:
                torch.cuda.empty_cache()
            if i % 25 == 0:
                el = time.time() - started
                print(f"  {i}/{len(samples)}  {el / i:.1f}s/条  "
                      f"预计剩余 {(len(samples) - i) * el / i / 60:.0f} 分钟", flush=True)

    print(f"\n生成完成: {out_path}")
    print(f"耗时 {(time.time() - started) / 60:.1f} 分钟，共 {len(samples)} 条")
    return 0


# ---------------------------------------------------------------- train

def cmd_train(args):
    tok = load_tokenizer()
    model = load_dual(paths.find_model(), args.opp)

    pairs_path = args.pairs or os.path.join(paths.RUNS_DIR, "spin", f"pairs_iter{args.iter}.jsonl")
    data = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    print(f"[data] {pairs_path}  共 {len(data)} 条")

    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"[model] 可训练参数 {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    total_steps = len(data) * args.epochs // args.grad_accum
    warmup = max(int(total_steps * 0.1), 3)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    out_dir = args.output_dir or os.path.join(paths.RUNS_DIR, "spin", f"iter{args.iter}")
    os.makedirs(out_dir, exist_ok=True)
    log_fh = open(os.path.join(out_dir, "train_log.jsonl"), "w", encoding="utf-8")

    print(f"[train] 总步数 {total_steps}，warmup {warmup}，lr {args.lr}，beta {args.beta}")
    random.seed(args.seed)

    use_cache = all("o_real" in r and "o_gen" in r for r in data[: min(len(data), 20)])
    print(f"[train] 对手 log-概率缓存: {'启用（训练前向 2 次/样本）' if use_cache else '未启用（4 次/样本）'}")

    step, micro, t0 = 0, 0, time.time()
    opt.zero_grad()
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for idx in order:
            rec = data[idx]
            if use_cache:
                model.set_adapter("policy")
                pi_real = batch_logprob(model, tok, [rec["prompt"]], [rec["real"]], args.max_len)
                pi_gen = batch_logprob(model, tok, [rec["prompt"]], [rec["generated"]], args.max_len)
                logits = (pi_real - pi_gen) - (rec["o_real"] - rec["o_gen"])
                loss = -F.logsigmoid(args.beta * logits).mean()
            else:
                loss, logits = spin_loss(
                    model, tok, [rec["prompt"]], [rec["real"]], [rec["generated"]],
                    args.beta, args.max_len)
            (loss / args.grad_accum).backward()
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step()
                opt.zero_grad()
                step += 1

                # 关键修复：PyTorch 的显存缓存池只增不减，训练久了会逼近显存上限，
                # 之后每次分配都要先驱逐旧块，速度从 0.6s/样本掉到 11s/样本。
                # 主动归还缓存块可以避免池子饱和。
                # （bitsandbytes 在 sm_120 上走反量化回退，反复申请释放大块显存，
                #   是碎片化的主要来源。）
                if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()

                if step % 5 == 0 or step == 1:
                    rec_log = {
                        "step": step, "epoch": epoch, "loss": round(loss.item(), 6),
                        "logit_mean": round(logits.mean().item(), 4),
                        "lr": round(lr_at(step), 9),
                        "elapsed_s": round(time.time() - t0, 1),
                        "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                        "mem_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                    }
                    log_fh.write(json.dumps(rec_log) + "\n")
                    log_fh.flush()
                    print(f"  step {step}/{total_steps}  loss {loss.item():.4f}  "
                          f"logit {logits.mean().item():+.3f}  "
                          f"{(time.time() - t0) / max(step, 1):.1f}s/step  "
                          f"显存 {torch.cuda.memory_allocated() / 1024**3:.2f}/"
                          f"{torch.cuda.memory_reserved() / 1024**3:.2f}GiB", flush=True)

    model.set_adapter("policy")
    model.save_pretrained(out_dir, selected_adapters=["policy"])
    tok.save_pretrained(out_dir)
    # PEFT 在多 adapter 场景下会把权重写进以 adapter 名命名的子目录
    adapter_dir = os.path.join(out_dir, "policy")
    print(f"\n保存到 {out_dir}")
    print(f"实际 adapter 路径（下游要用这个）: {adapter_dir}")
    return 0


# ---------------------------------------------------------------- eval

def cmd_eval(args):
    """在目标集（mepo test）和保持集（refinement test+val）上算 assistant-only loss。"""
    tok = load_tokenizer()
    model = load_dual(paths.find_model(), args.adapter)
    model.set_adapter("opp")

    def load_holdout(dataset):
        rows = []
        for split in ("test", "val"):
            p = os.path.join(paths.data_dir(dataset), f"{split}.jsonl")
            if os.path.exists(p):
                rows += [json.loads(l)["messages"] for l in open(p, encoding="utf-8")]
        return rows

    results = {}
    for name, dataset in (("target(mepo)", "mepo"), ("retain(refinement)", "refinement")):
        rows = load_holdout(dataset)
        step = max(len(rows) // args.n, 1)
        rows = rows[::step][: args.n]
        losses = []
        for m in rows:
            with torch.no_grad():
                lp = batch_logprob(model, tok, [m[1]["content"]], [m[2]["content"]], args.max_len)
            ntok = len(tok(m[2]["content"], add_special_tokens=False)["input_ids"]) + 1
            losses.append(-lp.item() / max(ntok, 1))     # 转成每 token 平均，便于跨集比较
        results[name] = sum(losses) / len(losses)
        print(f"  {name:<22} n={len(rows):<5} assistant-only loss = {results[name]:.4f}")

    out = os.path.join(paths.RUNS_DIR, "spin", f"eval_{os.path.basename(args.adapter.rstrip('/'))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explain", help="验证实现正确性（loss 必须等于 ln2）")
    e.add_argument("--init", required=True, help="起点 adapter，如 runs/refinement/final")
    e.add_argument("--n", type=int, default=4)
    e.add_argument("--beta", type=float, default=0.1)
    e.add_argument("--max_len", type=int, default=384)
    e.add_argument("--max_new_tokens", type=int, default=80)
    e.set_defaults(func=cmd_explain)

    g = sub.add_parser("generate", help="用 θt 生成 y′")
    g.add_argument("--model", required=True, help="θt 的 adapter 路径")
    g.add_argument("--iter", type=int, required=True)
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--max_new_tokens", type=int, default=160)
    g.add_argument("--max_len", type=int, default=384)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="跑一轮 SPIN 训练")
    t.add_argument("--opp", required=True, help="对手模型 = 本轮起始 checkpoint")
    t.add_argument("--iter", type=int, required=True)
    t.add_argument("--pairs", default=None)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--grad_accum", type=int, default=8)
    t.add_argument("--max_len", type=int, default=384)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output_dir", default=None, help="覆盖输出目录，便于跑不同配置的变体")
    t.add_argument("--empty_cache_every", type=int, default=1,
                   help="每隔多少步主动清一次显存缓存池（0=不清）")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("eval", help="评测目标能力与保持能力")
    v.add_argument("--adapter", required=True)
    v.add_argument("--n", type=int, default=100)
    v.add_argument("--max_len", type=int, default=384)
    v.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
