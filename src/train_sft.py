"""QLoRA SFT training for the prompt-refinement dataset.

Usage (inside WSL, spin env):
    python train_sft.py train --smoke              # ~10 steps, verifies the pipeline
    python train_sft.py train                      # full run, 3 epochs
    python train_sft.py merge --adapter runs/sft/final
    python train_sft.py chat  --model runs/sft/merged
"""

import argparse
import json
import os
import sys

import paths
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DATA_DIR = paths.DATA_DIR
RUNS_DIR = paths.RUNS_DIR
find_model = paths.find_model


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    # padding side matters: for training with packing/assistant masking we want right padding
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(path, use_4bit=True):
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, device_map="auto"
        )
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")


def load_split(split, limit=None):
    """trl 1.14 requires a datasets.Dataset, not a plain list."""
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    from datasets import Dataset

    return Dataset.from_list(rows)


def cmd_train(args):
    global DATA_DIR
    DATA_DIR = paths.data_dir(args.data)
    model_path = args.model or find_model()
    print(f"base model : {model_path}")
    print(f"data       : {DATA_DIR}")

    tok = load_tokenizer(model_path)
    model = build_model(model_path, use_4bit=not args.no_4bit)

    # required for k-bit training: casts norm layers to fp32 and enables input grads
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.init_adapter:
        # 增量训练：在已有 LoRA 权重上继续训，而不是新建一个 adapter。
        # 原有的 adapter 目录不会被改动，新结果存到 runs/<data>/。
        if not os.path.isdir(args.init_adapter):
            raise SystemExit(f"初始化 adapter 不存在: {args.init_adapter}")
        print(f"init adapter: {args.init_adapter}  (增量训练，不新建 LoRA)")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None          # 权重已在 PeftModel 内，无需再注入
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable / 1e6:.1f}M")
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    train_ds = load_split("train", limit=100 if args.smoke else None)
    eval_ds = load_split("val")
    print(f"train rows : {len(train_ds)} | eval rows: {len(eval_ds)}")

    if args.output_dir:
        out_dir = args.output_dir
    elif args.smoke:
        out_dir = os.path.join(RUNS_DIR, f"{args.data}-smoke")
    else:
        out_dir = paths.run_dir(args.data)

    cfg = SFTConfig(
        output_dir=out_dir,
        max_length=args.max_length,
        # packing reshuffles samples into fixed-length blocks; that is incompatible
        # with per-token assistant masking, which we want, so keep it off.
        packing=False,
        assistant_only_loss=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        # trl 1.14 dropped `warmup_ratio`; only `warmup_steps` exists.
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps * 2,
        save_total_limit=2,
        max_steps=10 if args.smoke else -1,
        # TensorBoard needs report_to + a logging dir. Read it with:
        #   tensorboard --logdir runs/sft --host 0.0.0.0 --port 6006
        # NOTE: trl 1.14 / transformers 5.x removed the `logging_dir` argument;
        # event files land in <output_dir>/runs/<run_name>/ automatically.
        report_to=["tensorboard"],
        run_name=args.data if not args.smoke else f"{args.data}-smoke",
        seed=42,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )

    trainer.train()
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"\nadapter saved to {final_dir}")
    print(f"trainer state: {out_dir}")
    return 0


def cmd_merge(args):
    base = args.model or find_model()
    print(f"base model : {base}")
    print(f"adapter    : {args.adapter}")
    out = args.output_dir or (args.adapter.rstrip("/") + "_merged")

    tok = load_tokenizer(base)
    # merge needs the unquantized model, otherwise the LoRA math runs on 4-bit weights
    model = build_model(base, use_4bit=False)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"merged model saved to {out}")
    return 0


def cmd_chat(args):
    model_path = args.model
    tok = load_tokenizer(model_path)
    model = build_model(model_path, use_4bit=not args.no_4bit)

    system = (
        "You are a prompt engineer. Rewrite the user's request into a clear, "
        "well-structured, directly executable prompt with explicit constraints. "
        "Output only the rewritten prompt itself -- no explanation, no restating "
        "the request, no quotation marks, no markdown code fence."
    )

    prompts = args.prompt or ["Find a poem that best conveys the feeling of sadness."]
    for user in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature or None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("=" * 78)
        print("USER:", user)
        print("-" * 78)
        print(reply.strip())
        print()
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default=None)
    t.add_argument("--data", default=paths.DEFAULT_DATASET,
                   help="data/ 下的数据集名，如 refinement / mepo / enhancement")
    t.add_argument("--init_adapter", default=None,
                   help="增量训练：在此 LoRA adapter 基础上继续训（方案 B）")
    t.add_argument("--smoke", action="store_true", help="10 steps on 100 samples")
    t.add_argument("--max_length", type=int, default=768)
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch_size", type=int, default=1)
    t.add_argument("--grad_accum", type=int, default=16)
    t.add_argument("--lora_r", type=int, default=16)
    t.add_argument("--lora_alpha", type=int, default=32)
    t.add_argument("--eval_steps", type=int, default=50)
    t.add_argument("--warmup_steps", type=int, default=10)
    t.add_argument("--output_dir", default=None)
    t.add_argument("--no_4bit", action="store_true")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge")
    m.add_argument("--adapter", required=True)
    m.add_argument("--model", default=None)
    m.add_argument("--output_dir", default=None)
    m.set_defaults(func=cmd_merge)

    c = sub.add_parser("chat")
    c.add_argument("--model", required=True)
    c.add_argument("--prompt", action="append")
    c.add_argument("--max_new_tokens", type=int, default=256)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--no_4bit", action="store_true")
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
