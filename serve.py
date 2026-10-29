"""提示词生成服务 —— 项目根目录启动入口。

用法：
    python serve.py                    # 默认 0.0.0.0:8000，自动加载交付模型
    python serve.py --port 8080        # 换端口
    python serve.py --no-adapter       # 只加载基座模型（用于效果对比）

启动后浏览器打开 http://localhost:8000/
"""

import argparse
import os
import sys
import threading
import time

# ---- 让 src/ 可被导入（paths.py 在那里）----
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

STATIC_DIR = os.path.join(ROOT, "app", "static")


def _check_deps():
    """缺依赖时给出人话提示，而不是一堆 traceback。"""
    missing = []
    for mod, pkg in (("torch", "torch"), ("transformers", "transformers"),
                     ("peft", "peft"), ("fastapi", "fastapi"),
                     ("uvicorn", "uvicorn"), ("bitsandbytes", "bitsandbytes")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("缺少依赖：" + "、".join(missing))
        print()
        print("请先安装：")
        print("    pip install -r requirements.txt")
        print()
        print("注意 torch 需按你的 CUDA 版本安装，见 https://pytorch.org")
        sys.exit(1)


def _check_torch_cuda():
    """显卡不可用时提前说清楚。"""
    import torch

    if not torch.cuda.is_available():
        print("警告：检测不到可用的 NVIDIA 显卡。")
        print("      服务会用 CPU 运行，功能正常但速度很慢。")
        print()


_check_deps()
_check_torch_cuda()

import paths  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from peft import PeftModel  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# 与训练时完全一致的 system prompt，改这里就等于改模型的输出风格
DEFAULT_SYSTEM = (
    "You are a prompt engineer. Rewrite the user's request into a clear, "
    "well-structured, directly executable prompt with explicit constraints. "
    "Output only the rewritten prompt itself -- no explanation, no restating "
    "the request, no quotation marks, no markdown code fence. "
    "IMPORTANT: The output language MUST match the user's input language. "
    "If the user writes in Chinese, the entire output must be in Chinese."
)

EXAMPLES = [
    "做个记账的",
    "我们班要选班长，想做个投票的",
    "我想给女儿的生日做个邀请函网页，要有动画",
    "帮我想个适合发朋友圈的文案",
    "帮我写个周报",
    "Find a poem that best conveys the feeling of sadness.",
]

STATE = {"tokenizer": None, "model": None, "base_model": None, "adapter": None}


class GenerateRequest(BaseModel):
    brief: str = Field(..., min_length=1, max_length=4000, description="用户的简述")
    system: str | None = Field(None, description="覆盖默认 system prompt")
    max_new_tokens: int = Field(512, ge=16, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=1.0, le=2.0)


def load(adapter_path=None):
    """加载基座 + LoRA 权重。"""
    base = paths.find_model()
    print(f"[加载] 基座模型 : {os.path.basename(base)}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        dtype=torch.bfloat16 if quant is None else None,
        device_map="auto",
    )

    if adapter_path:
        if not os.path.isdir(adapter_path):
            raise SystemExit(
                f"模型权重目录不存在：{adapter_path}\n"
                "分发包里应带有 runs/spin/spin_beta5/policy/，请确认解压是否完整。"
            )
        print(f"[加载] 模型权重 : {os.path.basename(os.path.dirname(adapter_path))}"
              f"/{os.path.basename(adapter_path)}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        STATE["adapter"] = adapter_path

    model.eval()
    STATE["tokenizer"] = tok
    STATE["model"] = model
    STATE["base_model"] = base
    print("[加载] 完成", flush=True)


def build_inputs(req: GenerateRequest):
    tok = STATE["tokenizer"]
    messages = [
        {"role": "system", "content": req.system or DEFAULT_SYSTEM},
        {"role": "user", "content": req.brief.strip()},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok([text], return_tensors="pt").to(STATE["model"].device), text


def gen_kwargs(req: GenerateRequest):
    kwargs = {
        "max_new_tokens": req.max_new_tokens,
        "repetition_penalty": req.repetition_penalty,
        "pad_token_id": STATE["tokenizer"].pad_token_id,
    }
    if req.temperature > 0:
        kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        kwargs.update(do_sample=False)
    return kwargs


app = FastAPI(title="提示词生成服务", version="1.0.0")


@app.get("/")
def index():
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(500, f"前端文件缺失：{path}")
    return FileResponse(path)


@app.get("/api/health")
def health():
    model = STATE["model"]
    return {
        "status": "ok" if model is not None else "loading",
        "base_model": STATE["base_model"],
        "adapter": STATE["adapter"],
        "device": str(model.device) if model is not None else None,
        "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available() else None,
    }


@app.get("/api/examples")
def examples():
    return {"examples": EXAMPLES}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if STATE["model"] is None:
        raise HTTPException(503, "模型还没加载完")
    tok = STATE["tokenizer"]
    inputs, _ = build_inputs(req)
    started = time.time()
    with torch.no_grad():
        out = STATE["model"].generate(**inputs, **gen_kwargs(req))
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "prompt": text,
        "tokens_in": int(inputs["input_ids"].shape[1]),
        "tokens_out": int(new_ids.shape[0]),
        "elapsed_ms": int((time.time() - started) * 1000),
    }


@app.post("/api/generate/stream")
def generate_stream(req: GenerateRequest):
    """SSE 流式输出，前端逐字渲染。"""
    if STATE["model"] is None:
        raise HTTPException(503, "模型还没加载完")

    import json

    tok = STATE["tokenizer"]
    inputs, _ = build_inputs(req)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, **gen_kwargs(req), streamer=streamer)

    thread = threading.Thread(target=STATE["model"].generate, kwargs=kwargs, daemon=True)
    thread.start()

    def event_source():
        started = time.time()
        produced = 0
        try:
            for chunk in streamer:
                if not chunk:
                    continue
                produced += 1
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps({"done": True, "chunks": produced,
                              "elapsed_ms": int((time.time() - started) * 1000)},
                             ensure_ascii=False)
                + "\n\n"
            )
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    ap = argparse.ArgumentParser(description="提示词生成服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--adapter", default=None,
                    help="模型权重目录，默认 runs/spin/spin_beta5/policy")
    ap.add_argument("--no-adapter", action="store_true", help="只加载基座模型")
    ap.add_argument("--verbose", action="store_true", help="打印完整的路径信息（排查问题用）")
    args = ap.parse_args()

    if args.verbose:
        paths.describe()
        print()

    adapter = None if args.no_adapter else (args.adapter or paths.LORA_DIR)
    load(adapter)

    print()
    print(f"  界面 : http://localhost:{args.port}/")
    print(f"  文档 : http://localhost:{args.port}/docs")
    print()
    sys.stdout.flush()

    # log_level=warning：请求日志会刷屏，关掉；报错仍然会显示
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
