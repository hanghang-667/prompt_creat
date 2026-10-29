"""统一路径解析（跨平台，Windows / Linux / macOS 通用）。

所有脚本都从这里取路径，任何地方都不再硬编码绝对路径，
整个项目目录可以随意移动、复制给别人。

期望的目录结构（接收方拿到分发包后照着放即可）：

    promptgen/
    ├── serve.py            启动入口：python serve.py
    ├── requirements.txt
    ├── models/             基座模型放这里（需自行下载）
    │   └── Qwen3-4B-Instruct-2507/
    ├── app/static/         网页界面
    ├── src/                代码
    ├── data/               训练数据（可选）
    └── runs/               模型权重与实验记录

代码在 src/ 下，所以 PROJECT_ROOT = src/ 的上一级。
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
APP_DIR = os.path.join(PROJECT_ROOT, "app")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
STATIC_DIR = os.path.join(APP_DIR, "static")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 每个数据集一个子目录，互不覆盖
DEFAULT_DATASET = "refinement"
DATA_DIR = os.path.join(DATA_ROOT, DEFAULT_DATASET)


def data_dir(name=DEFAULT_DATASET):
    """返回某个数据集的训练文件目录。"""
    return os.path.join(DATA_ROOT, name)


def run_dir(name=DEFAULT_DATASET):
    """训练输出目录：runs/<数据集名>/"""
    return os.path.join(RUNS_DIR, name)


def log_file(name=DEFAULT_DATASET):
    return os.path.join(RUNS_DIR, f"{name}_train.log")


# 交付模型：SPIN 优化后的 adapter（服务端默认加载这个）
DELIVERABLE_ADAPTER = os.path.join(RUNS_DIR, "spin", "spin_beta5", "policy")
# 训练起点模型（第一轮 SFT 产物）
BASE_ADAPTER = os.path.join(RUNS_DIR, "refinement", "final")

# 兼容旧调用：默认指向交付模型
LORA_DIR = DELIVERABLE_ADAPTER
LOG_FILE = log_file()


def _looks_like_model(path):
    """目录里是否有模型文件（比按名字匹配更可靠）。"""
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    has_config = "config.json" in names
    has_weight = any(n.endswith((".safetensors", ".bin")) for n in names)
    return has_config and has_weight


def _scan_for_model(base):
    """在 base 里找模型：base 本身是模型，或 base 下有子目录是模型。"""
    if _looks_like_model(base):
        return base
    if not os.path.isdir(base):
        return None
    # 优先找名字像 qwen 的
    entries = sorted(os.listdir(base))
    for prefer_qwen in (True, False):
        for name in entries:
            if prefer_qwen and not name.lower().startswith("qwen"):
                continue
            sub = os.path.join(base, name)
            if _looks_like_model(sub):
                return sub
    return None


def find_model():
    """返回本地 Qwen 基座模型目录。

    查找顺序：
      1. 环境变量 BASE_MODEL（可以指向模型目录本身，也可以指向存放模型的上一级）
      2. 项目内的 models/
      3. 项目同级的 models/

    **不依赖 datasets/ 目录**——把项目发给别人时，对方通常只有模型没有数据集。
    """
    env = os.environ.get("BASE_MODEL")
    if env:
        found = _scan_for_model(env)
        if found:
            return found
        raise SystemExit(
            f"环境变量 BASE_MODEL 指向的目录里没找到模型：{env}\n"
            "该目录下应该有 config.json 和 model.safetensors。"
        )

    candidates = [
        MODELS_DIR,
        os.path.join(os.path.dirname(PROJECT_ROOT), "models"),
    ]
    for base in candidates:
        found = _scan_for_model(base)
        if found:
            return found

    raise SystemExit(
        "找不到基座模型。请二选一：\n"
        f"  1) 把模型下载到 {MODELS_DIR} 下，例如\n"
        f"     {os.path.join(MODELS_DIR, 'Qwen3-4B-Instruct-2507')}\n"
        "  2) 设置环境变量指向它\n"
        "     Windows:  set BASE_MODEL=D:\\models\\Qwen3-4B-Instruct-2507\n"
        "     Linux:    export BASE_MODEL=/path/to/Qwen3-4B-Instruct-2507"
    )


def find_datasets_root():
    """返回包含 datasets/ 的目录；找不到返回 None。

    只有数据准备脚本需要它，服务端不需要，所以不抛异常。
    """
    candidates = [
        PROJECT_ROOT,
        os.path.dirname(PROJECT_ROOT),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "datasets")):
            return cand
    return None


def dataset_path(name):
    """返回某个数据集在 datasets/ 下的路径。"""
    root = find_datasets_root()
    if root is None:
        raise SystemExit(
            "找不到 datasets/ 目录。数据准备脚本需要它，请把数据集放在项目同级。"
        )
    return os.path.join(root, "datasets", name)


def describe():
    """启动时打印一遍，方便排查路径问题。"""
    print(f"项目目录 : {PROJECT_ROOT}")
    print(f"数据目录 : {DATA_ROOT}")
    print(f"输出目录 : {RUNS_DIR}")
    ds = find_datasets_root()
    print(f"数据集   : {os.path.join(ds, 'datasets') if ds else '（未找到，训练脚本需要）'}")
    print(f"模型权重 : {LORA_DIR}")
    try:
        print(f"基座模型 : {find_model()}")
    except SystemExit as exc:
        print(f"基座模型 : !! {str(exc).splitlines()[0]}")
    print(f"Python   : {sys.version.split()[0]} ({sys.platform})")


if __name__ == "__main__":
    describe()
