#!/usr/bin/env bash
# ============================================================================
# SPIN pilot 完整流水线（可复现）
#
# 做四件事：
#   1) 按迭代轮次跑 SPIN：生成 y′ → 训练 → 记录输出
#   2) 跑「继续 SFT」对照组（使用与 SPIN 完全相同的样本）
#   3) 跑 DPO 对照组（用 MePO 的 chosen/rejected 偏好对）
#   4) 汇总成对照表
#
# 用法：
#   N=200 ITERS=2 bash src/run_spin_experiment.sh
#
# 环境变量：
#   N        训练样本数（默认 200）
#   ITERS    SPIN 迭代轮数（默认 2）
#   EPOCHS   每轮训练轮数（默认 1）
#   START    SPIN 起点 adapter（默认 runs/refinement/final）
#   MAXTOK   生成时最大新 token 数（默认 128）
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/home/hanghang/miniconda3/envs/spin/bin/python
export HF_HUB_OFFLINE=1

N=${N:-200}
ITERS=${ITERS:-2}
EPOCHS=${EPOCHS:-1}
START=${START:-runs/refinement/final}
MAXTOK=${MAXTOK:-128}
MAXLEN=${MAXLEN:-256}

echo "=============================================================="
echo "SPIN pilot: N=$N ITERS=$ITERS EPOCHS=$EPOCHS MAXLEN=$MAXLEN"
echo "起点: $START  生成上限: $MAXTOK token"
echo "开始时间: $(date '+%F %T')"
echo "=============================================================="

CUR="$START"

for t in $(seq 0 $((ITERS - 1))); do
    echo ""
    echo "========== SPIN 第 $t 轮 =========="
    echo "[$(date '+%T')] 1/3 用 θ$t 生成 y′"
    $PY -u src/spin_train.py generate \
        --model "$CUR" --iter "$t" --n "$N" --max_new_tokens "$MAXTOK"

    echo "[$(date '+%T')] 2/3 SPIN 训练"
    $PY -u src/spin_train.py train \
        --opp "$CUR" --iter "$t" --epochs "$EPOCHS" --grad_accum 8 --lr 1e-5 \
        --max_len "$MAXLEN" --empty_cache_every 1

    # PEFT 把权重写在 <out>/policy/ 子目录下，下游必须用这个路径
    NUEXT="runs/spin/iter$t/policy"
    echo "[$(date '+%T')] 3/3 记录 iter$t 的输出与指标"
    $PY -u src/record_outputs.py --adapter "$NUEXT" --tag "spin_iter$t" --n 80

    CUR="$NUEXT"
done

echo ""
echo "========== 对照组 A：继续 SFT =========="
echo "[$(date '+%T')] 用与 SPIN 相同的样本构建对照组数据"
$PY src/build_sft_control.py 0

echo "[$(date '+%T')] 训练继续 SFT（同起点、同样本数）"
$PY -u src/train_sft.py train --data spin_sft --init_adapter "$START" \
    --epochs "$EPOCHS" --lr 1e-5 --max_length 384 \
    --output_dir runs/spin/sft_control

echo "[$(date '+%T')] 记录对照组输出"
$PY -u src/record_outputs.py --adapter runs/spin/sft_control/final --tag sft_control --n 80

echo ""
echo "========== 汇总对比 =========="
$PY -u src/record_outputs.py --compare --baseline baseline

echo ""
echo "=============================================================="
echo "全部完成: $(date '+%F %T')"
echo "对比表: runs/spin/对照结果.md"
echo "=============================================================="
