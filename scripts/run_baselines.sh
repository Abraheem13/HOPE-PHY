#!/usr/bin/env bash
set -e
EPOCHS=${1:-300}
SEED=${2:-42}
CKPT=results/checkpoints
for MODEL in lstm gru transformer; do
  echo "=================== TRAIN $MODEL ==================="
  python scripts/train.py model=$MODEL data=llm4cp \
    train.epochs=$EPOCHS train.batch_size=256 seed=$SEED
  echo "=================== EVAL  $MODEL ==================="
  python scripts/eval_matched.py \
    --ckpt $CKPT/${MODEL}_llm4cp_seed${SEED}/best.pt --seed $SEED
done
echo ""
echo "=================== BASELINE TABLE ==================="
cat results/logs/matched_table.csv
