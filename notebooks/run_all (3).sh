#!/usr/bin/env bash
# Runs every (dataset, split, seed) for ChemBERTa. Safe to re-run after a crash or timeout:
# finished runs are skipped and unfinished ones resume from their last epoch.
#
# Usage (inside tmux):   bash run_all.sh            # pretrained checkpoint
#                        bash run_all.sh scratch    # random init, for the scratch-vs-pretrained comparison
set -u
cd /home/jovyan
EXTRA=""
if [ "${1:-}" = "scratch" ]; then EXTRA="--from_scratch"; fi

# Smallest first, so problems show up in minutes rather than hours.
for DATASET in bace bbbp hiv; do
  for SPLIT in scaffold random; do
    for SEED in 0 1 2; do
      SPLITFILE=""
      # If Ana B.'s shared split file exists, use it; otherwise the script builds its own split.
      # Name files splits/<dataset>_<split>.<npz|json|csv>, e.g. splits/hiv_scaffold.npz
      # (datasets: bace, bbbp, hiv; splits: random, scaffold). A per-seed file
      # splits/<dataset>_<split>_seed<k>.<ext> is also accepted.
      for EXT in npz json csv; do
        for CAND in "splits/${DATASET}_${SPLIT}_seed${SEED}.${EXT}" "splits/${DATASET}_${SPLIT}.${EXT}"; do
          if [ -z "$SPLITFILE" ] && [ -f "$CAND" ]; then SPLITFILE="--split_file $CAND"; fi
        done
      done
      # One fixed random split shared by all seeds, so the 3 seeds can be ensembled.
      SS=""
      if [ "$SPLIT" = "random" ] && [ -z "$SPLITFILE" ]; then SS="--split_seed 0"; fi
      echo "=== ${DATASET} ${SPLIT} seed ${SEED} ${EXTRA} ==="
      python chemberta_finetune.py --dataset "$DATASET" --split "$SPLIT" --seed "$SEED" $SS $SPLITFILE $EXTRA \
        || echo "!!! FAILED: ${DATASET} ${SPLIT} seed ${SEED}"
    done
  done
done
