#!/bin/bash
# run_hybrid.sh NAME SRC_RUN SEED REPEAT   (R2.7 hybrid real-plus-synthetic run)
# Same make call as run_job.sh for the corrected configuration. The synthetic
# lists, all-gt and unicharset are copied from SRC_RUN (the synthetic-only run
# with the same seed), and every real line in data/gt_real is inserted REPEAT
# times into list.train at uniformly random positions (rng "hybrid:SEED").
# list.eval is SRC_RUN's, unchanged. So the only difference from SRC_RUN is the
# real lines.
#   REAL_ONLY=1 run_hybrid.sh tam_real - 0 1   -> stock tam on the real lines alone
set -u
NAME=$1; SRC=$2; SEED=$3; REP=$4
export PATH=$HOME/opt/tesseract-5.5.0/bin:$PATH LD_LIBRARY_PATH=$HOME/opt/tesseract-5.5.0/lib
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-8}
R=$HOME/revision/runs; D=$R/$NAME; REAL=$HOME/revision/data/gt_real
[ -e $R/$NAME.traineddata ] && { echo "$NAME already done"; exit 0; }
mkdir -p $D
if [ ! -e $D/list.train ]; then
  if [ "${REAL_ONLY:-0}" = 1 ]; then
    GT=$REAL; ITER=${ITER:-10000}
    ls $REAL/*.lstmf > $D/all-lstmf; sleep 1
    cp $D/all-lstmf $D/list.train
    # shuffled as tesstrain shuffles list.train (shuffle.py RANDOM_SEED)
    python3 $HOME/tesseract_training/tesstrain/shuffle.py $SEED $D/list.train
    cp $D/list.train $D/list.eval
  else
    GT=$HOME/revision/data/gt_indic; ITER=100000
    cp $R/$SRC/all-gt $D/all-gt; sleep 1
    cp $R/$SRC/my.unicharset $D/my.unicharset; sleep 1
    cp $R/$SRC/all-lstmf $D/all-lstmf; sleep 1
    cp $R/$SRC/list.eval $D/list.eval
    python3 - "$R/$SRC/list.train" "$D/list.train" "$SEED" "$REP" $REAL/*.lstmf <<'P'
import sys, random
src, dst, seed, rep, real = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
lines = open(src).read().split()
rng = random.Random(f"hybrid:{seed}")
ins = real * rep
rng.shuffle(ins)
pos = sorted(rng.randrange(len(lines) + 1) for _ in ins)
out, j = [], 0
for i, l in enumerate(lines + [None]):
    while j < len(pos) and pos[j] == i:
        out.append(ins[j]); j += 1
    if l is not None: out.append(l)
open(dst, "w").write("\n".join(out) + "\n")
print(f"list.train: {len(lines)} synthetic + {len(ins)} real ({len(real)} x {rep}) = {len(out)}")
P
  fi
else
  if [ "${REAL_ONLY:-0}" = 1 ]; then GT=$REAL; ITER=${ITER:-10000}; else GT=$HOME/revision/data/gt_indic; ITER=100000; fi
fi
cd $HOME/tesseract_training/tesstrain
[ "${PREP_ONLY:-0}" = 1 ] && { echo prepared $NAME; exit 0; }
echo "[$(date)] start $NAME SRC=$SRC SEED=$SEED REP=$REP ITER=$ITER"
make training MODEL_NAME=$NAME START_MODEL=tam TESSDATA=$HOME/revision/tessdata \
  GROUND_TRUTH_DIR=$GT DATA_DIR=$R LANG_TYPE=Indic RECODER=--pass_through_recoder \
  RANDOM_SEED=$SEED MAX_ITERATIONS=$ITER > $R/$NAME.make.log 2>&1 || { echo "[$(date)] FAILED $NAME"; exit 1; }
echo "[$(date)] trained $NAME; sweeping"
cd $HOME/revision/tgb && python3 sweep_checkpoints.py --data-dir $R --model $NAME \
  --out $HOME/revision/results/sweep_$NAME.json --jobs 6 > $R/$NAME.sweep.log 2>&1
echo "[$(date)] done $NAME"
