#!/usr/bin/env bash
# X5 short runs (IJDAR revision): the first 12,000 steps of the 4-epoch
# fine-tunes (same cosine schedule with one-epoch warmup), checkpointed at
# 250, 500 and every 1,000 steps, each scored on the 300-line test set.
# Needs the X5_SAVE_STEPS / X5_MAX_STEP patch in tools/program.py.
#   run_x5_short.sh            both arms: train model, then train rebuilt
#                              while scoring model on CPU, then score rebuilt
set -uo pipefail
R=$HOME/revision/rec; TGB=$HOME/revision/tgb
CFG=$TGB/experiments/paddle/ft_tamilsynth.yml
STEPS=250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000
source $R/env/bin/activate; cd $R/PaddleOCR
log() { echo "[$(date)] $*"; }
dict() { case $1 in model) echo $R/PaddleOCR/ppocr/utils/dict/ppocrv5_ta_dict.txt;; rebuilt) echo $R/data/tamilsynth_dict.txt;; esac; }

train() {  # arm
  local NAME=x5short_$1 D=$(dict $1) OUT=$R/output/x5short_$1
  log "train $NAME"
  X5_SAVE_STEPS=$STEPS X5_MAX_STEP=12000 python tools/train.py -c $CFG -o \
    Global.character_dict_path=$D Global.save_model_dir=$OUT \
    Global.save_res_path=$OUT/predicts.txt Global.epoch_num=4 \
    Global.eval_batch_step="[0,12000]" \
    Global.use_amp=True Global.scale_loss=1024.0 \
    Global.use_dynamic_loss_scaling=True Global.amp_level=O1 \
    Train.loader.batch_size_per_card=8 Eval.loader.batch_size_per_card=8 \
    > $OUT.train.log 2>&1 || log "TRAIN EXIT $? $NAME"
  log "trained $NAME"
}

score() {  # arm, on CPU
  local NAME=x5short_$1 D=$(dict $1) OUT=$R/output/x5short_$1
  for s in ${STEPS//,/ }; do
    local p=$OUT/iter_step_$s inf=$OUT/inference_s$s preds=$OUT/preds_s$s
    [ -f $p.pdparams ] || { log "missing $p"; continue; }
    [ -f $inf/inference.pdiparams ] || CUDA_VISIBLE_DEVICES="" python tools/export_model.py -c $CFG -o \
      Global.character_dict_path=$D Global.save_model_dir=$OUT Global.use_gpu=False \
      Global.pretrained_model=$p Global.save_inference_dir=$inf > $OUT.export_s$s.log 2>&1 \
      || { log "EXPORT FAILED $NAME s$s"; continue; }
    [ "$(ls $preds 2>/dev/null | wc -l)" = 300 ] || CUDA_VISIBLE_DEVICES="" python \
      $TGB/experiments/paddle_predict.py --model-name ta_PP-OCRv5_mobile_rec --model-dir $inf \
      --testset $TGB/testset --pred-dir $preds > $OUT.predict_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES="" python $TGB/experiments/score_preds.py $preds \
      --label ${NAME}_s$s --json $HOME/revision/results/x5_short_$1.json
  done
  log "scored $NAME"
}

mkdir -p $R/output/x5short_model $R/output/x5short_rebuilt
train model
score model > $HOME/revision/x5_short_score_model.log 2>&1 &
train rebuilt
wait
score rebuilt
log "done x5 short"
