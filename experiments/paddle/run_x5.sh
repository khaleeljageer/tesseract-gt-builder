#!/usr/bin/env bash
# X5 (IJDAR revision): fine-tune PaddleOCR's Tamil recogniser on TamilSynth,
# then export, predict and score every epoch on the 300-line test set.
#
#   run_x5.sh NAME DICT EPOCHS [eval-only]
#     NAME    output dir under ~/revision/rec/output
#     DICT    "model" = the pretrained 513-entry dictionary (head kept);
#             "rebuilt" = the 84-character TamilSynth dictionary (head
#             reinitialised: PaddleOCR's analogue of rebuilding the recoder)
#
# Epoch selection uses the 98-line selection split; reporting uses rep202.
set -uo pipefail
NAME=$1 DICT=$2 EPOCHS=$3 MODE=${4:-train}
R=$HOME/revision/rec
TGB=$HOME/revision/tgb
CFG=$TGB/experiments/paddle/ft_tamilsynth.yml
OUT=$R/output/$NAME
case $DICT in
  model)   D=$R/PaddleOCR/ppocr/utils/dict/ppocrv5_ta_dict.txt ;;
  rebuilt) D=$R/data/tamilsynth_dict.txt ;;
  *) echo "bad DICT $DICT"; exit 2 ;;
esac
source $R/env/bin/activate
cd $R/PaddleOCR
COMMON="Global.character_dict_path=$D Global.save_model_dir=$OUT Global.save_res_path=$OUT/predicts.txt"
log() { echo "[$(date)] $*"; }

if [ "$MODE" = train ]; then
  log "train $NAME dict=$DICT epochs=$EPOCHS"
  python tools/train.py -c $CFG -o $COMMON Global.epoch_num=$EPOCHS \
    Global.eval_batch_step="[0,12000]" \
    Global.use_amp=True Global.scale_loss=1024.0 \
    Global.use_dynamic_loss_scaling=True Global.amp_level=O1 \
    Train.loader.batch_size_per_card=8 Eval.loader.batch_size_per_card=8 \
    > $OUT.train.log 2>&1 || { log "TRAIN FAILED $NAME"; exit 1; }
fi

for p in $OUT/iter_epoch_*.pdparams; do
  e=$(basename $p .pdparams); e=${e#iter_epoch_}
  inf=$OUT/inference_e$e; preds=$OUT/preds_e$e
  if [ ! -f $inf/inference.pdiparams ]; then
    python tools/export_model.py -c $CFG -o $COMMON \
      Global.pretrained_model=${p%.pdparams} Global.save_inference_dir=$inf \
      > $OUT.export_e$e.log 2>&1 || { log "EXPORT FAILED e$e"; continue; }
  fi
  [ "$(ls $preds 2>/dev/null | wc -l)" = 300 ] || \
    python $TGB/experiments/paddle_predict.py --model-name ta_PP-OCRv5_mobile_rec \
      --model-dir $inf --testset $TGB/testset --pred-dir $preds > $OUT.predict_e$e.log 2>&1
  python $TGB/experiments/score_preds.py $preds --label ${NAME}_e$e \
    --json $HOME/revision/results/x5_$NAME.json
done
log "done $NAME"
