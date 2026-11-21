#!/bin/bash
# One model on the GPU at a time: unload everything, then run.
cd ~/revision
for m in gemma3:4b qwen3-vl:8b-instruct; do
  d=$(echo $m | tr ":." "__")
  for x in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop $x; done
  sleep 3
  rm -rf vlm/preds/$d
  python3 vlm_predict.py --model $m --images tgb/testset/images --out vlm/preds/$d > vlm/run_$d.log 2>&1 \
    && echo "DONE $m $(tail -1 vlm/run_$d.log) | $(ollama ps | awk 'NR>1{print $1, $5, $6}' | tr '\n' ';')" \
    || echo "FAIL $m $(tail -3 vlm/run_$d.log | tr '\n' ' ')"
done
echo Q4DONE
