#!/bin/bash
# X10: recognise the 300 test lines with three models under two Tesseract builds.
set -e
cd ~/revision/x10
cp --update=none ~/revision/tessdata/tam.traineddata tessdata/
md5sum tessdata/*.traineddata > models.md5
declare -A BIN=( [v550]=$HOME/opt/tesseract-5.5.0/bin/tesseract [v552]=$HOME/opt/tesseract-5.5.2/bin/tesseract [v552dev]=/usr/local/bin/tesseract [v551]=$HOME/opt/tesseract-5.5.1/bin/tesseract )
for v in v551; do
  ${BIN[$v]} --version > version_$v.txt 2>&1
  for m in tam tamv3dcorpus tamv3i1400; do
    d=preds/$v/$m; mkdir -p $d
    ls ~/revision/tgb/testset/images/*.tif | xargs -P 20 -I{} bash -c '
      f={}; s=$(basename "$f" .tif)
      OMP_THREAD_LIMIT=1 TESSDATA_PREFIX='"$PWD"'/tessdata '"${BIN[$v]}"' "$f" stdout -l '"$m"' --psm 13 2>/dev/null | tr -s "[:space:]" " " | sed "s/^ //;s/ $//" > '"$d"'/$s.txt'
    echo "$v $m $(ls $d | wc -l)"
  done
done
