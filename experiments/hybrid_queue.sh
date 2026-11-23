#!/bin/bash
# Runs behind Online Resource Table F16 (desktop, ~/revision). a, b, c ran in
# parallel; the real-only control (d) ran afterwards, three seeds in parallel.
# The first real-only run (tam_real) used an unshuffled list, so its seeds
# were identical; it is superseded by tam_realsh_s0..s2 and not reported.
cd $HOME/revision
case $1 in
 a) ./run_hybrid.sh tam_hyb_s0 tam_x9 0 25 ;;
 b) ./run_hybrid.sh tam_hyb_s1 tam_v3_s1 1 25 ;;
 c) ./run_hybrid.sh tam_hyb_s2 tam_v3_s2 2 25 ;;
 d) for k in 0 1 2; do REAL_ONLY=1 ./run_hybrid.sh tam_realsh_s$k - $k 1 & done; wait ;;
esac
