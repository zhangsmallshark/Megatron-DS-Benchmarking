#!/bin/bash -l

module load conda/2023-01-11
conda activate base
cd /grand/projects/datascience/mtanaka/dsseq/Megatron-DS/ALCF
source /grand/projects/datascience/mtanaka/dsseq/venv/dsseq/bin/activate


NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
NGPU_PER_HOST=8
export SPSIZE=$((${NHOSTS}*${NGPU_PER_HOST}))

export MODEL_TYPE=bert
export ZERO_STAGE=0
export USE_SEQUENCE_PARALLEL=1

SEQLEN_VALS=(
  2048
  4096
  8192
  16384
  32768
  65536
  131072
  262144
  524288
  # 1048576
  # 2097152
)
 
SPSIZE_VALS=(
1
# 2
# 4
# 8
)

MPSIZE_VALS=(
# 1
# 2
# 4
# 8
16
)

MODEL_SIZE_VALS=(
  # "GPT125M"
  # "BERT1.2B"
  # "GPT1_5B"
  # "GPT2_7B"
  # "GPT6_7B"
  # "GPT13B"
  "GPT25B"
)

for MODEL_SIZE_KEY in "${MODEL_SIZE_VALS[@]}"; do
  export MODEL_SIZE_KEY
  for SEQ_LEN in "${SEQLEN_VALS[@]}"; do
    for SPSIZE in "${SPSIZE_VALS[@]}"; do
      for MPSIZE in "${MPSIZE_VALS[@]}"; do
          export SEQ_LEN
          export SPSIZE
          export MPSIZE
          bash ./benchmark_train.sh
      done
    done
  done
done
