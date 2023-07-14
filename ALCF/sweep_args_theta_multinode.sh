#!/bin/bash -l

module load conda/2023-01-11
conda activate base
cd /home/czh5/seq/Megatron-DS-Benchmarking/ALCF
source /home/czh5/seq/Megatron-DS-Benchmarking/venvs/thetaGPU/2023-01-11-deepspeed/bin/activate

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "My current script is: ${SCRIPT_DIR[0]}"

NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
NGPU_PER_HOST=8
PARALLEL_SIZE=$((${NHOSTS}*${NGPU_PER_HOST}))

export MODEL_TYPE=${MODEL_TYPE:-"gpt"} # set bert or gpt
export SP_TYPE=${SP_TYPE:-"megatron"} # set ds or megatron

SEQLEN_VALS=(
#   2048
#   4096
  8192
  16384
  32768
  65536
#   131072
#   262144
#   524288
#   1048576
#   2097152
)

MODEL_SIZE_VALS=(
  # "GPT125M"
  # "BERT1.2B"
#   "GPT1_5B"
  # "GPT2_7B"
#   "GPT6_7B"
  # "GPT13B"
  # "GPT25B"
  "GPT30B"
)

for MODEL_SIZE_KEY in "${MODEL_SIZE_VALS[@]}"; do
  export MODEL_SIZE_KEY
  for SEQ_LEN in "${SEQLEN_VALS[@]}"; do

    export SEQ_LEN

    if [[ ${SP_TYPE} == "ds" ]]; then
      echo "DS sequence parallel"
      export SPSIZE=${PARALLEL_SIZE}
      export MPSIZE=1
      export ZERO_STAGE=3
      export USE_SEQUENCE_PARALLEL=0
      bash ./benchmark_train.sh
    fi

    if [[ ${SP_TYPE} == "megatron" ]]; then
      echo "Megatron's sequence parallel"

        if [ ${SEQ_LEN} -eq 8192 ]; then
            PARALLEL_SIZE=8
        fi

        if [ ${SEQ_LEN} -eq 16384 ]; then
            PARALLEL_SIZE=8
        fi

        if [ ${SEQ_LEN} -eq 32768 ]; then
            PARALLEL_SIZE=16
        fi

        if [ ${SEQ_LEN} -eq 65536 ]; then
            PARALLEL_SIZE=16
        fi

      export SPSIZE=1
      export MPSIZE=${PARALLEL_SIZE}
      export ZERO_STAGE=3
      export USE_SEQUENCE_PARALLEL=1
      bash ./benchmark_train.sh
    fi

    printf "\n------------------------"
    echo SEQ_LEN=${SEQ_LEN}
  done
done
