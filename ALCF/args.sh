#!/bin/bash -login

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
  SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
  SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
  [[ ${SCRIPT_PATH} != /* ]] && SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_PATH}"
done
SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
SCRIPT_DIR="$(cd -P "$(dirname -- "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

function sourceFile() {
  FILE="$1"
  echo "source-ing ${FILE}"
  if [[ -f "${FILE}" ]]; then
    # shellcheck source="${FILE}"
    source "${FILE}"
  else
    echo "ERROR: UNABLE TO SOURCE ${FILE}"
  fi
}

USER=$(whoami)

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname "$DIR")

echo "------------------------"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "SCRIPT_PATH=$SCRIPT_PATH"
echo "------------------------"
echo "SOURCE=$SOURCE"
echo "DIR=$DIR"
echo "------------------------"

export MODEL_SIZE_KEY="${MODEL_SIZE_KEY:-GPT13B}"
echo "+=========================+"
echo "| Using ${MODEL_SIZE_KEY}"
echo "+=========================+"

sourceFile "${DIR}/model.sh"

MODEL_TYPE=${MODEL_TYPE:-gpt}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Parallel / Pipeline Parallel ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ----------
# Originals
# MPSIZE=8
# PPSIZE=16
# ----------
# NHOSTS=$(wc -l < "${PBS_NODEFILE}")
export DDP_IMPL="local"   # FSDP | local | torch
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}  # 1 | 0
export USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
export SEQ_LEN=${SEQ_LEN:-1024}
export MPSIZE=${MPSIZE:-1}
export PPSIZE=${PPSIZE:-1}
export SPSIZE=${SPSIZE:-1}
export MICRO_BATCH=${MICRO_BATCH:-1}
export ZERO_STAGE=${ZERO_STAGE:-1}  # 0 | 1 | 2 | 3
export NHOSTS="$NHOSTS"
export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}
export USE_SEQUENCE_PARALLEL=${USE_SEQUENCE_PARALLEL:-0}  # 1 | 0

# NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
# # NGPU_PER_HOST=8
# NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
# PARALLEL_SIZE=$((${NHOSTS}*${NGPU_PER_HOST}))

export MODEL_TYPE=${MODEL_TYPE:-"gpt"} # set bert or gpt
export SP_TYPE=${SP_TYPE:-"megatron"} # set ds or megatron

if [[ ${SP_TYPE} == "ds" ]]; then
  echo "Using DS sequence parallel"
  export SPSIZE=${PARALLEL_SIZE:-1}
  export MPSIZE=1
  export ZERO_STAGE=3
  export USE_SEQUENCE_PARALLEL=0
elif [[ ${SP_TYPE} == "megatron" ]]; then
    echo "Megatron's sequence parallel"
      # if [ ${SEQ_LEN} -eq 8192 ]; then
      #     PARALLEL_SIZE=8
      # fi
      # if [ ${SEQ_LEN} -eq 16384 ]; then
      #     PARALLEL_SIZE=8
      # fi
      # if [ ${SEQ_LEN} -eq 32768 ]; then
      #     PARALLEL_SIZE=16
      # fi
      # if [ ${SEQ_LEN} -eq 65536 ]; then
      #     PARALLEL_SIZE=16
      # fi
    export SPSIZE=1
    export MPSIZE=${PARALLEL_SIZE:-${WORLD_SIZE}}
    export ZERO_STAGE=0
    export USE_SEQUENCE_PARALLEL=1
  else 
    echo "Unexpected SP_TYPE: ${SP_TYPE}"
    exit 1
  fi

echo "+..................................................+"
echo "| SP_SIZE: ${SPSIZE}"
echo "| MPSIZE: ${MPSIZE}"
echo "| ZERO_STAGE: ${ZERO_STAGE}"
echo "| USE_SEQUENCE_PARALLEL: ${USE_SEQUENCE_PARALLEL}"
echo "+..................................................+"
# if [ ${SEQ_LEN} -eq 8192 ]; then
#     MICRO_BATCH=4
# fi

# if [ ${SEQ_LEN} -eq 16384 ]; then
#     MICRO_BATCH=2
# fi

# if [ ${SEQ_LEN} -eq 32768 ]; then
#     MICRO_BATCH=1
# fi
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
# NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"
echo "NHOSTS * (NGPU / HOST) = $NHOSTS * $NGPU_PER_HOST = $NGPUS"

# GLOBAL_BATCH=$(( $NGPUS * $MICRO_BATCH * $GRADIENT_ACCUMULATION_STEPS ))
# GLOBAL_BATCH=$(( $GLOBAL_BATCH / $MPSIZE / $PPSIZE / $SPSIZE ))
GLOBAL_BATCH=$(( NGPUS * MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS ))
echo "GB = NGPUS * MB * GAS = ${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS} = ${GLOBAL_BATCH}"

GLOBAL_BATCH=$(( GLOBAL_BATCH / MPSIZE / PPSIZE / SPSIZE))
echo "GB = (NGPUS * MB * GAS) / (MP * PP * SP) = (${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS}) / (${MPSIZE} * ${PPSIZE} * ${SPSIZE}) = ${GLOBAL_BATCH}"
# GLOBAL_BATCH=1
export GLOBAL_BATCH="$GLOBAL_BATCH"

echo "--------------------------------"
echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "--------------------------------"


# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
# DATA_PATH=/lus/grand/projects/datascience/vsastry/genslm_subsample_200k_sequence_document/genslm_subsample_200k_sequence_document
DATA_DIR="${PARENT}/dataset"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"

# DATA_PATH="/home/czh5/genome/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document"
# VOCAB_FILE="/home/czh5/genome/Megatron-DeepSpeed/dataset/gpt2-vocab.json"
# MERGE_FILE="/home/czh5/genome/Megatron-DeepSpeed/dataset/gpt2-merges.txt"
# DATA_PATH="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/BookCorpusDataset_text_document"
# VOCAB_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/gpt2-vocab.json"
# MERGE_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_sp${SPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE}_${RUN_STR}"

if [[ $USE_FLASH_ATTN == 1 ]] ; then
  RUN_STR="flashAttn_${RUN_STR}"
fi
if [[ $DDP_IMPL == 'FSDP' ]]; then
  RUN_STR="FSDP_${RUN_STR}"
fi
if [[ $USE_ACTIVATION_CHECKPOINTING == 1 ]] ;then
  RUN_STR="actCkpt_${RUN_STR}"
fi
if [[ $USE_SEQUENCE_PARALLEL == 1 ]] ; then
  RUN_STR="SP_${RUN_STR}"
fi

RUN_STR="${MODEL_TYPE}_${RUN_STR}"

OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

export MODEL_SIZE="$MODEL_SIZE"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT TO: ${OUTPUT_DIR}"

# if [[ -z "${NVME_PATH}" ]]; then
#   echo "NVME_PATH: $NVME_PATH"
# else
#   if [[ $(hostname) == x* ]]; then
#     export NVME_PATH="/local/scratch/"
#   elif [[ $(hostname) == theta* ]]; then
#     export NVME_PATH="/raid/scratch/"
#   else
#     export NVME_PATH="/tmp/"
#   fi
# fi

# echo "NVME_PATH: ${NVME_PATH}"

if [[ $MODEL_TYPE == "gpt" ]] ; then
  DATA_LOAD_ARGS="--data-path $DATA_PATH --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE"
else
  DATA_LOAD_ARGS=""
fi

# Set to cpu for offloading to cpu for larger models
OFFLOAD_DEVICE="cpu"
CPU_OPTIM=" --cpu-optimizer"

# # Set to none and empty string for no cpu offloading
# OFFLOAD_DEVICE="none"  
# CPU_OPTIM=" "

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
DS_CONFIG=${PARENT}/ds_config-gpt.json

if [[ $ZERO_STAGE == "3" ]] ; then
cat <<EOT > "$DS_CONFIG"
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "train_batch_size" : $GLOBAL_BATCH,
  "steps_per_print": 1,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 1e9,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 5e7,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "initial_scale_power" : 12,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": true,
  "zero_allow_untested_optimizer": false,
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
  },
  "comms_logger": {
    "enabled": true,
    "verbose": false,
    "prof_all": false,
    "debug": false
  },
  "wandb": {
    "enabled": true,
    "group": "megatron-DS-rebase",
    "project": "megatron-DS-rebase"
  }
}
EOT
else
# "train_batch_size" : $GLOBAL_BATCH,
# 'offload_optimizer': 'cpu'
  # "train_batch_size" : $GLOBAL_BATCH,
# "offload_optimizer": {
#   "device": "cpu",
#   "nvme_path": "/raid/scratch/"
# }
#
# "optimizer": {
#    "type": "AdamW",
#    "params": {
#      "lr": 0.001,
#      "betas": [0.8, 0.999],
#      "eps": 1e-8,
#      "weight_decay": 3e-7
#    }
# },

cat <<EOT > "$DS_CONFIG"
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "steps_per_print": 1,
  "wall_clock_breakdown" : true,
  "zero_force_ds_cpu_optimizer": true,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
   "type": "WarmupLR",
   "params": {
     "warmup_min_lr": 0,
     "warmup_max_lr": 0.001,
     "warmup_num_steps": 1000
   }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
  },
  "comms_logger": {
    "enabled": true,
    "verbose": false,
    "prof_all": false,
    "debug": false
  },
  "wandb": {
    "enabled": true,
    "group": "megatron-DS-rebase",
    "project": "megatron-DS-rebase"
  }
}
EOT
fi

# "optimizer": {
#   "type": "OneBitAdam",
#   "params": {
#     "lr": 0.001,
#     "betas": [
#       0.8,
#       0.999
#     ],
#     "eps": 1e-8,
#     "weight_decay": 3e-7,
#     "freeze_step": 400,
#     "cuda_aware": false,
#     "comm_backend_name": "nccl"
#   }
# },
#
# "optimizer": "Adam",
# "optimizer": {
#   "type": "OneBitAdam",
#   "params": {
#     "lr": 0.001,
#     "betas": [
#       0.8,
#       0.999
#     ],
#     "eps": 1e-8,
#     "weight_decay": 3e-7,
#     "freeze_step": 400,
#     "cuda_aware": true,
#     "comm_backend_name": "nccl"
#   }
# },
#
#
# 'deepspeed_mpi': True,
# 'ds_pipeline_enabled': False,
# 'rank': 0,
# 'world_size': 1,
# 'transformer_pipeline_model_parallel_size': 1,
# 'data_parallel_size': 1,
# 'virtual_pipeline_model_parallel_ size': None,


# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
if [[ "$DDP_IMPL" != "FSDP" ]] ; then
  ds_args=""
  ds_args=" --deepspeed ${ds_args}"
  ds_args=" --deepspeed_mpi ${ds_args}"
  ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
  ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
  if [[ "$PPSIZE" == 1 ]]; then
    ds_args="--no-pipeline-parallel ${ds_args}"
  else
    ds_args=" --pipeline-model-parallel-size ${PPSIZE} ${ds_args}"
  fi
  if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
  fi
fi

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
# --sequence-parallel \
gpt_args=(
  "--seed ${RANDOM}"
  "--DDP-impl ${DDP_IMPL}"
  "--pipeline-model-parallel-size ${PPSIZE}"
  "--tensor-model-parallel-size ${MPSIZE}"
  "--sequence-parallel-size ${SPSIZE}"
  "--num-layers ${NLAYERS}"
  "--hidden-size ${HIDDEN}"
  "--num-attention-heads ${ATEN_HEADS}"
  "--micro-batch-size ${MICRO_BATCH}"
  "--global-batch-size ${GLOBAL_BATCH}"
  "--seq-length ${SEQ_LEN}"
  "--max-position-embeddings ${SEQ_LEN}"
  "--train-iters 5"
  "--lr-decay-iters 320000"
  "--num-workers 1"
  "$DATA_LOAD_ARGS"
  "--data-impl mmap"
  "--split 949,50,1"
  "--distributed-backend nccl"
  "--lr 0.00015"
  "--lr-decay-style cosine"
  "--min-lr 1.0e-5"
  "--weight-decay 1e-2"
  "--clip-grad 1.0"
  "--lr-warmup-fraction .01"
  "--log-interval 1"
  "--save-interval 1000"
  "--eval-interval 1000"
  "--eval-iters 1"
  "--override-opt_param-scheduler"
  "--tensorboard-dir ${TENSORBOARD_DIR}"
  "--log-timers-to-tensorboard"
  "--tensorboard-log-interval 1"
  "${ARG_ENABLE_SEQUENCE_PARALLEL}"
)
# --tensorboard-log-interval 1" \
# ${ARG_ENABLE_SEQUENCE_PARALLEL}

# --recompute-activations \
# --recompute-granularity full \
# --recompute-method uniform \
# --recompute-num-layers 1 \
if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
  gpt_args=(
    "--checkpoint-activations"
    "--checkpoint-num-layers 1"
    "${gpt_args[*]}"
  )
    # --distribute-checkpointed-activations \
fi

if [[ "$DDP_IMPL" != "FSDP" ]] ; then
  gpt_args=(
    "${gpt_args[*]}"
    "--fp16"
  )
else
  gpt_args=(
    "${gpt_args[*]}"
    "--bf16"
  )
fi

if [[ "$USE_FLASH_ATTN" == 1 ]] ; then
  gpt_args=(
    "${gpt_args[*]}"
    "--use-flash-attn"
  )
  # gpt_args="\
  #   --use-flash-attn \
  #   ${gpt_args}"
fi

if [[ "$USE_SEQUENCE_PARALLEL" == 1 ]]; then
  # gpt_args="\
  #   --sequence-parallel \
  #   ${gpt_args}"
  gpt_args=(
    "${gpt_args[*]}"
    "--sequence-parallel"
  )
fi
export gpt_args=(
  "${gpt_args[*]}"
  "$CPU_OPTIM" 
  "${ds_args}"
)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "gpt_args: ${gpt_args[*]}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
