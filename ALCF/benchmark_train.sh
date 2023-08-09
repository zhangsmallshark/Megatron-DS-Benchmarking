#!/bin/bash --login

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
# DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
#

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep pretrain_gpt.py | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi


SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )


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

SETUP_FILE="${DIR}/setup.sh"
MODEL_FILE="${DIR}/model.sh"
ARGS_FILE="${DIR}/args.sh"
LAUNCH_FILE="${DIR}/launch.sh"


sourceFile "${SETUP_FILE}"
sourceFile "${MODEL_FILE}"
sourceFile "${ARGS_FILE}"
sourceFile "${LAUNCH_FILE}"


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "My current script is: ${SCRIPT_DIR[0]}"

NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
# NGPU_PER_HOST=8
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
PARALLEL_SIZE=$(( NHOSTS * NGPU_PER_HOST ))

export MODEL_TYPE=${MODEL_TYPE:-"gpt"} # set bert or gpt
export SP_TYPE=${SP_TYPE:-"megatron"} # set ds or megatron

echo "+-----------------------------+"
echo "| MODEL TYPE: ${MODEL_TYPE}"
echo "| SP TYPE: ${SP_TYPE}"
echo "+-----------------------------+"

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

export MODEL_TYPE=${MODEL_TYPE:-gpt}

setup
# singleGPU "$@" 2>&1 &
# fullNode "$@" 2>&1 &
TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
export TORCH_VERSION=$TORCH_VERSION
export CUDA_DEVICE_MAX_CONNECTIONS=1
# fullNode "$@"
# elasticDistributed "$@" 2>&1 &
# elasticDistributed "$@"
PID=$!
wait $PID
