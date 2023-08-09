#!/bin/bash --login

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )


#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep pretrain_gpt.py | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

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

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
LAUNCH_FILE="${DIR}/launch.sh"

sourceFile "${DIR}/setup.sh"
sourceFile "${DIR}/model.sh"
sourceFile "${DIR}/args.sh"
sourceFile "${LAUNCH_FILE}"

setup
# singleGPU "$@" 2>&1 &
# fullNode "$@" 2>&1 &
TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
export TORCH_VERSION=$TORCH_VERSION
export CUDA_DEVICE_MAX_CONNECTIONS=1
# elasticDistributed "$@" 2>&1 &
# elasticDistributed "$@"
# PID=$!
# wait $PID
elasticDistributed "$@" 2>&1 &
