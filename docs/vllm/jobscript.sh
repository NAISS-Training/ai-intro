#!/bin/bash -l
#SBATCH -A C3SE-STAFF
#SBATCH -t 1:00:00
#SBATCH -J vllm
#SBATCH --ntasks-per-node=1 --cpus-per-task=64 --nodes 2
#SBATCH --gpus-per-node "A40:4"

export HEAD_HOSTNAME="$(hostname)"
export HF_MODEL=/mimer/NOBACKUP/Datasets/LLM/huggingface/hub/models--neuralmagic--Llama-3.3-70B-Instruct-quantized.w8a8/snapshots/dc36722e6cb1e6b98d0144fd6059933d19c00ebf/
#export HF_MODEL=neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16
#export HF_HOME=/path/to/your/model/cache
export SIF_IMAGE=/apps/containers/vLLM/vllm-0.11.0.sif

# start ray cluster
export RAY_PORT=$(find_ports)
export RAY_CMD_HEAD="ray start --block --head --port=${RAY_PORT}"
export RAY_CMD_WORKER="ray start --block --address=${HEAD_HOSTNAME}:${RAY_PORT}"

srun -J "head ray node-step-%J" \
  -N 1 --tasks-per-node=1 -w ${HEAD_HOSTNAME} \
  apptainer exec ${SIF_IMAGE} ${RAY_CMD_HEAD} &
RAY_HEAD_PID=$!
sleep 10

srun -J "worker ray node-step-%J" \
  -N $(( SLURM_NNODES-1 )) --tasks-per-node=1 -x ${HEAD_HOSTNAME} \
  apptainer exec ${SIF_IMAGE} ${RAY_CMD_WORKER} &
RAY_WORKER_PID=$!
sleep 10

# start vllm
vllm_opts=(
    "--tensor-parallel-size=${SLURM_GPUS_ON_NODE}"
    "--pipeline-parallel-size=${SLURM_NNODES}"
    "--max-model-len=10000"
    "--distributed-executor-backend=ray"
)
export API_PORT=$(find_ports)

echo "Starting server"
apptainer exec ${SIF_IMAGE} vllm serve ${HF_MODEL} \
  --port ${API_PORT} "${vllm_opts[@]}" \
  > vllm-${SLURM_JOB_ID}.out  2> vllm-${SLURM_JOB_ID}.err &
VLLM_PID=$!
sleep 20

# wait at most 10 min for the model to start, otherwise abort
if timeout 600 bash -c "tail -f vllm-${SLURM_JOB_ID}.err | grep -q 'Application startup complete.'"; then
  echo "Starting client"
  apptainer exec ${SIF_IMAGE} python3 async_text_gen.py \
    > client-${SLURM_JOB_ID}.out 2> client-${SLURM_JOB_ID}.err
else
  echo "vLLM doesn't seem to start, aborting"
fi

echo "Terminating VLLM" && kill -15 ${VLLM_PID}
echo "Terminating Ray workers" && kill -15 ${RAY_WORKER_PID}
echo "Terminating Ray head" && kill -15 ${RAY_HEAD_PID}

