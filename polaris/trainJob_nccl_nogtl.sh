#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o train_nccl_nogtl.%j.out
#PBS -N ddp-nccl-nogtl

export REPO_ROOT=/home/mrest/FABRIC_ddp
cd "$REPO_ROOT" || exit 1
echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source "$REPO_ROOT/polaris/envScript_nccl.sh" || exit 1
echo "=== Starting training ==="
mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env ZFP_HOME="${ZFP_HOME}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    python interface.py

echo "=== Job finished: $(date) ==="
