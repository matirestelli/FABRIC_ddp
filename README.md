# FABRIC: Framework for Architecture, Backend, and Reduction Interchange Comparison

FABRIC is a PyTorch DistributedDataParallel (DDP) framework for evaluating how communication backends and all-reduce algorithms affect distributed training. It lets you compare the built-in gradient synchronization path, custom communication hooks, and compressed algorithms within a shared training workflow.

The framework builds on the poster *Evaluating All-Reduce Operations in DDP Training with FABRIC* and earlier EuroMPI work on GPU-based compression for MPI AllReduce. It supports experiments across GPU systems, communication backends, model workloads, and scaling configurations.

## How FABRIC works

During distributed training, DDP groups gradients into buckets and synchronizes them across ranks. FABRIC lets you select the workload and communication path used for that synchronization:

| Component | Choices |
| --- | --- |
| ML workload | Model, dataset, optimizer, batch size, and training parameters |
| Training loop | DDP forward pass, backward pass, and gradient synchronization |
| Backend | MPI or PyTorch's `nccl` backend (NCCL on NVIDIA; RCCL on AMD) |
| Communication | DDP's built-in path or a registered communication hook |

Job scripts set experiment parameters and launch the training entry point. `interface.py` reads job environment overrides and passes the configuration to the training loop. A hook can implement a different synchronization algorithm for each gradient bucket.

## What you can do

- Compare the built-in DDP synchronization path with custom ring, recursive-doubling, and other registered hooks.
- Run compatible ZFP-compressed hooks when the appropriate GPU extension and libraries are installed.
- Test MPI, NCCL, or RCCL communication using the configuration appropriate to the machine.
- Study strong or weak scaling by changing node count, ranks, batch sizes, and gradient accumulation.
- Run image-classification workloads or nanoGPT-style language-model training, depending on the models and datasets available in your checkout.
- Add a Python communication hook, or test a collective implemented inside an MPI library through the built-in DDP path.

Available hooks and backend combinations depend on the checkout. After loading an appropriate environment, inspect the registry with:

```bash
python -c 'from communication_strategy import list_available_hooks; print(list_available_hooks())'
```

For a fair performance comparison, keep the model, data, training parameters, node count, and relevant software configuration consistent between runs. End-to-end epoch and iteration times show the effect on training; hook timing can help analyze communication work. The built-in `none` path has no custom hook to time.

## Repository overview

```text
.
├── interface.py                 # Experiment entry point and environment overrides
├── config.py                    # Training configuration, models, and data loaders
├── ddp_training.py              # DDP training, logging, and timing
├── communication_strategy.py    # Communication hooks and registry
├── nanogpt_model.py             # nanoGPT-style model
├── prepare_tokens.py            # Prepare text datasets for the language model
├── zfp_api.py                   # ZFP integration
├── polaris/                     # Polaris PBS jobs and environments
├── frontier/                    # Frontier Slurm jobs and environments
├── utilities/                   # Auxiliary checks and build helpers
└── data                         # Data directory or symbolic link
```

The repository may also contain build artifacts, checkpoints, logs, and system-specific ZFP installations. The `data` link can point outside the repository; verify that its target is accessible on the system running the job.

## Configure experiments

Select a system-specific environment and job script from `polaris/` or `frontier/`. Set the scheduler resources, model and dataset, backend, and experiment list in the job script. Jobs can override defaults defined in `interface.py`.

In the supplied scripts, experiment entries use `algorithm:rate`. For example:

```bash
BACKENDS=("mpi")
EXPERIMENTS=("none:" "ring:")
```

`none:` uses the built-in DDP synchronization path with no custom hook. `ring:` selects a registered ring hook. A compressed hook uses a rate after the colon, for example `ring_zfp_naive:8`, and needs the matching ZFP installation. Check the registry and job script in your checkout before selecting an algorithm.

Polaris jobs use PBS (`qsub`); Frontier jobs use Slurm (`sbatch`). Set the appropriate account, queue, paths, and compatible GPU/MPI environment before submitting. See the focused guide below for a Polaris nanoGPT/MPICH example.

<details>
<summary><strong>Polaris guide: testing your own MPICH all-reduce with nanoGPT</strong></summary>

## 1. Select your MPICH and the shared PyTorch build

The main files are:

- `polaris/envScript_Polaris_mpich.sh`: loads the software environment and selects the MPICH and PyTorch libraries.
- `polaris/trainJob_Polaris_mpich.sh`: selects the workload, experiment, process count, and PBS resources.
- `interface.py`, `config.py`, `ddp_training.py`, and `nanogpt_model.py`: configure and run the workload.
- `communication_strategy.py`: contains the Python communication hooks and their registry.

In `polaris/envScript_Polaris_mpich.sh`, find section **CUDA-aware upstream MPICH** and change `MPI_HOME` to your MPICH installation prefix:

```bash
export MPI_HOME="/path/to/your/mpich/install"
MPI_LIB="${MPI_HOME}/lib/libmpi.so.12"
```

Check the real library name and directory in your installation. The script checks specifically for `lib/libmpi.so.12`; adjust `MPI_LIB` and the later `LD_LIBRARY_PATH` if your build uses a different location or SONAME. The SONAME and ABI must be compatible with the MPI library against which PyTorch was built: changing a filename alone does not make incompatible libraries compatible. MPICH must support the GPU buffers and process manager/launcher used by this job. The existing configuration uses CUDA, the CXI provider and Polaris PALS; review its module, `FI_PROVIDER`, GPU-support, and launcher settings against your build. A build configured with its own process manager may need different launch settings.

The environment script currently points to an existing MPI-enabled PyTorch runtime:

```bash
PYTORCH_BUILD="/lus/eagle/projects/UIC-HPC/mrest/mpi-experiments/pytorch-runtime-upstream"
```

You can leave this path unchanged if you have permission to read it and your MPICH is binary compatible with that PyTorch build. Both its MPI backend and GPU-aware operation need to work with your MPICH; an MPI-enabled build alone does not establish that compatibility. Ask the owner for access if needed. If you cannot use it, build PyTorch from source with MPI support enabled against an appropriate MPI installation and update `PYTORCH_BUILD` and its associated Python/library paths. Confirm its CUDA and Python dependencies also match the runtime.

The script puts `${MPI_HOME}/bin` ahead of `/opt/cray/pals/1.8/bin` in `PATH`. Therefore `mpiexec` may resolve to an MPICH launcher if your installation supplies one; verify it is compatible with your PBS/PALS allocation instead of assuming it uses PALS.

After updating the script, from the repository root on a Polaris login node, inspect the resolved installation:

```bash
export REPO_ROOT="$PWD"
source "$REPO_ROOT/polaris/envScript_Polaris_mpich.sh"
command -v mpiexec
python -c 'import torch; import torch.distributed as d; print(torch.__file__); print("MPI available:", d.is_mpi_available())'
ldd "$PYTORCH_BUILD/torch/lib/libtorch_cpu.so" | grep -i mpi
```

`is_mpi_available()` confirms only that PyTorch was built with MPI. Check the actual resolved library and run a small multi-rank GPU job to verify launcher, ABI, and GPU communication. Login-node CUDA availability is not a compute-node test. If sourcing fails, fix the reported dependencies before continuing.

The job script must set its own absolute repository root and source the matching environment script:

```bash
export REPO_ROOT=/absolute/path/to/your/repository
cd "$REPO_ROOT" || exit 1
source "$REPO_ROOT/polaris/envScript_Polaris_mpich.sh" || exit 1
```

Ensure the environment script uses `PROJECT_ROOT="${REPO_ROOT:-$(pwd)}"`, so it does not accidentally treat the submission directory as the repository root.

## 2. Prepare TinyStories tokens and nanoGPT

The nanoGPT-style model is implemented in `nanogpt_model.py`; a separate nanoGPT checkout or downloaded model weights are not required for the supplied from-scratch configuration. The job's `MODEL_NAME` selects the architecture, for example `gpt2_12x12x768`.

The tokenizer reads TinyStories and writes `train.bin` and `val.bin` under `DATA_DIR/tinystories/`. If accessible token files already exist, reuse them. Otherwise, on a login node with dataset access and enough shared storage, use the desired Python environment and install the preparation dependencies according to your environment's policies (`numpy`, `datasets`, and `tiktoken`). Then, from the repository root:

```bash
python prepare_tokens.py --dataset tinystories \
  --num-proc 8 --out /path/to/shared/data/tinystories
ls -lh /path/to/shared/data/tinystories/{train,val}.bin
```

For a small smoke-test subset, add `--max-docs 10000` and use a separate output directory. Tokenization downloads the corpus and GPT-2 tokenizer files; it requires network access or a populated cache and sufficient disk space. All compute nodes must be able to read the output. The `openwebtext` dataset is also supported, but is much larger.

In `polaris/trainJob_Polaris_mpich.sh`, set the parent directory, not the `tinystories` subdirectory:

```bash
DATA_DIR="/path/to/shared/data"
MODELS=("gpt2_12x12x768")
DATASETS=("tinystories")
```

The supplied script points to the owner's old data directory; keep it only if it exists and you have access. Training reads `<DATA_DIR>/tinystories/train.bin` and `val.bin`.

## 3. Configure and run experiments

Edit `polaris/trainJob_Polaris_mpich.sh`. Its PBS resource request and the values of `NUM_PROCS` and `PPN` must agree. For example, four Polaris nodes with four ranks per node require `select=4`, `NUM_PROCS=16`, and `PPN=4`. Choose an appropriate queue, allocation and walltime. For the accumulation setup in the script, `GRAD_ACCUM_GLOBAL` must be divisible by `NUM_PROCS`.

Select the MPI backend:

```bash
BACKENDS=("mpi")
```

To test an all-reduce **implemented inside MPICH**, select:

```bash
EXPERIMENTS=("none:")
```

This selects no custom DDP communication hook. DDP uses the PyTorch MPI backend for gradient synchronization, so the collective reaches the MPI library. You must still configure or instrument your MPICH implementation to verify that its intended all-reduce algorithm is actually selected. MPI may choose an algorithm dynamically based on message size, communicator and runtime settings.

Submit with:

```bash
qsub /absolute/path/to/your/repository/polaris/trainJob_Polaris_mpich.sh
```

Inspect the PBS output and error logs for the resolved PyTorch/MPI settings, job status and training results. Output paths in PBS directives can depend on scheduler configuration and submission location. Start with a short small allocation, and compare correctness with a known baseline before running larger tests.

## 4. Add a Python DDP communication hook

Use this route if the algorithm itself should execute as Python/PyTorch hook logic. An algorithm implemented only in MPICH does not need a new Python hook: use `none:` above. A Python hook may call distributed operations supplied by the MPI backend, but it does not automatically call an arbitrary new C function in MPICH; that requires an appropriate binding or an MPI operation that selects the implementation.

In `communication_strategy.py`:

1. Implement a hook following an existing non-ZFP example, such as `_default_allreduce_hook` or `_ring_allreduce_hook`.
2. The hook receives `(state, bucket)`, accesses the gradient tensor through `bucket.buffer()`, and returns a `torch.futures.Future` resolving to the reduced tensor. DDP does not average for a custom hook: ensure the result has the correct world-size averaging and tensor shape, and respect synchronization, device and collective-order requirements on every rank.
3. Add it to `_HOOK_REGISTRY`, for example:

   ```python
   ("mpi", "my_allreduce"): my_allreduce_hook,
   ```

4. In `get_comm_hook`, add a state-construction branch if the algorithm needs state. The reference copy defaults to `state = None` for unrecognized state types; the registry entry must still exist. Keep any necessary lifetime and asynchronous completion handling correct.
5. In the job script, select it with:

   ```bash
   BACKENDS=("mpi")
   EXPERIMENTS=("my_allreduce:")
   ```

The colon separates the algorithm name from the optional ZFP rate; leave the rate empty for a non-ZFP hook. Test numerical correctness across ranks against a trusted all-reduce with representative tensor sizes before interpreting performance. The existing `default:` hook is a useful baseline, but it is a registered hook and has different instrumentation/behavior from the no-hook `none:` path.

## 5. If you do not need ZFP

Use `none:` or a non-ZFP hook such as `ring:` and do not build or select a `*_zfp_*` experiment. In the reference Python code, the ZFP import is guarded, so an unavailable ZFP module should not prevent the non-ZFP paths from being selected. Verify that this is also true in your checkout. The supplied environment script still adds ZFP directories to environment variables; those settings can be removed or commented out if the rest of your environment does not require them. Keep the `PROJECT_ROOT` assignment, because the PyTorch `PYTHONPATH` uses it. Check the resulting Python imports and job on Polaris before relying on the setup.

</details>
