---
title: "Running AI/ML workloads on NAISS systems"
---


## Scope

<!-- At some point cover the specifics for Alvis: -->
<!--    - C3SE_quota, where-are-my-files -->
<!--    - job-killing -->
<!--    - job_stats.py -->

- Will be covered:
    - Introduction for running Deep Learning workloads on the main NAISS AI/ML resource
- Will **not** be covered:
    - A general introduction to machine learning
    - Running classical ML or GOFAI
    - General HPC intro

## NAISS GPU resources overview
1. **Alvis** (End-of-life 2026-06-30)
    - NVIDIA GPUs: 332 A40s, 318 A100s, 160 T4s, 44 V100s
    - Only for AI/ML
2. Arrhenius (to be in operation spring 2026)
    - 1528 NVIDIA GH200s
3. Dardel
    - 248 AMD MI250X
4. Bianca
    - 20 NVIDIA A100s
    - Only for sensitive data

## Alvis specifics
- What is potentially different on Alvis?
- See extended version at [Alvis introduction material](https://www.c3se.chalmers.se/documentation/first_time_users/intro-alvis/slides/)

<style>
img.cluster_sketch{
  max-width: 80%;
}
</style>

![The cluster environment](cluster_sketch.png){.cluster_sketch}

### Connecting - Firewall

![Cluster firewall only allows connection from within SUNET](cluster_firewall.png){.cluster_sketch}

- Firewall limits connections to within
  [SUNET](https://www.sunet.se/om-sunet/anslutna-organisationer)
- Use a [VPN](https://www.c3se.chalmers.se/documentation/connecting/#vpn) if needed

### Log-in nodes

- `alvis1.c3se.chalmers.se` has 4 T4 GPUs for light testing and debugging
- `alvis2.c3se.chalmers.se` is dedicated data transfer node
- Will be restarted from time to time
- Login nodes are shared resources for all users:
    - don't run jobs here,
    - don't use up too much memory,
    - preparing jobs and
    - light testing/debugging is fine

### SSH - Secure Shell

- `ssh <CID>@alvis1.c3se.chalmers.se`, `ssh <CID>@alvis2.c3se.chalmers.se`
- Gives command line access to do anything you could possibly need
- If used frequently you can set-up a password protected
  [SSH-key](https://www.c3se.chalmers.se/documentation/connecting/ssh/#setting-up-ssh-key) for convenience

### Alvis Open OnDemand portal

- <https://alvis.c3se.chalmers.se>
- Browse files and see disk and file quota
- Launch interactive apps on compute nodes
    - Desktop
    - Jupyter notebooks
    - MATLAB proxy
    - RStudio
    - VSCode
- Launch apps on log-in nodes
    - TensorBoard
    - Desktop
- See our [documentation](https://www.c3se.chalmers.se/documentation/connecting/ondemand/) for more

### Remote desktop

- RDP-based remote desktop solution on shared login nodes (use portal for
  heavier interactive jobs)
- In-house-developed web client:
    - <https://alvis1.c3se.chalmers.se>
    - <https://alvis2.c3se.chalmers.se>
- Can also be accessed via desktop clients such as Windows Remote Desktop
  Connection (Windows), FreeRDP/Remmina/krdc (Linux) and Windows App (Mac) at
  <alvis1.c3se.chalmers.se> and <alvis2.c3se.chalmers.se> (standard port 3389).
- Desktop clients tend to offer better quality and more ergonomic experiences.
- See
  [the documentation](https://www.c3se.chalmers.se/documentation/connecting/remote_graphics/#rdp-support)
  for more details

### Files and Storage
- Cephyr `/cephyr/`, and Mimer `/mimer/` are parallel filesytems, accessible
  from all nodes
- Backed up home directory at `/cephyr/users/<CID>/Alvis`
  ([alt. use `~`](https://www.gnu.org/software/bash/manual/html_node/Tilde-Expansion.html))
- Project storage at `/mimer/NOBACKUP/groups/<storage-name>`
- The `C3SE_quota` shows you all your centre storage areas, usage and quotas.
    - On Cephyr see file usage with `where-are-my-files`
- File-IO is usually the limiting factor on parallel filesystems
- If you can deal with a few large files instead of many small, that is
  preferable

### Datasets

- When allowed, we provide popular datasets at `/mimer/NOBACKUP/Datasets/`
- To request an additional dataset, do so through the [support form](https://supr.naiss.se/support/?problem_type=other&centre_resource=r75)
- It is your responsibility to make sure you comply with any licenses and limitations
    - In all cases only for non-commercial research applications
    - Citation often needed
- Read more on the [dataset](https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/) page and/or the respective README files

### Software
- [Containers](https://www.c3se.chalmers.se/documentation/miscellaneous/containers/) through Apptainer
- Optimized software in [modules]
    - Flat module scheme, load modules directly
- Read the [Python instructions](https://www.c3se.chalmers.se/documentation/module_system/python/) for installing your own Python packages

### GPU hardware details

<!-- This table is also used in about/Alvis -->

| #GPUs | GPUs    | Capability | CPU     | Note       |
|-------|---------|------------|---------|------------|
| 44    | [V100](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf)    | 7.0        | Skylake |            |
| 160   | [T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)      | 7.5        | Skylake |            |
| 332   | [A40](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)     | 8.6        | Icelake | No IB      |
| 296   | [A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)    | 8.0        | Icelake | Fast Mimer |
| 32    | [A100fat](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) | 8.0        | Icelake | Fast Mimer |

### SLURM specifics
- Main allocatable resource is `--gpus-per-node=<GPU type>:<no. gpus>`
    - e.g. `#SBATCH --gpus-per-node=A40:1`
- Cores and memory are allocated proportional to number of GPUs and related node type
- Maximum 7 days walltime
    - Use checkpointing for longer runs
- Jobs that don't use allocated GPUs may be automatically cancelled

### GPU cost on Alvis

| Type    | VRAM | System memory per GPU | CPU cores per GPU | Cost |
| ------- | ---- | --------------------- | ----------------- | ---- |
| T4      | 16GB | 72 or 192 GB          | 4                 | 0.35 |
| A40     | 48GB | 64 GB                 | 16                | 1    |
| V100    | 32GB | 192 or 384 GB         | 8                 | 1.31 |
| A100    | 40GB | 64 or 128 GB          | 16                | 1.84 |
| A100fat | 80GB | 256 GB                | 16                | 2.2  |

- Example: using 2xT4 GPUs for 10 hours costs 7 "GPU hours" (2 x 0.35 x 10).
- The cost reflects the actual price of the hardware (normalised against an A40
  node/GPU).

### Monitoring tools
- You can SSH to nodes where you have an ongoing job
    - From where you can use CLI tools like `htop`, `nvidia-smi`, `nvtop`, ...
- Use `job_stats.py <JOBID>` to view graphs of usage
- `jobinfo -s` can be used to get a summary of currently available resources

## Running ML
- How you run on GPUs with PyTorch and TensorFlow

### PyTorch
- Move tensors or models to the GPU "by hand"

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.Tensor([1, 1, 2, 3]).to(device)
```

### TensorFlow
- Automatically tries to use a single GPU
- Will also pre-allocate GPU memory, obfuscating actual memory usage to external monitoring tools
- <https://www.tensorflow.org/guide/gpu>

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```

### Demos
- PyTorch: TBD
- TensorFlow: TBD

<!--    - Software recap -->
<!--    - SLURM recap and allocating GPUs on NAISS SLURM clusters -->
<!--    - Basic checkpointing for long running jobs (Do I want PyTorch lightning?, Yes, use it for this example at least) -->

## Performance and GPUs
- Floating point precision and GPU performance
<!--    - Explain GPU-parallelism -->
<!--    - Table for NVIDIA GPUs on all NAISS systems (optionally AMD table for Dardel) -->
<!--    - Mixed precision and how to select precision in PyTorch and TensorFlow -->
<!--    - GPU monitoring (nvtop, job_stats.py (Alvis only)) -->

## Performance and parallel filesystems
- Performance considerations for data loading on parallel filesystems
<!--    - Explain parallel filesystems or at least talk about FileIO -->
<!--    - What to show? Arrow, Zip -->

## Profiling
- Profiling your ML workload
<!--    - Print statements with timestamps? -->
<!--    - Figure out how PyTorchs new replacement work and get something useful from that -->
<!--    - TensorFlow + TensorBoard -->
<!--    - Scalene? Others? -->

## Multi-GPU parallelism
- Multi-GPU parallelism
    - Conceptual overview
    - Data Parallellism
    - Fully Sharded Data Parallel
<!--- Basics on HTC and random parameter search with job-arrays if time permits -->

## Basic LLM inference
- vLLM
<!--    - find_ports  -->
<!--    - chat mode  -->
<!--    - batch mode  -->
<!--    - model parallelism -->
<!--    - for more recommend NAISS LLM Workshop -->
