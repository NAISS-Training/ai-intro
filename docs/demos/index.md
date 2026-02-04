# Demos
These notebooks and files are used to demonstrate certain concepts that are
especially relevant when it comes to running on HPC resources.

## Basics
The basics covers how to make sure your machine learning models run on the GPU
as well as how to use checkpointing. Using GPUs can give order of magnitude
speed-ups compared to CPU for the right workloads, while checkpointing is good
practice for anything long running and the only way to get around the maximum
allowed walltimes (which varies per resouce/partition but is commonly 7-days).

<!-- TODO link to the different demos -->
- [PyTorch](./basics_pytorch.html)

## Performance and Profiling
In these demos we investigate a script with profilers and monitoring to investigate some common bottlenecks and improvements.

Recommended is to run the scripts in an interactive job through the terminal.

## Parallelism
Multi-GPU parallelism is the next step when one GPU is too small:

- Case 1: Running each batch is very slow and/or you need to do batch accumulation to reach your wanted effective batch size without running out of memory.
  - Use data parallelism
- Case 2: Your model is too big to fit on the GPU even with a batch size of 1.
  - Use some flavour of model parallelism.

## Basic LLM inference

Remember to set `HF_HOME` to your project storage before downloading models. If you're using provided models, specify path to specific snapshots like e.g.

```bash
vllm serve "/mimer/NOBACKUP/Datasets/LLM/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1/"
```

