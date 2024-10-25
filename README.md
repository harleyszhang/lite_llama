# Pytriton_llama

The llama model inference lite framework by tirton.

## GPU Information

趋动云 GPU 开发环境

```bash
/code# nvidia-smi
Using config file: /etc/orion/env/env.conf
+--------------------------------------------------------------------------------------------+
| ORION-SMI 1.0             Time: 2024-10-11 20:36:48            CUDA Version: N/A           |
+-----------------------------------------------+----------------------+---------------------+
| IP               vGPU Name       Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC|
| pGPU  vGPU       Physical GPU Name            |         Memory-Usage | GPU-Util  Compute M.|
|===============================================+======================+=====================|
| 0.0.0.0          Orion vGPU              Off  |   N/A            Off |                 N/A |
|  0     0         B1.gpu.small                 |      0MiB /  6062MiB |      0%     Default |
+--------------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------------+
| Processes:                                                                     vGPU Memory |
| IP               pGPU  vGPU   PID      Type  Process name                         Usage    |
|============================================================================================|
|  No running processes found                                                                |
+--------------------------------------------------------------------------------------------+
```

cuda 版本：
```bash
# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Python 3.11.8 包版本:

```bash
torch                          2.1.2
triton                         2.1.0
triton-nightly                 3.0.0.post20240716052845
```

## Some triton kernels repositories

- https://github.com/ELS-RD/kernl/tree/main
- https://github.com/unslothai/unsloth/tree/main
