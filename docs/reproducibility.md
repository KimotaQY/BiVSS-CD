# Reproducibility environment

The paper results were reproduced with the following verified runtime:

| Component | Verified value |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| CUDA runtime used by PyTorch | 12.6 |
| cuDNN | 9.5.1 |
| Pillow | 12.0.0 |
| libjpeg reported by Pillow | 6.2 |
| NumPy | 1.26.4 |
| timm | 1.0.22 |
| GPU | NVIDIA L20 |
| NVIDIA driver | 580.95.05 |

`nvidia-smi` reports CUDA 13.0 on the verified server. This is the maximum CUDA
version supported by the installed driver, not the CUDA runtime bundled with
PyTorch. The verified PyTorch wheel uses CUDA 12.6.

Inference is sensitive to the software environment. In particular, Pillow and
libjpeg encode the two temporary JPEG frames consumed by SAM3. A different
encoding can alter predictions close to the configured detection thresholds.
PyTorch, CUDA, cuDNN, and GPU differences can also introduce small numerical
changes.

## Installation

Create the verified environment and install BiVSS-CD and its pinned SAM3 fork:

```bash
conda env create -f environment-reproduction.yml
conda activate bivss-cd
pip install -e .
pip install -e third_party/sam3
```

If another editable SAM3 checkout is already installed in the environment,
remove it before installing the pinned submodule. Confirm the imported path:

```bash
python -c "import sam3; print(sam3.__file__)"
```

It must point to `BiVSS-CD/third_party/sam3/sam3/__init__.py`. As a temporary,
non-destructive alternative for an existing environment, place the pinned
checkout first on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/third_party/sam3${PYTHONPATH:+:$PYTHONPATH}"
```

## Runtime record

Every evaluation writes `run_manifest.json`. Keep this file with reported
results: it records the imported SAM3 path and revision, checkpoint SHA-256,
Pillow/libjpeg, PyTorch/CUDA, and detected GPUs. Use a new output directory or
pass `--force` after changing the environment or inference implementation.

The model checkpoint used for the verified experiments has SHA-256:

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```
