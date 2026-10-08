# LeMA: Scalable Levenberg-Marquardt Training Framework for Deep Learning

**LeMA** is a multi-GPU Levenberg-Marquardt training framework for deep learning.

## Get started



## Install from source

1. Install `uv` command: LeMA use [uv](https://github.com/astral-sh/uv) to manage dependencies and building process.
    ```bash
    conda install conda-forge::uv
    ```
2. Create and activate isolated environmnet for this project at `.venv`.
    ```bash
    uv sync
    source .venv/bin/activate
    ```
2. Build and install LeMA from the source codes
    ```bash
    uv pip install -e .
    ```


<!-- ## Get Started

### 1. Install
```bash
pip install --prev lema
```

## Hands-on Examples
* Train convolutional neural network on MNIST dataset: [experiments/mnist](experiments/mnist/)
* TODO: Train physics-informed neural network -->