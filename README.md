# Horseshoe Splatting: Handling Structural Sparsity for Uncertainty-Aware Gaussian-Splatting Radiance Field Rendering

This repository contains the official open-source implementation of the paper "Horseshoe Splatting: Handling Structural Sparsity for Uncertainty-Aware Gaussian-Splatting Radiance Field Rendering". We introduce Horseshoe Splatting, a Bayesian extension of 3D Gaussian Splatting (3DGS) that jointly addresses structured sparsity in per-splat covariances and delivers calibrated uncertainty.

## Requirements

**Hardware Requirements**

CUDA-ready GPU with Compute Capability 7.0+

**Software Requirements**

Conda (recommended for easy setup)

C++ Compiler for PyTorch extensions

CUDA SDK 11 for PyTorch extensions

C++ Compiler and CUDA SDK must be compatible

## Usage

### Cloning the Repository

Please clone with submodules (The repository will be public after the paper is accepted)
```shell
# SSH
git clone git@github.com:KaedeGo/hs-gs.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/KaedeGo/hs-gs.git --recursive
```

### Setup

We provide conda environment file to creat experiment environment: 
```shell
conda env create --file environment.yml
conda activate hs_splatting
```
We test our code on ubuntu system, please refer to original 3DGS repo about the potential error building the environment or running on windows. 

### Preparing Dataset

The LF dataset and LLFF dataset files are provided here: [LF dataset](https://drive.google.com/file/d/1RrfrMN5wSaishYJu5vYiTy6gUPZfLaDM/view?usp=sharing), [LLFF dataset](https://drive.google.com/file/d/1kDclWpEpUPm9Nw0tGoQTLWz3L4g5Hu2L/view?usp=sharing). 

Please unzip and put them under the a dataset folder: 

```bash
├──dataset
│   │  
│   ├──── LF
│   └──── nerf_llff_data
```

### Running

To train and evaluate the image quality and the image/depth uncertainty on LF dataset: 

```shell
sh scripts/train_render_lf.sh
```

To train and evaluate the image quality and image uncertainty quality on LLFF dataset: 

```shell
sh scripts/train_render_llff.sh
```

To perform the active training on LLFF dataset: 

```shell
sh scripts/active_llff.sh
```
