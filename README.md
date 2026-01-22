***********************************************************************************************************

# Noise-Generating and Imaging Mechanism Inspired Implicit Regularization Learning Network for Low-Dose CT Reconstruction  
*(IEEE Transactions on Medical Imaging, 2024)*

## Overview

This repository provides the official implementation of the paper:

> **Noise-Generating and Imaging Mechanism Inspired Implicit Regularization Learning Network for Low-Dose CT Reconstruction**  
> *IEEE Transactions on Medical Imaging (TMI), 2024*

If you find this code useful for your research, please consider citing our paper.

---

## Citation

```bibtex
@article{li2023noise,
  title={Noise-generating and imaging mechanism inspired implicit regularization learning network for low dose ct reconstrution},
  author={Li, Xing and Jing, Kaili and Yang, Yan and Wang, Yongbo and Ma, Jianhua and Zheng, Hairong and Xu, Zongben},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={5},
  pages={1677--1689},
  year={2023},
  publisher={IEEE}
}
````

---

## Installation

This guide shows how to set up the environment using **conda**.

### 1. Clone the repository

```bash
git clone https://github.com/lixing0810/NGIM-IRL.git
cd NGIM-IRL
```

### 2. Create and activate the conda environment

```bash
conda create --name NGIM-IRL python=3.9
conda activate NGIM-IRL
```

### 3. Install PyTorch

```bash
pip install torch torchvision
```

> ⚠️ You may need to install the CUDA-specific version of PyTorch depending on your GPU setup.

### 4. Install required dependencies

```bash
pip install -r requirements.txt
```

---

## Training

To start training the model, simply run:

```bash
python train.py
```
