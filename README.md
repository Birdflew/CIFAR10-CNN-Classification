# CIFAR-10 Image Classification with a LeNet-style CNN

A PyTorch implementation of a classic LeNet-style convolutional neural network for
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification, including
loss visualization, regularization (L2 + Dropout), and a hyperparameter study.

## Overview

This project was completed as the final assignment for the course *Introduction to
Artificial Intelligence* (Fudan University, Fall 2024). It trains a small CNN
(`conv1 -> conv2 -> fc1 -> fc2 -> fc3`) on CIFAR-10 and explores how design choices
and hyperparameters affect classification performance.

### My contributions

The baseline network and training loop were provided as course materials. On top of
that baseline I completed the following tasks:

| Task | What I did |
| --- | --- |
| 1. Loss curve visualization | Recorded per-epoch training loss and plotted it with `matplotlib` |
| 2. Regularization | Added `nn.Dropout(0.5)` between `fc1` and `fc2`, and L2 regularization via SGD `weight_decay=0.01` |
| 3. Hyperparameter search | Swept `num_epochs ∈ {5,...,10}` × `learning_rate ∈ {0.001, 0.01}` and compared final loss / test accuracy |
| 4. Custom network | Left as a TODO extension |

The experiment report (in Chinese) is available under [`report/`](report/).

## Results

Best result: **58% test accuracy** with `num_epochs=9`, `learning_rate=0.001`
(corresponding checkpoint: `epoch_9_model.pth`).

| num_epochs | learning_rate | final_loss | test_accuracy |
| --- | --- | --- | --- |
| 5 | 0.001 | 1.328 | 54% |
| 5 | 0.01 | 1.978 | 25% |
| 6 | 0.001 | 1.294 | 55% |
| 6 | 0.01 | 1.959 | 27% |
| 7 | 0.001 | 1.264 | 54% |
| 7 | 0.01 | 1.973 | 27% |
| 8 | 0.001 | 1.270 | 52% |
| 8 | 0.01 | 1.965 | 25% |
| **9** | **0.001** | **1.219** | **58%** |
| 9 | 0.01 | 1.965 | 25% |
| 10 | 0.001 | 1.257 | 57% |
| 10 | 0.01 | 1.965 | 30% |

Key observations:
- `learning_rate=0.01` is too high — the model fails to converge (loss stays ≈1.96, accuracy 25–30%).
- With `learning_rate=0.001`, more epochs consistently lower the loss; test accuracy peaks at ~58% around epoch 9.

## How to run

```bash
# 1. Install dependencies (Python 3.8+)
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook cifar10_image_classification.ipynb
```

The first run downloads CIFAR-10 automatically into `./dataset` (torchvision), so no
manual data download is needed.

## Project structure

```
CIFAR10-CNN-Classification/
├── cifar10_image_classification.ipynb   # main notebook (training + tasks 1-3)
├── epoch_9_model.pth                    # best checkpoint (58% accuracy)
├── report/
│   └── CIFAR-10_实验报告.pdf             # experiment report (Chinese)
├── requirements.txt
└── .gitignore
```

## Disclaimer

Educational project. The baseline code was provided by the course instructor; the
completed tasks and report are the author's own work. CIFAR-10 is
[licensed](https://www.cs.toronto.edu/~kriz/cifar.html) by the Canadian Institute for
Advanced Research and is used here for academic purposes.
