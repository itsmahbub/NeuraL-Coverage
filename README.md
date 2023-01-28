# NeuraL-Coverage
Research Artifact of ICSE 2023 Paper: *Revisiting Neuron Coverage for DNN Testing: A Layer-Wise and Distribution-Aware Criterion*

Preprint: https://arxiv.org/pdf/2112.01955.pdf

## Implementations

This repo implements the **NLC** proposed in our paper and previous neuron coverage criteria (optimized if possible), including

- [x] Neuron Coverage (**NC**) [1]
- [x] K-Multisection Neuron Coverage (**KMNC**) [2]
- [x] Neuron Boundary Coverage (**NBC**) [2]
- [x] Strong Neuron Activation Coverage (**SNAC**) [2]
- [x] Top-K Neuron Coverage (**TKNC**) [2]
- [x] Top-K Neuron Patterns (**TKNP**) [2]
- [x] Cluster-based Coverage (**CC**) [3]
- [x] Likelihood Surprise Coverage (**LSC**) [4]
- [x] Distance-ratio Surprise Coverage (**DSC**) [5]
- [x]  Mahalanobis Distance Surprise Coverage (**MDSC**) [5]

Each criterion is implemented as one Python class in `coverage.py`.

[1] *DeepXplore: Automated whitebox testing of deep learning systems*, SOSP 2017.  
[2] *DeepGauge: Comprehensive and multi granularity testing criteria for gauging the robustness of deep learning systems*, ASE 2018.  
[3] *Tensorfuzz: Debugging neural networks with coverage-guided fuzzing*, ICML 2019.  
[4]  *Guiding deep learning system testing using surprise adequacy*, ICSE 2019.  
[5] *Reducing dnn labelling cost using surprise adequacy: An industrial case study for autonomous driving*, FSE Industry Track 2020.


## Installation

- Build from source code

    ```setup
    git clone https://github.com/Yuanyuan-Yuan/NeuraL-Coverage
    cd NeuraL-Coverage
    pip install requirements.txt
    ```

## Model & Dataset

- Pretrained models: please see [MODEL](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage/blob/main/MODEL.md).
- Datasets: please see [DATASET](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage/blob/main/DATASET.md).

## Getting Started

```python
import torch
# Implemented using Pytorch

import tool
import coverage

# 0. Get layer size in model
input_size = (1, image_channel, image_size, image_size)
random_input = torch.randn(input_size).to(device)
layer_size_dict = tool.get_layer_output_sizes(model, random_input)

# 1. Initialization
# `hyper` denotes the hyper-paramter of a criterion;
# set `hyper` as None if a criterion is hyper-paramter free (e.g., NLC).
criterion = coverage.NLC(model, layer_size_dict, hyper=None)
# KMNC/NBC/SNAC/LSC/DSC/MDSC requires training data statistics of the tested model,
# which is implemented in `build`. `train_loader` can be a DataLoader object in Pytorch or a list of data samples.
# For other criteria, `build` function is empty.
criterion.build(train_loader)

# 2. Calculation
# `test_loader` stores all test inputs; it can be a DataLoader object in Pytorch or a list of data samples.
criterion.assess(test_loader)
# If test inputs are gradually given (e.g., in fuzzing), then calculate the coverage as the following way.
for data in data_stream:
    criterion.step(data)

# 3. Result
# The following instruction assigns the current coverage value to `cov`.
cov = criterion.current
```


