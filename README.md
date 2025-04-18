# <a id="title"> Mitigating Negative Flips in Model Updates by Compatible Learning Representation
</a>

This repository contains the code used in the experiments for my thesis:: <br>
**Mitigating Negative Flips in Model Updates by Compatible Learning Representation** <br>

## Requirements

```
pip install -r requirements.txt
```

## Repository Overview

There are two directories:
 * [Experiments_CIFAR10/](./Experiments_CIFAR10/) contains the code for the initial experiment on the simpler CIFAR-10 dataset. Models were trained following the approach from https://github.com/KellerJordan/cifar10-airbench;
 * [Experiments_CIFAR100](./Experiments_CIFAR10/) contains the code for more complex experiments conducted on the CIFAR-100 dataset;


[NFR_losses](.NFR_losses.py) implements various loss functions designed to reduce negative flips.
[compatibility_eval](.compatibility_eval.py) provides tools for evaluating backward compatibility between models.
[features_alignment](.features_alignment.py) contains code to compare image features, class prototypes or adjacency matrices.
[negative_flip](negative_flip.py) computes negative flips during model updates.

