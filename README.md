<img width="600" alt="bl_plot" src="docs/assets/abstract.png">

# Behavior Learning (BL)

This is the github repo for the paper ["Behavior Learning (BL): Discovering Hierarchical Optimization Structures from Data"](https://arxiv.org/abs[需填入arxiv网址]). You may want to quickstart with [hellobl](https://github.com/YueLiang-hye/pybl/blob/master/hellobl.ipynb) or try more examples in [examples](https://github.com/YueLiang-hye/pybl/examples).

Behavior Learning (BL) is a general-purpose machine learning framework grounded in behavioral science. It unifies predictive performance and intrinsic interpretability within a single modeling paradigm. BL learns explicit optimization structures from data by parameterizing a compositional utility function built from interpretable modular blocks. Each block represents a Utility Maximization Problem (UMP), a foundational framework of decision-making and optimization. BL supports architectures ranging from a single UMP to hierarchical compositions, enabling expressive yet structurally transparent models. Unlike post-hoc explanation methods, BL provides interpretability by design while maintaining strong empirical performance on high-dimensional tasks.

## Installation
Pybl can be installed via PyPI or directly from GitHub. 

**Pre-requisites:**

```
Python 3.10.9 or higher
pip
```

**For developers**

```
git clone https://github.com/YueLiang-hye/pybl.git
cd pybl
pip install -e .
```

**Installation via github**

```
pip install git+https://github.com/YueLiang-hye/pybl.git
```

**Installation via PyPI:**
```
pip install pybl
```

Requirements

```python
# python==3.10.9
torch==2.9.1
numpy==1.26.4
pandas==2.3.0
scikit-learn==1.6.1
matplotlib==3.8.4
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

**Optional: Conda Environment Setup**
For those who prefer using Conda:
```
conda create --name pybl-env python=3.10.9
conda activate pybl-env
pip install git+https://github.com/YueLiang-hye/pybl.git  # For GitHub installation
# or
pip install pybl  # For PyPI installation
```

## Computation Requirements

BL is implemented in PyTorch and supports both CPU and GPU training.

- Small-scale tabular examples run on a single CPU within a few minutes.
- High-dimensional settings may benefit from GPU acceleration (e.g., NVIDIA L40).

For most tabular tasks, CPU training is sufficient.

## Examples

Start with the notebooks in [`examples/`](./examples/):

- [Example 1: Boston Housing (continuous)](./examples/Example_1_boston_housing.ipynb)
- [Example 2: German Credit (classification)](./examples/Example_2_german_credit.ipynb)

These notebooks also demonstrate exporting the learned BL structure to readable text files.

## Advice on hyperparameter tuning
In many cases, BL can achieve comparable (or slightly better) performance than an MLP baseline using roughly one third of the hidden width.

Other hyperparameters can be initialized based on standard MLP tuning, and then refined for the specific task.

## Contact
If you have any questions, please contact yue.liang@student.uni-tuebingen.de
