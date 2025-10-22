# GAMMA: Gated Multi-hop Message Passing for Homophily-Agnostic Node Representation in GNNs

Official implementation of **GAMMA** (Gated Multi-hop Message Passing for Homophily-Agnostic Node Representation in GNNs), accepted at **NeurIPS 2025**.

**Authors:** Amir Ghazizadeh, Rickard Ewetz, Hao Zheng

## Overview

GAMMA is a novel Graph Neural Network architecture designed for heterophilic graphs, where connected nodes often belong to different classes. Unlike standard GNNs that assume homophily (similar nodes are connected), GAMMA adaptively leverages multi-hop neighborhood information through a lightweight iterative gating mechanism, achieving state-of-the-art accuracy with up to **20× faster inference** and **12× lower memory** consumption compared to existing heterophilic GNN methods.

### Key Features

- **Adaptive Multi-hop Aggregation:** Dynamic node-specific gating mechanism to identify informative hop distances
- **Efficient Weight Sharing:** Unified transformation with learnable channel-wise scaling reduces memory by up to 12×
- **Practical Efficiency:** Up to 20× faster inference than H2GCN while maintaining competitive accuracy
- **Homophily-Agnostic:** Effective across both homophilic and heterophilic graph structures

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- PyTorch Geometric 2.0.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/GAMMA.git
cd GAMMA

# Install dependencies
pip install -r requirements.txt
```

For PyTorch and PyTorch Geometric installation with CUDA support, please refer to their official documentation:
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Repository Structure

```
GAMMA/
├── GAMMA_layer.py          # Core GAMMA layer implementation
├── gamma.py                # Training script for large-scale datasets
├── gamma_multi_split.py    # Training script for small-scale heterophilic datasets
├── models.py               # Baseline model implementations
├── create_splits.py        # Script to generate fixed data splits
├── Bernpro.py             # BernNet baseline implementation
├── requirements.txt        # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Quick Start

### 1. Generate Data Splits

For small-scale heterophilic datasets (Texas, Wisconsin, Cornell, Chameleon, Squirrel, Actor), first generate the fixed train/validation/test splits:

```bash
python create_splits.py
```

This creates a `splits/` directory containing 10 different 60/20/20 train/val/test splits for each dataset, ensuring reproducibility across experiments.

### 2. Training on Heterophilic Datasets

```bash
# Train on Texas dataset with default settings
python gamma_multi_split.py --dataset Texas

# Train on Chameleon with custom hyperparameters
python gamma_multi_split.py --dataset Chameleon --hidden_channels 64 --num_layers 3 --num_iterations 2

# Available datasets: Texas, Wisconsin, Cornell, Chameleon, Squirrel, Actor
```

**Key Arguments:**
- `--dataset`: Dataset name (required)
- `--hidden_channels`: Hidden dimension size (default: 32)
- `--num_layers`: Number of GAMMA layers (default: 2)
- `--num_iterations`: Routing iterations in GAMMA layer (default: 2)
- `--num_splits`: Number of splits to evaluate (default: 10)


### Heterophilic Benchmarks

To reproduce the results in Table 1 of the paper for small-scale heterophilic datasets:

```bash

python gamma_multi_split.py --dataset Texas --hidden_channels 32 --num_layers 2 --num_iterations 2

python gamma_multi_split.py --dataset Wisconsin --hidden_channels 32 --num_layers 2 --num_iterations 2

python gamma_multi_split.py --dataset Cornell --hidden_channels 32 --num_layers 2 --num_iterations 2

python gamma_multi_split.py --dataset Actor --hidden_channels 32 --num_layers 2 --num_iterations 2

python gamma_multi_split.py --dataset Squirrel --hidden_channels 32 --num_layers 2 --num_iterations 2

python gamma_multi_split.py --dataset Chameleon --hidden_channels 32 --num_layers 2 --num_iterations 2
```

### Homophilic Benchmarks

```bash

python gamma.py --dataset Cora --hidden_channels 32 --num_layers 2

python gamma.py --dataset CiteSeer --hidden_channels 32 --num_layers 2

python gamma.py --dataset PubMed --hidden_channels 32 --num_layers 2
```

## Model Architecture

The GAMMA layer implements the gating mechanism described in Section 4 of the paper. Key components:

1. **Shared Projection:** All hops share a single weight matrix `W` for feature transformation
2. **Hop-Specific Scaling:** Learnable channel-wise scaling factors `γ_p` for each hop
3. **Dynamic Routing:** Iterative gating coefficients `α_i,p` computed via agreement scores

The routing mechanism adaptively weights hop-specific embeddings based on their agreement with the node's evolving representation, enabling each node to focus on the most informative hop distances without requiring separate parameters per hop.

See `GAMMA_layer.py` for the complete implementation with detailed documentation.

## Computational Efficiency

GAMMA achieves significant computational advantages over state-of-the-art heterophilic GNNs:

| Model | Forward+Backward (ms) | Memory (MB) | Speedup vs GAMMA |
|-------|----------------------|-------------|------------------|
| **GAMMA** | **23.17** | **480.60** | **1.0×** |
| GCN | 15.24 | 239.03 | 0.66× (baseline) |
| H2GCN | 467.91 | 1993.90 | 20.2× slower |
| M2MGNN | 172.47 | 5813.80 | 7.4× slower |
| MixHop | 114.75 | 768.49 | 4.95× slower |

*Benchmark on Flickr dataset using NVIDIA RTX A2000 GPU (12GB VRAM)*

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{ghazizadeh2025gamma,
  title={GAMMA: Gated Multi-hop Message Passing for Homophily-Agnostic Node Representation in GNNs},
  author={Ghazizadeh, Amir and Ewetz, Rickard and Zheng, Hao},
  booktitle={Proceedings of the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgments

We thank the authors of the following repositories for their baseline implementations:

- [CoGNN](https://github.com/benfinkelshtein/CoGNN) - Cooperative Graph Neural Networks baseline
- [GloGNN](https://github.com/RecklessRonan/GloGNN) - Global homophily baseline and dataset preprocessing utilities

This work was supported by DARPA under Cooperative Agreement FA8750-23-2-0501, the U.S. Department of Energy under grant DE-SC0024428, and the U.S. National Science Foundation under CAREER Award CCF-2441973.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Amir Ghazizadeh: amir.g@ucf.edu

