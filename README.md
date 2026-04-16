# AeTHERON
### Autoregressive Topology-aware Heterogeneous Graph Operator Network for Fluid-Structure Interaction

[![arXiv](https://img.shields.io/badge/arXiv-2604.13369-b31b1b.svg)](https://arxiv.org/abs/2604.13369)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

AeTHERON is a heterogeneous graph neural operator for surrogate modeling of body-driven fluid-structure interaction (FSI) governed by the sharp-interface immersed boundary method. The architecture mirrors the IBM discretization through a dual-graph representation coupled via sparse cross-attention, enabling fast prediction of coupled fluid-membrane dynamics. Full details in the [paper](https://arxiv.org/abs/2604.13369).

> **Continuously developing project — results and code will be updated in subsequent versions.**

---

## Installation

```bash
git clone https://github.com/sushrutkr/AeTHERON.git
cd AeTHERON
pip install torch torch-geometric
pip install -r requirements.txt
```

---

## Usage

```bash
# Training
python src/train.py --config input/config.json

# Inference
python src/inference.py --config input/config.json --checkpoint ref_models/
```

---

## Model Architecture

```
Encoder  : fluid (R^4) + membrane (R^10) -> shared latent space (d_h = 32)
           sinusoidal time embedding (d_t = 16)
Processor: L = 10 heterogeneous message-passing layers
           intra-domain GNO + cross-domain sparse attention (d_A = 32)
           time-conditioning via LayerNorm scale-shift
Decoder  : Euler update  x_f(t+tau) = x_f(t) + tau * psi(xi_f)
Params   : ~0.6M
```

---

## Citation

```bibtex
@article{kumar2026aetheron,
  title   = {AeTHERON: Autoregressive Topology-aware Heterogeneous Graph
             Operator Network for Fluid-Structure Interaction},
  author  = {Kumar, Sushrut},
  journal = {arXiv preprint arXiv:2604.13369},
  year    = {2026}
}
```

## Model Visual Overview

![model](Model_overview.png)
