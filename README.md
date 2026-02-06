# CausalShapGNN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of  
**CausalShapGNN: Causal Disentangled Graph Neural Networks with Topology-Aware Shapley Explanations for Recommender Systems**

---

## 🚀 Key Features

- **Causal Disentanglement Module (CDM)**  
  Separates causal and confounding signals within GNN message passing

- **Contrastive Causal SSL (CC-SSL)**  
  Self-supervised learning using counterfactual graph augmentations

- **Topology-Aware Shapley (TASEM)**  
  Efficient Shapley value estimation via d-separation

- **Multi-granularity Explanations**  
  Feature-, path-, and user-level explanations

---

## 📊 Supported Datasets

| Dataset              | Users   | Items   | Interactions | Available |
|----------------------|---------|---------|--------------|-----------|
| Gowalla              | 29,858  | 40,981  | 1,027,370    | ✅ |
| Yelp2018             | 31,668  | 38,048  | 1,561,406    | ✅ |
| Amazon-Book          | 52,643  | 91,599  | 2,984,108    | ✅ |
| Alibaba-iFashion     | 300,000 | 81,614  | 1,607,813    | ✅ |
| MovieLens-10M        | 69,878  | 10,677  | 10,000,054   | ✅ |

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/CausalShapGNN.git
cd CausalShapGNN
pip install -r requirements.txt
pip install -e .
```

---

## 📥 Download Data

```bash
python scripts/download_data.py --dataset all
```

---

## 🏃 Quick Start

```bash
python scripts/train.py --config config/gowalla.yaml
python scripts/evaluate.py --config config/gowalla.yaml --checkpoint checkpoints/best_model.pt
python scripts/explain.py --config config/gowalla.yaml --user_id 42 --top_k 10
```

---

## 📈 Results

| Model              | Gowalla R@20 | Yelp2018 R@20 | Amazon-Book R@20 | Gini ↓ |
|--------------------|--------------|---------------|------------------|--------|
| LightGCN           | 0.1830       | 0.0649        | 0.0411           | 0.78   |
| SGL                | 0.1920       | 0.0675        | 0.0478           | 0.76   |
| SimGCL             | 0.1960       | 0.0694        | 0.0491           | 0.75   |
| **CausalShapGNN**  | **0.2118**   | **0.0756**    | **0.0563**       | **0.53** |

---

## 📖 Citation

```bibtex
@article{causalshapgnn2026,
  title   = {CausalShapGNN: Causal Disentangled Graph Neural Networks with Topology-Aware Shapley Explanations for Recommender Systems},
  author  = {Mouad Louhichi},
  journal = {IEEE},
  year    = {2026}
}
```