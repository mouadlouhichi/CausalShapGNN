#!/usr/bin/env python
"""
Full evaluation with all metrics including bias analysis and explanations
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataPreprocessor, BipartiteGraphProcessor, RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed
from utils.metrics import gini_coefficient
from explainers import FeatureShapley, ExplanationReport


def full_evaluation(dataset='movielens-100k', checkpoint='best_model.pt'):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    preprocessor = DataPreprocessor('./data', dataset)
    graph_data = preprocessor.load_data()
    
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users, graph_data.n_items,
        graph_data.train_interactions, device
    )
    
    # Load model
    config = {
        'n_users': graph_data.n_users,
        'n_items': graph_data.n_items,
        'embed_dim': 64,
        'n_factors': 4,
        'n_layers': 3,
        'temperature': 0.2,
        'alpha': 0.1, 'beta': 0.1, 'gamma': 0.1, 'delta': 0.1,
        'reg_weight': 1e-5,
        'training': {'lr': 0.001, 'batch_size': 1024}
    }
    
    model = CausalShapGNN(config, device)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")
    model.eval()
    
    trainer = Trainer(model, graph_processor, config, device)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # 1. Standard Metrics
    print("\n1. RECOMMENDATION METRICS")
    print("-"*40)
    test_metrics = trainer.evaluate(graph_processor.norm_adj, graph_data.test_interactions)
    for k, v in sorted(test_metrics.items()):
        print(f"   {k}: {v:.4f}")
    
    # 2. Popularity Bias Analysis
    print("\n2. POPULARITY BIAS ANALYSIS")
    print("-"*40)
    
    with torch.no_grad():
        user_emb, item_emb, _ = model(graph_processor.norm_adj, use_causal_only=True)
    
    # Get recommendations for all test users
    test_users = list(set(u for u, _ in graph_data.test_interactions))
    recommendation_counts = np.zeros(graph_data.n_items)
    
    for user in test_users:
        if user >= user_emb.size(0):
            continue
        scores = torch.matmul(user_emb[user], item_emb.t())
        train_items = list(graph_processor.train_user_items[user])
        if train_items:
            scores[train_items] = -float('inf')
        _, top_items = torch.topk(scores, 20)
        for item in top_items.cpu().numpy():
            recommendation_counts[item] += 1
    
    gini = gini_coefficient(recommendation_counts)
    print(f"   Gini Coefficient: {gini:.4f} (lower = more fair)")
    
    # Coverage
    coverage = np.sum(recommendation_counts > 0) / graph_data.n_items
    print(f"   Item Coverage@20: {coverage:.4f} ({int(coverage * graph_data.n_items)}/{graph_data.n_items} items)")
    
    # Popularity correlation
    item_pops = np.array([graph_processor.item_popularity.get(i, 0) for i in range(graph_data.n_items)])
    pop_correlation = np.corrcoef(item_pops, recommendation_counts)[0, 1]
    print(f"   Popularity Correlation: {pop_correlation:.4f}")
    
    # 3. Explanation Quality (sample)
    print("\n3. EXPLANATION SAMPLE")
    print("-"*40)
    
    sample_user = test_users[0]
    feature_explainer = FeatureShapley(model, device)
    feature_explainer._compute_population_means(user_emb, item_emb)
    
    # Get top recommendation
    scores = torch.matmul(user_emb[sample_user], item_emb.t())
    train_items = list(graph_processor.train_user_items[sample_user])
    if train_items:
        scores[train_items] = -float('inf')
    top_item = scores.argmax().item()
    
    shapley = feature_explainer.compute(sample_user, top_item, user_emb, item_emb)
    
    print(f"   User {sample_user} → Item {top_item}")
    print(f"   Factor Contributions (Shapley values):")
    factor_names = ['Genre', 'Recency', 'Quality', 'Social'][:len(shapley)]
    for i, (name, val) in enumerate(zip(factor_names, shapley)):
        bar = '█' * int(abs(val) * 50)
        sign = '+' if val >= 0 else '-'
        print(f"     {name}: {sign}{abs(val):.4f} {bar}")
    
    # 4. Model Statistics
    print("\n4. MODEL STATISTICS")
    print("-"*40)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Embedding dim: {config['embed_dim']}")
    print(f"   Causal factors: {config['n_factors']}")
    print(f"   GNN layers: {config['n_layers']}")
    
    # Causal gate values
    gate_values = []
    for layer_gates in model.cdm.causal_gates:
        for g in layer_gates:
            gate_values.append(torch.sigmoid(g).mean().item())
    print(f"   Avg gate activation: {np.mean(gate_values):.4f}")
    
    print("\n" + "="*70)
    
    return test_metrics, gini


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='movielens-100k')
    parser.add_argument('--checkpoint', default='best_model.pt')
    args = parser.parse_args()
    
    full_evaluation(args.dataset, args.checkpoint)