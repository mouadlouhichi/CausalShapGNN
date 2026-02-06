#!/usr/bin/env python
"""
Evaluation script for CausalShapGNN
"""

import argparse
import os
import sys
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, get_default_config
from data import DataPreprocessor, BipartiteGraphProcessor
from models import CausalShapGNN
from trainers import Evaluator
from utils.metrics import compute_all_metrics, gini_coefficient
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Evaluate CausalShapGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    # Load config
    config = get_default_config()
    config.update(load_config(args.config))
    
    # Load data
    preprocessor = DataPreprocessor(config['dataset']['path'], config['dataset']['name'])
    graph_data = preprocessor.load_data()
    
    config['n_users'] = graph_data.n_users
    config['n_items'] = graph_data.n_items
    
    # Process graph
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users, graph_data.n_items,
        graph_data.train_interactions, device
    )
    
    # Load model
    model = CausalShapGNN(config, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print("Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating on test set...")
    
    # Build ground truth
    test_user_items = defaultdict(set)
    for u, i in graph_data.test_interactions:
        test_user_items[u].add(i)
    
    # Generate recommendations
    with torch.no_grad():
        user_emb, item_emb, _ = model(graph_processor.norm_adj, use_causal_only=True)
    
    recommendations = {}
    
    for user in test_user_items.keys():
        if user >= user_emb.size(0):
            continue
        
        scores = torch.matmul(user_emb[user], item_emb.t())
        
        # Mask training items
        train_items = list(graph_processor.train_user_items[user])
        if train_items:
            scores[train_items] = -float('inf')
        
        _, top_items = torch.topk(scores, 50)
        recommendations[user] = top_items.cpu().numpy().tolist()
    
    # Compute metrics
    metrics = compute_all_metrics(
        recommendations, test_user_items,
        k_list=[10, 20, 50],
        n_items=graph_data.n_items
    )
    
    # Compute bias metrics
    all_recs = []
    for recs in recommendations.values():
        all_recs.extend(recs[:20])
    
    rec_counts = defaultdict(int)
    for item in all_recs:
        rec_counts[item] += 1
    
    counts_array = np.array([rec_counts.get(i, 0) for i in range(graph_data.n_items)])
    metrics['gini'] = gini_coefficient(counts_array)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    # Save results
    import json
    results_path = os.path.join(args.output, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import numpy as np
    main()