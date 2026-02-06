#!/usr/bin/env python
"""
scripts/tune_hyperparams.py - Hyperparameter tuning for CausalShapGNN
"""

import os
import sys
import torch
import itertools
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataPreprocessor, BipartiteGraphProcessor, RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed


def tune():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    preprocessor = DataPreprocessor('./data', 'movielens-100k')
    graph_data = preprocessor.load_data()
    
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users, graph_data.n_items,
        graph_data.train_interactions, device
    )
    
    # Hyperparameter grid
    param_grid = {
        'embed_dim': [64, 128],
        'n_factors': [4, 8],
        'n_layers': [2, 3, 4],
        'lr': [0.001, 0.0005],
        'alpha': [0.05, 0.1, 0.2],
    }
    
    best_recall = 0
    best_params = None
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"Total combinations to try: {len(combinations)}")
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        config = {
            'n_users': graph_data.n_users,
            'n_items': graph_data.n_items,
            'embed_dim': params['embed_dim'],
            'n_factors': params['n_factors'],
            'n_layers': params['n_layers'],
            'temperature': 0.2,
            'alpha': params['alpha'],
            'beta': 0.1,
            'gamma': 0.1,
            'delta': 0.1,
            'reg_weight': 1e-5,
            'training': {'lr': params['lr'], 'batch_size': 1024}
        }
        
        train_dataset = RecommendationDataset(graph_processor, graph_data.train_interactions)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, 
                                  collate_fn=collate_fn, num_workers=0)
        
        model = CausalShapGNN(config, device)
        trainer = Trainer(model, graph_processor, config, device)
        
        # Quick training (30 epochs)
        for epoch in range(30):
            trainer.train_epoch(train_loader, graph_processor.norm_adj)
        
        # Evaluate
        metrics = trainer.evaluate(graph_processor.norm_adj, graph_data.val_interactions)
        recall = metrics['recall@20']
        
        print(f"  Recall@20: {recall:.4f}")
        
        results.append({**params, 'recall@20': recall, 'ndcg@20': metrics['ndcg@20']})
        
        if recall > best_recall:
            best_recall = recall
            best_params = params
            print(f"  *** New best! ***")
    
    print("\n" + "="*60)
    print("TUNING RESULTS")
    print("="*60)
    print(f"Best Recall@20: {best_recall:.4f}")
    print(f"Best params: {best_params}")
    
    # Print top 5
    results.sort(key=lambda x: x['recall@20'], reverse=True)
    print("\nTop 5 configurations:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. R@20={r['recall@20']:.4f} | {r}")
    
    return best_params


if __name__ == "__main__":
    tune()