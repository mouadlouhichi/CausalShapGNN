#!/usr/bin/env python
"""
Ablation study for CausalShapGNN
"""

import argparse
import os
import sys
import torch
import json
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, get_default_config
from data import DataPreprocessor, BipartiteGraphProcessor, RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed
from utils.logger import Logger


ABLATION_VARIANTS = {
    'full': {
        'description': 'Full CausalShapGNN model',
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.1
    },
    'no_cdm': {
        'description': 'Without Causal Disentanglement Module',
        'alpha': 0.0,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.1
    },
    'no_ccssl': {
        'description': 'Without Contrastive Causal SSL',
        'alpha': 0.1,
        'beta': 0.0,
        'gamma': 0.0,
        'delta': 0.0
    },
    'no_intervention': {
        'description': 'Without interventional regularization',
        'alpha': 0.0,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.1
    },
    'no_disentangle': {
        'description': 'Without disentanglement loss',
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.0,
        'delta': 0.1
    },
    'no_counterfactual': {
        'description': 'Without counterfactual loss',
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.0
    }
}


def run_ablation(variant_name: str, variant_config: dict,
                 base_config: dict, graph_data, device: torch.device,
                 n_epochs: int = 50) -> dict:
    """Run a single ablation variant"""
    
    print(f"\n{'='*60}")
    print(f"Running: {variant_name}")
    print(f"Description: {variant_config['description']}")
    print(f"{'='*60}")
    
    # Update config
    config = base_config.copy()
    config['loss'] = config.get('loss', {})
    config['loss']['alpha'] = variant_config['alpha']
    config['loss']['beta'] = variant_config['beta']
    config['loss']['gamma'] = variant_config['gamma']
    config['loss']['delta'] = variant_config['delta']
    
    # Also update model-level attributes
    config['alpha'] = variant_config['alpha']
    config['beta'] = variant_config['beta']
    config['gamma'] = variant_config['gamma']
    config['delta'] = variant_config['delta']
    
    # Create graph processor
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users, graph_data.n_items,
        graph_data.train_interactions, device
    )
    
    # Create data loader
    train_dataset = RecommendationDataset(graph_processor, graph_data.train_interactions)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 2048),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model = CausalShapGNN(config, device)
    
    # Initialize trainer
    trainer = Trainer(model, graph_processor, config, device)
    
    # Train
    best_recall = 0
    best_metrics = {}
    
    for epoch in range(n_epochs):
        train_losses = trainer.train_epoch(train_loader, graph_processor.norm_adj)
        
        if (epoch + 1) % 10 == 0:
            val_metrics = trainer.evaluate(
                graph_processor.norm_adj,
                graph_data.val_interactions
            )
            
            print(f"Epoch {epoch+1}: Loss={train_losses['total']:.4f}, "
                  f"Val R@20={val_metrics['recall@20']:.4f}")
            
            if val_metrics['recall@20'] > best_recall:
                best_recall = val_metrics['recall@20']
                best_metrics = val_metrics.copy()
    
    # Final test evaluation
    test_metrics = trainer.evaluate(
        graph_processor.norm_adj,
        graph_data.test_interactions
    )
    
    return {
        'variant': variant_name,
        'description': variant_config['description'],
        'config': variant_config,
        'val_metrics': best_metrics,
        'test_metrics': test_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='CausalShapGNN Ablation Study')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--variants', type=str, nargs='+', default=['all'],
                       help='Variants to run (or "all")')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per variant')
    parser.add_argument('--output', type=str, default='./ablation_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    # Load config and data
    config = get_default_config()
    config.update(load_config(args.config))
    
    preprocessor = DataPreprocessor(config['dataset']['path'], config['dataset']['name'])
    graph_data = preprocessor.load_data()
    
    config['n_users'] = graph_data.n_users
    config['n_items'] = graph_data.n_items
    
    # Determine variants to run
    if 'all' in args.variants:
        variants_to_run = list(ABLATION_VARIANTS.keys())
    else:
        variants_to_run = args.variants
    
    # Run ablations
    results = []
    
    for variant_name in variants_to_run:
        if variant_name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {variant_name}")
            continue
        
        result = run_ablation(
            variant_name,
            ABLATION_VARIANTS[variant_name],
            config,
            graph_data,
            device,
            args.epochs
        )
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"\n{'Variant':<20} {'R@10':<10} {'R@20':<10} {'N@10':<10} {'N@20':<10}")
    print("-" * 60)
    
    for r in results:
        m = r['test_metrics']
        print(f"{r['variant']:<20} {m.get('recall@10', 0):<10.4f} "
              f"{m.get('recall@20', 0):<10.4f} {m.get('ndcg@10', 0):<10.4f} "
              f"{m.get('ndcg@20', 0):<10.4f}")
    
    print("=" * 80)
    
    # Save results
    results_path = os.path.join(args.output, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()