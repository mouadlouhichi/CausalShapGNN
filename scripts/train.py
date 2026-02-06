#!/usr/bin/env python
"""
Training script for CausalShapGNN
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, get_default_config
from data import DataPreprocessor, BipartiteGraphProcessor, RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train CausalShapGNN')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--dataset', type=str, default='movielens-100k', 
                       help='Dataset name (if not using config)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_factors', type=int, default=4, help='Number of factors')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config if provided, otherwise use defaults
    if args.config and os.path.exists(args.config):
        config = get_default_config()
        config.update(load_config(args.config))
        dataset_name = config.get('dataset', {}).get('name', args.dataset)
        data_dir = config.get('dataset', {}).get('path', args.data_dir)
        
        # Fix: if path already contains dataset name, use parent
        if data_dir.endswith(dataset_name):
            data_dir = os.path.dirname(data_dir)
    else:
        config = get_default_config()
        dataset_name = args.dataset
        data_dir = args.data_dir
    
    print(f"\nDataset: {dataset_name}")
    print(f"Data directory: {data_dir}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    preprocessor = DataPreprocessor(data_dir, dataset_name)
    graph_data = preprocessor.load_data()
    
    # Build config
    config['n_users'] = graph_data.n_users
    config['n_items'] = graph_data.n_items
    config['embed_dim'] = args.embed_dim
    config['n_factors'] = args.n_factors
    config['n_layers'] = args.n_layers
    config['temperature'] = config.get('temperature', 0.2)
    config['alpha'] = config.get('alpha', 0.1)
    config['beta'] = config.get('beta', 0.1)
    config['gamma'] = config.get('gamma', 0.1)
    config['delta'] = config.get('delta', 0.1)
    config['reg_weight'] = config.get('reg_weight', 1e-5)
    
    # Training config
    training_config = config.get('training', {})
    batch_size = args.batch_size or training_config.get('batch_size', 2048)
    n_epochs = args.epochs or training_config.get('n_epochs', 100)
    lr = args.lr or training_config.get('lr', 0.001)
    early_stop_patience = training_config.get('early_stop_patience', 20)
    eval_interval = training_config.get('eval_interval', 5)
    
    config['training'] = {
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lr': lr,
        'early_stop_patience': early_stop_patience,
        'eval_interval': eval_interval
    }
    
    # Process graph
    print("\n" + "="*60)
    print("Building Graph")
    print("="*60)
    
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users,
        graph_data.n_items,
        graph_data.train_interactions,
        device
    )
    
    print(f"Graph nodes: {graph_processor.n_nodes}")
    print(f"Adjacency matrix shape: {graph_processor.norm_adj.shape}")
    
    # Create data loaders
    train_dataset = RecommendationDataset(graph_processor, graph_data.train_interactions)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Kaggle compatibility
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)
    
    model = CausalShapGNN(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Initialize trainer
    trainer = Trainer(model, graph_processor, config, device)
    
    # Create checkpoint directory
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    best_recall = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader, graph_processor.norm_adj)
        
        # Evaluate
        if (epoch + 1) % eval_interval == 0:
            val_metrics = trainer.evaluate(
                graph_processor.norm_adj,
                graph_data.val_interactions
            )
            
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(BPR: {train_losses['bpr']:.4f}, CDM: {train_losses['cdm']:.4f})")
            print(f"  Val Recall@20: {val_metrics['recall@20']:.4f}, "
                  f"NDCG@20: {val_metrics['ndcg@20']:.4f}")
            
            # Early stopping
            if val_metrics['recall@20'] > best_recall:
                best_recall = val_metrics['recall@20']
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model (Recall@20: {best_recall:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
    
    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best model")
    
    test_metrics = trainer.evaluate(
        graph_processor.norm_adj,
        graph_data.test_interactions
    )
    
    print("\nTest Set Results:")
    print("-" * 40)
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("-" * 40)
    
    # Compute bias metrics
    test_users = list(set(u for u, _ in graph_data.test_interactions))
    
    print("\nTraining Complete!")
    
    return model, test_metrics


if __name__ == "__main__":
    main()