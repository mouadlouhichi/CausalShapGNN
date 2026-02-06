#!/usr/bin/env python
"""
Quick training script for CausalShapGNN
No config file required - uses command line arguments
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataDownloader, DataPreprocessor, BipartiteGraphProcessor
from data import RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Quick Train CausalShapGNN')
    parser.add_argument('--dataset', type=str, default='movielens-100k',
                       choices=['movielens-100k', 'movielens-1m', 'gowalla', 
                               'yelp2018', 'amazon-book'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_factors', type=int, default=4, help='Number of causal factors')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--download', action='store_true', help='Download dataset if not exists')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check/download data
    dataset_path = os.path.join(args.data_dir, args.dataset)
    train_file = os.path.join(dataset_path, 'train.txt')
    
    if not os.path.exists(train_file):
        if args.download:
            print(f"\nDownloading {args.dataset}...")
            downloader = DataDownloader(args.data_dir)
            downloader.download(args.dataset)
        else:
            print(f"\nDataset not found at {dataset_path}")
            print(f"Run with --download flag or manually download:")
            print(f"  python scripts/download_data.py --dataset {args.dataset}")
            return
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {args.dataset}")
    print(f"{'='*60}")
    
    preprocessor = DataPreprocessor(args.data_dir, args.dataset)
    graph_data = preprocessor.load_data()
    
    # Build config
    config = {
        'n_users': graph_data.n_users,
        'n_items': graph_data.n_items,
        'embed_dim': args.embed_dim,
        'n_factors': args.n_factors,
        'n_layers': args.n_layers,
        'temperature': 0.2,
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.1,
        'reg_weight': 1e-5,
        'training': {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'n_epochs': args.epochs,
        }
    }
    
    # Build graph
    print(f"\n{'='*60}")
    print("Building Graph")
    print(f"{'='*60}")
    
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users,
        graph_data.n_items,
        graph_data.train_interactions,
        device
    )
    
    # Data loader
    train_dataset = RecommendationDataset(graph_processor, graph_data.train_interactions)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Model
    print(f"\n{'='*60}")
    print("Initializing Model")
    print(f"{'='*60}")
    
    model = CausalShapGNN(config, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = Trainer(model, graph_processor, config, device)
    
    # Train
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")
    
    best_recall = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train epoch
        losses = trainer.train_epoch(train_loader, graph_processor.norm_adj)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = trainer.evaluate(graph_processor.norm_adj, graph_data.val_interactions)
            
            print(f"Epoch {epoch+1:3d} | Loss: {losses['total']:.4f} | "
                  f"R@20: {metrics['recall@20']:.4f} | N@20: {metrics['ndcg@20']:.4f}")
            
            if metrics['recall@20'] > best_recall:
                best_recall = metrics['recall@20']
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    
    if os.path.exists('best_model.pt'):
        model.load_state_dict(torch.load('best_model.pt'))
    
    test_metrics = trainer.evaluate(graph_processor.norm_adj, graph_data.test_interactions)
    
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()