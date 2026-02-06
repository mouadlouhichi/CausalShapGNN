#!/usr/bin/env python
"""
Training script for CausalShapGNN
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, get_default_config
from data import DataPreprocessor, BipartiteGraphProcessor, RecommendationDataset, collate_fn
from models import CausalShapGNN
from trainers import Trainer
from utils import set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train CausalShapGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    config.update(load_config(args.config))
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    preprocessor = DataPreprocessor(
        config['dataset']['path'],
        config['dataset']['name']
    )
    graph_data = preprocessor.load_data()
    
    # Update config with data sizes
    config['n_users'] = graph_data.n_users
    config['n_items'] = graph_data.n_items
    
    # Process graph
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users,
        graph_data.n_items,
        graph_data.train_interactions,
        device
    )
    
    # Create data loaders
    train_dataset = RecommendationDataset(graph_processor, graph_data.train_interactions)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    model = CausalShapGNN(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, graph_processor, config, device)
    
    # Training loop
    best_recall = 0
    patience_counter = 0
    
    for epoch in range(config['training']['n_epochs']):
        # Train
        train_losses = trainer.train_epoch(train_loader, graph_processor.norm_adj)
        
        # Evaluate
        if (epoch + 1) % config['training']['eval_interval'] == 0:
            val_metrics = trainer.evaluate(
                graph_processor.norm_adj,
                graph_data.val_interactions
            )
            
            print(f"Epoch {epoch+1}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Recall@20: {val_metrics['recall@20']:.4f}")
            print(f"  Val NDCG@20: {val_metrics['ndcg@20']:.4f}")
            
            # Early stopping
            if val_metrics['recall@20'] > best_recall:
                best_recall = val_metrics['recall@20']
                patience_counter = 0
                
                # Save best model
                os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt')
                )
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stop_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt'))
    )
    
    test_metrics = trainer.evaluate(
        graph_processor.norm_adj,
        graph_data.test_interactions
    )
    
    print("\nFinal Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()