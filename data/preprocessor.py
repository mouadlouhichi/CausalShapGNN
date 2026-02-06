"""
Data Preprocessor for CausalShapGNN
"""

import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class GraphData:
    """Container for graph data structures"""
    n_users: int
    n_items: int
    train_interactions: List[Tuple[int, int]]
    val_interactions: List[Tuple[int, int]]
    test_interactions: List[Tuple[int, int]]
    user_features: Optional[torch.Tensor] = None
    item_features: Optional[torch.Tensor] = None


class DataPreprocessor:
    """
    Preprocessor for recommendation datasets.
    """
    
    def __init__(self, data_dir: str, dataset_name: str):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Root data directory
            dataset_name: Name of dataset
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_path = f"{data_dir}/{dataset_name}"
    
    def load_data(self, val_ratio: float = 0.1) -> GraphData:
        """
        Load and preprocess dataset.
        
        Args:
            val_ratio: Ratio of training data to use for validation
            
        Returns:
            GraphData object containing all data
        """
        import os
        
        train_file = os.path.join(self.dataset_path, 'train.txt')
        test_file = os.path.join(self.dataset_path, 'test.txt')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                f"Please download using: python scripts/download_data.py --dataset {self.dataset_name}"
            )
        
        # Load training data
        train_interactions = []
        n_users = 0
        n_items = 0
        
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    
                    for item in items:
                        train_interactions.append((user, item))
                    
                    n_users = max(n_users, user + 1)
                    n_items = max(n_items, max(items) + 1)
        
        # Load test data
        test_interactions = []
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        user = int(parts[0])
                        items = [int(i) for i in parts[1:]]
                        
                        for item in items:
                            test_interactions.append((user, item))
                        
                        n_users = max(n_users, user + 1)
                        n_items = max(n_items, max(items) + 1)
        
        # Split training into train/val
        random.seed(42)
        random.shuffle(train_interactions)
        
        n_val = int(len(train_interactions) * val_ratio)
        val_interactions = train_interactions[:n_val]
        train_interactions = train_interactions[n_val:]
        
        print(f"Loaded {self.dataset_name}:")
        print(f"  Users: {n_users}")
        print(f"  Items: {n_items}")
        print(f"  Train interactions: {len(train_interactions)}")
        print(f"  Val interactions: {len(val_interactions)}")
        print(f"  Test interactions: {len(test_interactions)}")
        
        return GraphData(
            n_users=n_users,
            n_items=n_items,
            train_interactions=train_interactions,
            val_interactions=val_interactions,
            test_interactions=test_interactions
        )
    
    def compute_statistics(self, graph_data: GraphData) -> Dict:
        """Compute dataset statistics"""
        
        # User degree distribution
        user_degrees = defaultdict(int)
        item_degrees = defaultdict(int)
        
        for u, i in graph_data.train_interactions:
            user_degrees[u] += 1
            item_degrees[i] += 1
        
        user_deg_vals = list(user_degrees.values())
        item_deg_vals = list(item_degrees.values())
        
        stats = {
            'n_users': graph_data.n_users,
            'n_items': graph_data.n_items,
            'n_train': len(graph_data.train_interactions),
            'n_val': len(graph_data.val_interactions),
            'n_test': len(graph_data.test_interactions),
            'density': len(graph_data.train_interactions) / (graph_data.n_users * graph_data.n_items),
            'avg_user_degree': np.mean(user_deg_vals),
            'avg_item_degree': np.mean(item_deg_vals),
            'user_degree_std': np.std(user_deg_vals),
            'item_degree_std': np.std(item_deg_vals),
            'max_user_degree': max(user_deg_vals),
            'max_item_degree': max(item_deg_vals),
        }
        
        # Compute Gini coefficient for popularity bias
        sorted_items = sorted(item_deg_vals)
        n = len(sorted_items)
        index = np.arange(1, n + 1)
        stats['item_gini'] = (2 * np.sum(index * sorted_items) - (n + 1) * np.sum(sorted_items)) / (n * np.sum(sorted_items))
        
        return stats