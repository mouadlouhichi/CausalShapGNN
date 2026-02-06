"""
PyTorch Dataset classes for CausalShapGNN
"""

import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set
import random


class BipartiteGraphProcessor:
    """
    Processes bipartite user-item interaction graph.
    """
    
    def __init__(self, n_users: int, n_items: int, 
                 train_interactions: List[Tuple[int, int]],
                 device: torch.device):
        """
        Initialize graph processor.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            train_interactions: List of (user, item) tuples
            device: Torch device
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.device = device
        
        # Build adjacency structures
        self.train_user_items: Dict[int, Set[int]] = defaultdict(set)
        self.train_item_users: Dict[int, Set[int]] = defaultdict(set)
        self.item_popularity: Dict[int, int] = defaultdict(int)
        
        for u, i in train_interactions:
            self.train_user_items[u].add(i)
            self.train_item_users[i].add(u)
            self.item_popularity[i] += 1
        
        # Compute popularity quintiles
        self._compute_popularity_quintiles()
        
        # Build normalized adjacency matrix
        self.norm_adj = self._build_normalized_adj(train_interactions)
        
        # Compute propensity scores
        self.propensity_scores = self._compute_propensity_scores()
    
    def _compute_popularity_quintiles(self):
        """Stratify items into popularity quintiles"""
        popularities = [(i, self.item_popularity.get(i, 0)) for i in range(self.n_items)]
        popularities.sort(key=lambda x: x[1])
        
        quintile_size = max(1, len(popularities) // 5)
        self.item_quintile: Dict[int, int] = {}
        self.quintile_items: Dict[int, List[int]] = defaultdict(list)
        
        for idx, (item, _) in enumerate(popularities):
            quintile = min(idx // quintile_size, 4)
            self.item_quintile[item] = quintile
            self.quintile_items[quintile].append(item)
    
    def _build_normalized_adj(self, interactions: List[Tuple[int, int]]) -> torch.sparse.Tensor:
        """
        Build symmetric normalized adjacency matrix: D^{-1/2} A D^{-1/2}
        Following LightGCN convention.
        """
        rows, cols = [], []
        
        for u, i in interactions:
            # User -> Item edge
            rows.append(u)
            cols.append(self.n_users + i)
            # Item -> User edge
            rows.append(self.n_users + i)
            cols.append(u)
        
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.ones(len(rows))
        
        adj = sp.coo_matrix((data, (rows, cols)), 
                           shape=(self.n_nodes, self.n_nodes))
        
        # Symmetric normalization
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        
        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)
    
    def _compute_propensity_scores(self) -> Dict[int, float]:
        """Compute propensity scores using item frequency"""
        total = sum(self.item_popularity.values())
        if total == 0:
            return {i: 1.0 / self.n_items for i in range(self.n_items)}
        return {
            i: (self.item_popularity.get(i, 0) + 1) / (total + self.n_items)
            for i in range(self.n_items)
        }
    
    def sample_negative_stratified(self, user: int, n_neg: int = 1) -> List[int]:
        """
        Popularity-stratified negative sampling.
        Sample uniformly across quintiles to avoid popularity bias.
        """
        positive_items = self.train_user_items.get(user, set())
        negatives = []
        
        for _ in range(n_neg):
            # Uniformly sample a quintile
            quintile = random.randint(0, 4)
            candidates = [i for i in self.quintile_items[quintile] 
                         if i not in positive_items]
            
            if candidates:
                negatives.append(random.choice(candidates))
            else:
                # Fallback to any negative
                neg = random.randint(0, self.n_items - 1)
                attempts = 0
                while neg in positive_items and attempts < 100:
                    neg = random.randint(0, self.n_items - 1)
                    attempts += 1
                negatives.append(neg)
        
        return negatives


class RecommendationDataset(Dataset):
    """PyTorch Dataset for training with stratified negative sampling"""
    
    def __init__(self, graph_processor: BipartiteGraphProcessor,
                 interactions: List[Tuple[int, int]], n_neg: int = 1):
        """
        Initialize dataset.
        
        Args:
            graph_processor: BipartiteGraphProcessor instance
            interactions: List of (user, item) tuples
            n_neg: Number of negative samples per positive
        """
        self.graph_processor = graph_processor
        self.interactions = interactions
        self.n_neg = n_neg
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict:
        user, pos_item = self.interactions[idx]
        neg_items = self.graph_processor.sample_negative_stratified(user, self.n_neg)
        
        return {
            'user': user,
            'pos_item': pos_item,
            'neg_items': neg_items
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Tuple of (users, pos_items, neg_items) tensors
    """
    users = torch.LongTensor([b['user'] for b in batch])
    pos_items = torch.LongTensor([b['pos_item'] for b in batch])
    neg_items = torch.LongTensor([b['neg_items'] for b in batch])
    
    return users, pos_items, neg_items