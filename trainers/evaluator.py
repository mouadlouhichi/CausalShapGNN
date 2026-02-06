"""
Evaluator for explanations
"""

import torch
import numpy as np
from typing import Dict, List

from models.tasem import TopologyAwareShapley


class Evaluator:
    """Evaluates model and explanation quality"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.tasem = TopologyAwareShapley(model, device)
    
    def compute_fidelity_plus(self, adj_norm: torch.sparse.Tensor,
                              user_idx: int, item_idx: int,
                              top_k: int = 3) -> float:
        with torch.no_grad():
            user_emb, item_emb, _ = self.model(adj_norm, use_causal_only=True)
        
        self.tasem._compute_population_means(user_emb, item_emb)
        
        original_score = (user_emb[user_idx] * item_emb[item_idx]).sum().item()
        
        shapley = self.tasem.compute_feature_shapley(
            user_idx, item_idx, user_emb[user_idx], item_emb[item_idx]
        )
        
        top_factors = set(np.argsort(shapley)[-top_k:])
        
        all_factors = set(range(self.model.cdm.n_factors))
        remaining_factors = all_factors - top_factors
        
        masked_score = self.tasem._compute_value_function(
            user_idx, item_idx, user_emb[user_idx], item_emb[item_idx],
            remaining_factors
        )
        
        return original_score - masked_score
    
    def compute_fidelity_minus(self, adj_norm: torch.sparse.Tensor,
                               user_idx: int, item_idx: int,
                               top_k: int = 3) -> float:
        with torch.no_grad():
            user_emb, item_emb, _ = self.model(adj_norm, use_causal_only=True)
        
        self.tasem._compute_population_means(user_emb, item_emb)
        
        original_score = (user_emb[user_idx] * item_emb[item_idx]).sum().item()
        
        shapley = self.tasem.compute_feature_shapley(
            user_idx, item_idx, user_emb[user_idx], item_emb[item_idx]
        )
        
        bottom_factors = set(np.argsort(shapley)[:top_k])
        
        all_factors = set(range(self.model.cdm.n_factors))
        remaining_factors = all_factors - bottom_factors
        
        masked_score = self.tasem._compute_value_function(
            user_idx, item_idx, user_emb[user_idx], item_emb[item_idx],
            remaining_factors
        )
        
        return original_score - masked_score
    
    def compute_bias_metrics(self, adj_norm: torch.sparse.Tensor,
                            test_users: List[int],
                            graph_processor,
                            top_k: int = 20) -> Dict:
        self.model.eval()
        
        with torch.no_grad():
            user_emb, item_emb, _ = self.model(adj_norm, use_causal_only=True)
        
        recommendation_counts = np.zeros(graph_processor.n_items)
        
        for user in test_users:
            if user >= user_emb.size(0):
                continue
            
            user_e = user_emb[user]
            scores = torch.matmul(user_e, item_emb.t())
            
            train_items = list(graph_processor.train_user_items[user])
            if train_items:
                scores[train_items] = -float('inf')
            
            _, top_items = torch.topk(scores, min(top_k, scores.size(0)))
            
            for item in top_items.cpu().numpy():
                recommendation_counts[item] += 1
        
        # Compute Gini coefficient
        sorted_counts = np.sort(recommendation_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts) - (n + 1) * np.sum(sorted_counts))
        gini /= (n * np.sum(sorted_counts) + 1e-8)
        
        return {
            'gini': gini,
            'recommendation_counts': recommendation_counts
        }