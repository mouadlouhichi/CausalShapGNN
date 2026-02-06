"""
Trainer for CausalShapGNN
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm


class Trainer:
    """Training pipeline for CausalShapGNN"""
    
    def __init__(self, model, graph_processor, config: Dict, device: torch.device):
        self.model = model
        self.graph_processor = graph_processor
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
    
    def train_epoch(self, train_loader: DataLoader, 
                    adj_norm: torch.sparse.Tensor) -> Dict:
        self.model.train()
        total_losses = defaultdict(float)
        n_batches = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            users, pos_items, neg_items = batch
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.model.compute_total_loss(
                adj_norm, users, pos_items, neg_items
            )
            
            loss.backward()
            self.optimizer.step()
            
            for k, v in loss_dict.items():
                total_losses[k] += v
            n_batches += 1
        
        return {k: v / n_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def evaluate(self, adj_norm: torch.sparse.Tensor,
                 test_interactions: List[Tuple[int, int]],
                 k_list: List[int] = [10, 20, 50]) -> Dict:
        self.model.eval()
        
        # Build test user-item dict
        test_user_items = defaultdict(set)
        for u, i in test_interactions:
            test_user_items[u].add(i)
        
        test_users = list(test_user_items.keys())
        
        # Get embeddings
        user_emb, item_emb, _ = self.model(adj_norm, use_causal_only=True)
        
        metrics = {f'recall@{k}': 0.0 for k in k_list}
        metrics.update({f'ndcg@{k}': 0.0 for k in k_list})
        
        n_users = 0
        
        for user in tqdm(test_users, desc='Evaluating'):
            if user >= user_emb.size(0):
                continue
            
            user_e = user_emb[user]
            scores = torch.matmul(user_e, item_emb.t())
            
            # Mask training items
            train_items = list(self.graph_processor.train_user_items[user])
            if train_items:
                scores[train_items] = -float('inf')
            
            max_k = max(k_list)
            _, top_items = torch.topk(scores, min(max_k, scores.size(0)))
            top_items = top_items.cpu().numpy()
            
            test_items = test_user_items[user]
            
            for k in k_list:
                top_k = set(top_items[:k])
                
                hits = len(top_k & test_items)
                metrics[f'recall@{k}'] += hits / min(k, len(test_items))
                
                dcg = 0.0
                for idx, item in enumerate(top_items[:k]):
                    if item in test_items:
                        dcg += 1.0 / np.log2(idx + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
                metrics[f'ndcg@{k}'] += dcg / idcg if idcg > 0 else 0
            
            n_users += 1
        
        for k in metrics:
            metrics[k] /= max(n_users, 1)
        
        return metrics