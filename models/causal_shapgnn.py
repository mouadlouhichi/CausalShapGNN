"""
CausalShapGNN: Main Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .cdm import CausalDisentanglementModule
from .cc_ssl import ContrastiveCausalSSL
from .tasem import TopologyAwareShapley


class CausalShapGNN(nn.Module):
    """
    CausalShapGNN: Causal Disentangled GNN with Topology-Aware Shapley Explanations.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Core architecture
        self.cdm = CausalDisentanglementModule(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embed_dim=config.get('embed_dim', 64),
            n_factors=config.get('n_factors', 8),
            n_layers=config.get('n_layers', 3),
            device=device
        ).to(device)
        
        # Contrastive learning module
        self.cc_ssl = ContrastiveCausalSSL(
            embed_dim=config.get('embed_dim', 64),
            temperature=config.get('temperature', 0.2)
        ).to(device)
        
        # Loss weights
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.1)
        self.gamma = config.get('gamma', 0.1)
        self.delta = config.get('delta', 0.1)
        self.reg_weight = config.get('reg_weight', 1e-5)
    
    def forward(self, adj_norm: torch.sparse.Tensor,
                use_causal_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        return self.cdm(adj_norm, use_causal_only)
    
    def compute_bpr_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                         users: torch.Tensor, pos_items: torch.Tensor,
                         neg_items: torch.Tensor) -> torch.Tensor:
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items.squeeze()]
        
        pos_scores = (user_e * pos_e).sum(dim=-1)
        neg_scores = (user_e * neg_e).sum(dim=-1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        return loss
    
    def compute_total_loss(self, adj_norm: torch.sparse.Tensor,
                          users: torch.Tensor, pos_items: torch.Tensor,
                          neg_items: torch.Tensor,
                          adj_norm_causal: Optional[torch.sparse.Tensor] = None,
                          adj_norm_confound: Optional[torch.sparse.Tensor] = None
                          ) -> Tuple[torch.Tensor, Dict]:
        # Forward pass
        user_emb, item_emb, aux_info = self.forward(adj_norm)
        
        # BPR loss
        loss_bpr = self.compute_bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
        
        # CDM loss
        loss_cdm = self.cdm.compute_cdm_loss(aux_info)
        
        # Initialize contrastive losses
        loss_inv = torch.tensor(0.0, device=self.device)
        loss_disent = torch.tensor(0.0, device=self.device)
        loss_cf = torch.tensor(0.0, device=self.device)
        
        # Compute contrastive losses if augmented views provided
        if adj_norm_causal is not None and adj_norm_confound is not None:
            user_emb_c, item_emb_c, aux_c = self.forward(adj_norm_causal)
            user_emb_s, item_emb_s, aux_s = self.forward(adj_norm_confound)
            
            all_emb = torch.cat([user_emb, item_emb], dim=0)
            all_emb_c = torch.cat([user_emb_c, item_emb_c], dim=0)
            
            batch_indices = torch.cat([users, self.cdm.n_users + pos_items])
            loss_inv = self.cc_ssl.compute_invariance_loss(all_emb, all_emb_c, batch_indices)
            
            causal_repr = aux_info['causal_repr'][-1]
            confound_repr = aux_info['confounding_repr'][-1]
            loss_disent = self.cc_ssl.compute_disentanglement_loss(
                causal_repr[batch_indices], confound_repr[batch_indices]
            )
            
            causal_repr_s = aux_s['causal_repr'][-1]
            confound_repr_s = aux_s['confounding_repr'][-1]
            loss_cf = self.cc_ssl.compute_counterfactual_loss(
                causal_repr, causal_repr_s,
                confound_repr, confound_repr_s,
                batch_indices
            )
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2)
        )
        
        # Total loss
        total_loss = (loss_bpr +
                     self.alpha * loss_cdm +
                     self.beta * loss_inv +
                     self.gamma * loss_disent +
                     self.delta * loss_cf +
                     reg_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'bpr': loss_bpr.item(),
            'cdm': loss_cdm.item(),
            'invariance': loss_inv.item(),
            'disentanglement': loss_disent.item(),
            'counterfactual': loss_cf.item(),
            'reg': reg_loss.item()
        }
        
        return total_loss, loss_dict
    
    def predict(self, adj_norm: torch.sparse.Tensor,
                users: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            user_emb, item_emb, _ = self.forward(adj_norm, use_causal_only=True)
            user_e = user_emb[users]
            scores = torch.matmul(user_e, item_emb.t())
            return scores
    
    def get_explanations(self, adj_norm: torch.sparse.Tensor,
                        user_idx: int, item_idx: int) -> Dict:
        tasem = TopologyAwareShapley(self, self.device)
        
        with torch.no_grad():
            user_emb, item_emb, _ = self.forward(adj_norm, use_causal_only=True)
        
        tasem._compute_population_means(user_emb, item_emb)
        
        feature_shapley = tasem.compute_feature_shapley(
            user_idx, item_idx, user_emb[user_idx], item_emb[item_idx]
        )
        
        path_shapley = tasem.compute_path_shapley(user_idx, item_idx, adj_norm)
        
        return {
            'feature_shapley': feature_shapley,
            'path_shapley': path_shapley,
            'user_embedding': user_emb[user_idx].cpu().numpy(),
            'item_embedding': item_emb[item_idx].cpu().numpy()
        }