"""
Contrastive Causal Self-Supervised Learning (CC-SSL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict


class CounterfactualAugmenter:
    """Generates counterfactual graph augmentations"""
    
    def __init__(self, causal_threshold: float = 0.5,
                 confounding_threshold: float = 0.5):
        self.causal_threshold = causal_threshold
        self.confounding_threshold = confounding_threshold
    
    def compute_edge_scores(self, causal_repr: List[torch.Tensor],
                           confounding_repr: List[torch.Tensor],
                           interactions: List[Tuple[int, int]],
                           n_users: int) -> Tuple[Dict, Dict]:
        
        causal_stack = torch.stack(causal_repr, dim=0).mean(dim=0)
        confounding_stack = torch.stack(confounding_repr, dim=0).mean(dim=0)
        
        causal_scores = {}
        confounding_scores = {}
        
        for u, i in interactions:
            user_causal = causal_stack[u]
            user_confound = confounding_stack[u]
            item_causal = causal_stack[n_users + i]
            item_confound = confounding_stack[n_users + i]
            
            causal_norm = (user_causal.norm() + item_causal.norm()) / 2
            confound_norm = (user_confound.norm() + item_confound.norm()) / 2
            total_norm = causal_norm + confound_norm + 1e-8
            
            causal_scores[(u, i)] = (causal_norm / total_norm).item()
            confounding_scores[(u, i)] = (confound_norm / total_norm).item()
        
        return causal_scores, confounding_scores
    
    def create_causal_preserving_view(self, interactions: List[Tuple[int, int]],
                                      confounding_scores: Dict) -> List[Tuple[int, int]]:
        causal_edges = [e for e in interactions 
                       if confounding_scores.get(e, 0) < self.confounding_threshold]
        
        if len(causal_edges) < len(interactions) * 0.5:
            sorted_edges = sorted(interactions, 
                                 key=lambda e: confounding_scores.get(e, 0))
            causal_edges = sorted_edges[:int(len(interactions) * 0.5)]
        
        return causal_edges
    
    def create_confounding_preserving_view(self, interactions: List[Tuple[int, int]],
                                           causal_scores: Dict) -> List[Tuple[int, int]]:
        confounding_edges = [e for e in interactions
                            if causal_scores.get(e, 0) < self.causal_threshold]
        
        if len(confounding_edges) < len(interactions) * 0.5:
            sorted_edges = sorted(interactions,
                                 key=lambda e: causal_scores.get(e, 0))
            confounding_edges = sorted_edges[:int(len(interactions) * 0.5)]
        
        return confounding_edges


class ContrastiveCausalSSL(nn.Module):
    """Contrastive Causal Self-Supervised Learning Module"""
    
    def __init__(self, embed_dim: int, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def compute_invariance_loss(self, z_original: torch.Tensor,
                                z_augmented: torch.Tensor,
                                batch_indices: torch.Tensor) -> torch.Tensor:
        z_orig_proj = F.normalize(self.projector(z_original[batch_indices]), dim=-1)
        z_aug_proj = F.normalize(self.projector(z_augmented[batch_indices]), dim=-1)
        
        sim_matrix = torch.mm(z_orig_proj, z_aug_proj.t()) / self.temperature
        
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def compute_disentanglement_loss(self, z_causal: torch.Tensor,
                                     z_confounding: torch.Tensor) -> torch.Tensor:
        z_c_norm = F.normalize(z_causal, dim=-1)
        z_s_norm = F.normalize(z_confounding, dim=-1)
        
        similarity = (z_c_norm * z_s_norm).sum(dim=-1)
        loss = (similarity ** 2).mean()
        
        return loss
    
    def compute_counterfactual_loss(self, z_causal_orig: torch.Tensor,
                                    z_causal_aug: torch.Tensor,
                                    z_confound_orig: torch.Tensor,
                                    z_confound_aug: torch.Tensor,
                                    batch_indices: torch.Tensor) -> torch.Tensor:
        causal_diff = (z_causal_orig[batch_indices] - z_causal_aug[batch_indices]).norm(dim=-1)
        confound_diff = (z_confound_orig[batch_indices] - z_confound_aug[batch_indices]).norm(dim=-1)
        
        loss = F.relu(confound_diff - causal_diff + 0.1).mean()
        
        return loss