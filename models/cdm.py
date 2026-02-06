"""
Causal Disentanglement Module (CDM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


class StructuralCausalModel(nn.Module):
    """Structural Causal Model over latent factors"""
    
    def __init__(self, n_factors: int, factor_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_factors = n_factors
        self.factor_dim = factor_dim
        
        self.causal_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(factor_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, factor_dim * 2)
            ) for _ in range(n_factors)
        ])
        
        self.confounding_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(factor_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, factor_dim * 2)
            ) for _ in range(n_factors)
        ])
        
        self.causal_attention = nn.ParameterList([
            nn.Parameter(torch.randn(factor_dim, factor_dim) * 0.01)
            for _ in range(n_factors)
        ])
        
        self.confounding_attention = nn.ParameterList([
            nn.Parameter(torch.randn(factor_dim, factor_dim) * 0.01)
            for _ in range(n_factors)
        ])
    
    def encode_causal(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                      factor_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([user_emb, item_emb], dim=-1)
        params = self.causal_encoder[factor_idx](combined)
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var
    
    def encode_confounding(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                          factor_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([user_emb, item_emb], dim=-1)
        params = self.confounding_encoder[factor_idx](combined)
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class CausalDisentanglementModule(nn.Module):
    """
    Causal Disentanglement Module.
    Decomposes GNN message-passing into causal and confounding channels.
    """
    
    def __init__(self, n_users: int, n_items: int, embed_dim: int,
                 n_factors: int, n_layers: int, device: torch.device):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_factors = n_factors
        self.factor_dim = embed_dim // n_factors
        self.n_layers = n_layers
        self.device = device
        
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        self.scm = StructuralCausalModel(n_factors, self.factor_dim)
        
        self.causal_gates = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(self.factor_dim))
                for _ in range(n_factors)
            ]) for _ in range(n_layers)
        ])
        
        self.register_buffer('population_mean_causal',
                            torch.zeros(n_factors, self.factor_dim))
        self.register_buffer('population_mean_confounding',
                            torch.zeros(n_factors, self.factor_dim))
    
    def _split_factors(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        return embeddings.view(batch_size, self.n_factors, self.factor_dim)
    
    def _merge_factors(self, factors: torch.Tensor) -> torch.Tensor:
        batch_size = factors.size(0)
        return factors.view(batch_size, -1)
    
    def forward(self, adj_norm: torch.sparse.Tensor,
                use_causal_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        all_mu_c, all_logvar_c = [], []
        all_mu_s, all_logvar_s = [], []
        causal_representations = []
        confounding_representations = []
        
        current_embeddings = all_embeddings
        
        for layer in range(self.n_layers):
            messages = torch.sparse.mm(adj_norm, current_embeddings)
            
            messages_factored = self._split_factors(messages)
            current_factored = self._split_factors(current_embeddings)
            
            causal_messages = []
            confounding_messages = []
            
            for k in range(self.n_factors):
                msg_k = messages_factored[:, k, :]
                curr_k = current_factored[:, k, :]
                
                mu_c, logvar_c = self.scm.encode_causal(curr_k, msg_k, k)
                mu_s, logvar_s = self.scm.encode_confounding(curr_k, msg_k, k)
                
                all_mu_c.append(mu_c)
                all_logvar_c.append(logvar_c)
                all_mu_s.append(mu_s)
                all_logvar_s.append(logvar_s)
                
                z_c = self.scm.reparameterize(mu_c, logvar_c)
                z_s = self.scm.reparameterize(mu_s, logvar_s)
                
                gate = torch.sigmoid(self.causal_gates[layer][k])
                
                if use_causal_only:
                    fused = z_c
                else:
                    fused = gate * z_c + (1 - gate) * z_s
                
                causal_messages.append(z_c)
                confounding_messages.append(z_s)
            
            causal_stacked = torch.stack(causal_messages, dim=1)
            confounding_stacked = torch.stack(confounding_messages, dim=1)
            
            causal_representations.append(self._merge_factors(causal_stacked))
            confounding_representations.append(self._merge_factors(confounding_stacked))
            
            gates = torch.stack([torch.sigmoid(self.causal_gates[layer][k])
                                for k in range(self.n_factors)], dim=0)
            gates = gates.unsqueeze(0).expand(messages_factored.size(0), -1, -1)
            
            if use_causal_only:
                current_embeddings = self._merge_factors(causal_stacked)
            else:
                fused = gates * causal_stacked + (1 - gates) * confounding_stacked
                current_embeddings = self._merge_factors(fused)
            
            embeddings_list.append(current_embeddings)
        
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)
        
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]
        
        aux_info = {
            'mu_causal': all_mu_c,
            'logvar_causal': all_logvar_c,
            'mu_confounding': all_mu_s,
            'logvar_confounding': all_logvar_s,
            'causal_repr': causal_representations,
            'confounding_repr': confounding_representations,
            'embeddings_list': embeddings_list
        }
        
        return user_embeddings, item_embeddings, aux_info
    
    def compute_cdm_loss(self, aux_info: Dict,
                         beta: float = 0.01, gamma: float = 0.01) -> torch.Tensor:
        kl_causal = 0.0
        kl_confounding = 0.0
        
        for mu_c, logvar_c in zip(aux_info['mu_causal'], aux_info['logvar_causal']):
            kl_causal += -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
        
        for mu_s, logvar_s in zip(aux_info['mu_confounding'], aux_info['logvar_confounding']):
            kl_confounding += -0.5 * torch.sum(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
        
        n_samples = aux_info['mu_causal'][0].size(0)
        kl_causal /= n_samples
        kl_confounding /= n_samples
        
        return beta * kl_causal + gamma * kl_confounding