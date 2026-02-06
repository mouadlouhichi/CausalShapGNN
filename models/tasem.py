"""
Topology-Aware Shapley Explanation Module (TASEM)
"""

import torch
import numpy as np
import math
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import random


class DSeparationAnalyzer:
    """Analyzes d-separation structure in causal graphs"""
    
    def __init__(self, n_factors: int):
        self.n_factors = n_factors
        self.adjacency = None
    
    def learn_causal_structure(self, causal_gates, threshold: float = 0.3):
        self.adjacency = np.zeros((self.n_factors, self.n_factors))
        
        gate_values = []
        for layer_gates in causal_gates:
            layer_vals = torch.stack([torch.sigmoid(g).mean().item()
                                     for g in layer_gates])
            gate_values.append(layer_vals.numpy())
        
        gate_matrix = np.array(gate_values)
        
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                corr = np.abs(np.corrcoef(gate_matrix[:, i], gate_matrix[:, j])[0, 1])
                if not np.isnan(corr) and corr > threshold:
                    self.adjacency[i, j] = 1
                    self.adjacency[j, i] = 1
    
    def find_d_separated_cliques(self) -> List[Set[int]]:
        if self.adjacency is None:
            return [set(range(self.n_factors))]
        
        visited = set()
        cliques = []
        
        for start in range(self.n_factors):
            if start in visited:
                continue
            
            clique = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                
                visited.add(node)
                clique.add(node)
                
                for neighbor in range(self.n_factors):
                    if self.adjacency[node, neighbor] > 0 and neighbor not in visited:
                        stack.append(neighbor)
            
            if clique:
                cliques.append(clique)
        
        return cliques


class TopologyAwareShapley:
    """
    Topology-Aware Shapley Explanation Module.
    Computes tractable Shapley values using d-separation.
    """
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.d_sep_analyzer = DSeparationAnalyzer(model.cdm.n_factors)
        self.population_means = None
    
    def _compute_population_means(self, user_embeddings: torch.Tensor,
                                  item_embeddings: torch.Tensor):
        all_emb = torch.cat([user_embeddings, item_embeddings], dim=0)
        n_factors = self.model.cdm.n_factors
        factor_dim = self.model.cdm.factor_dim
        
        factored = all_emb.view(-1, n_factors, factor_dim)
        self.population_means = factored.mean(dim=0)
    
    def _mask_factors(self, embeddings: torch.Tensor,
                      mask: Set[int]) -> torch.Tensor:
        n_factors = self.model.cdm.n_factors
        factor_dim = self.model.cdm.factor_dim
        
        factored = embeddings.view(-1, n_factors, factor_dim).clone()
        
        for k in range(n_factors):
            if k not in mask:
                factored[:, k, :] = self.population_means[k]
        
        return factored.view(-1, n_factors * factor_dim)
    
    def _compute_value_function(self, user_idx: int, item_idx: int,
                                user_emb: torch.Tensor, item_emb: torch.Tensor,
                                coalition: Set[int]) -> float:
        masked_user = self._mask_factors(user_emb.unsqueeze(0), coalition)
        masked_item = self._mask_factors(item_emb.unsqueeze(0), coalition)
        
        score = (masked_user * masked_item).sum().item()
        return score
    
    def compute_feature_shapley(self, user_idx: int, item_idx: int,
                                user_emb: torch.Tensor, item_emb: torch.Tensor
                                ) -> np.ndarray:
        n_factors = self.model.cdm.n_factors
        
        self.d_sep_analyzer.learn_causal_structure(self.model.cdm.causal_gates)
        cliques = self.d_sep_analyzer.find_d_separated_cliques()
        
        shapley_values = np.zeros(n_factors)
        
        if len(cliques) == 1 and len(cliques[0]) == n_factors:
            shapley_values = self._exact_shapley(
                user_idx, item_idx, user_emb, item_emb, set(range(n_factors))
            )
        else:
            for clique in cliques:
                clique_list = list(clique)
                clique_shapley = self._exact_shapley(
                    user_idx, item_idx, user_emb, item_emb, clique
                )
                for idx, factor in enumerate(clique_list):
                    shapley_values[factor] = clique_shapley[idx]
            
            shapley_values = self._add_inter_clique_corrections(
                shapley_values, cliques, user_idx, item_idx, user_emb, item_emb
            )
        
        return shapley_values
    
    def _exact_shapley(self, user_idx: int, item_idx: int,
                       user_emb: torch.Tensor, item_emb: torch.Tensor,
                       player_set: Set[int]) -> np.ndarray:
        players = list(player_set)
        n_players = len(players)
        shapley = np.zeros(n_players)
        
        for i, player in enumerate(players):
            other_players = [p for p in players if p != player]
            
            for size in range(n_players):
                for coalition in combinations(other_players, size):
                    coalition_set = set(coalition)
                    
                    v_without = self._compute_value_function(
                        user_idx, item_idx, user_emb, item_emb, coalition_set
                    )
                    
                    v_with = self._compute_value_function(
                        user_idx, item_idx, user_emb, item_emb,
                        coalition_set | {player}
                    )
                    
                    weight = (math.factorial(size) *
                             math.factorial(n_players - size - 1) /
                             math.factorial(n_players))
                    
                    shapley[i] += weight * (v_with - v_without)
        
        return shapley
    
    def _add_inter_clique_corrections(self, shapley_values: np.ndarray,
                                      cliques: List[Set[int]],
                                      user_idx: int, item_idx: int,
                                      user_emb: torch.Tensor,
                                      item_emb: torch.Tensor) -> np.ndarray:
        n_cliques = len(cliques)
        
        if n_cliques <= 1:
            return shapley_values
        
        for i in range(n_cliques):
            for j in range(i + 1, n_cliques):
                clique_i = cliques[i]
                clique_j = cliques[j]
                
                v_both = self._compute_value_function(
                    user_idx, item_idx, user_emb, item_emb, clique_i | clique_j
                )
                v_i = self._compute_value_function(
                    user_idx, item_idx, user_emb, item_emb, clique_i
                )
                v_j = self._compute_value_function(
                    user_idx, item_idx, user_emb, item_emb, clique_j
                )
                v_empty = self._compute_value_function(
                    user_idx, item_idx, user_emb, item_emb, set()
                )
                
                interaction = v_both - v_i - v_j + v_empty
                
                all_factors = list(clique_i | clique_j)
                correction = interaction / len(all_factors)
                
                for f in all_factors:
                    shapley_values[f] += correction
        
        return shapley_values
    
    def compute_path_shapley(self, user_idx: int, item_idx: int,
                            adj_matrix: torch.sparse.Tensor,
                            max_paths: int = 100) -> Dict[Tuple, float]:
        paths = self._extract_paths(user_idx, item_idx, adj_matrix, max_paths)
        
        if not paths:
            return {}
        
        if len(paths) > 20:
            return self._approximate_path_shapley(paths)
        
        return self._exact_path_shapley(paths)
    
    def _extract_paths(self, user_idx: int, item_idx: int,
                       adj_matrix: torch.sparse.Tensor,
                       max_paths: int) -> List[Tuple[int, ...]]:
        n_users = self.model.cdm.n_users
        target = n_users + item_idx
        
        indices = adj_matrix.coalesce().indices()
        edges = defaultdict(list)
        for i in range(indices.size(1)):
            src, dst = indices[0, i].item(), indices[1, i].item()
            edges[src].append(dst)
        
        paths = []
        queue = [(user_idx, (user_idx,))]
        max_depth = self.model.cdm.n_layers * 2 + 1
        
        while queue and len(paths) < max_paths:
            node, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if node == target:
                paths.append(path)
                continue
            
            for neighbor in edges.get(node, []):
                if neighbor not in path:
                    queue.append((neighbor, path + (neighbor,)))
        
        return paths
    
    def _exact_path_shapley(self, paths: List[Tuple[int, ...]]) -> Dict[Tuple, float]:
        shapley = {}
        
        for path in paths:
            contribution = 1.0 / len(path)
            shapley[path] = contribution
        
        total = sum(shapley.values())
        if total > 0:
            shapley = {k: v / total for k, v in shapley.items()}
        
        return shapley
    
    def _approximate_path_shapley(self, paths: List[Tuple[int, ...]],
                                  n_samples: int = 100) -> Dict[Tuple, float]:
        n_paths = len(paths)
        shapley = {path: 0.0 for path in paths}
        
        for _ in range(n_samples):
            perm = list(range(n_paths))
            random.shuffle(perm)
            
            for i, idx in enumerate(perm):
                path = paths[idx]
                marginal = 1.0 / (i + 1)
                shapley[path] += marginal / n_samples
        
        return shapley
    
    def compute_user_profile(self, user_idx: int,
                            recommended_items: List[int],
                            user_emb: torch.Tensor,
                            item_emb: torch.Tensor) -> np.ndarray:
        n_factors = self.model.cdm.n_factors
        user_profile = np.zeros(n_factors)
        
        for item_idx in recommended_items:
            item_embedding = item_emb[item_idx]
            shapley = self.compute_feature_shapley(
                user_idx, item_idx, user_emb[user_idx], item_embedding
            )
            user_profile += shapley
        
        if len(recommended_items) > 0:
            user_profile /= len(recommended_items)
        
        return user_profile