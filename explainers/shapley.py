"""
Shapley Value Computation for CausalShapGNN
Implements feature-level, path-level, and user-level Shapley explanations
"""

import torch
import numpy as np
import math
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union
import random
from abc import ABC, abstractmethod

from .d_separation import DSeparationAnalyzer


class ShapleyExplainer(ABC):
    """Abstract base class for Shapley explainers"""
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[np.ndarray, Dict]:
        """Compute Shapley values"""
        pass
    
    @staticmethod
    def shapley_weight(n_players: int, coalition_size: int) -> float:
        """
        Compute Shapley weight for a coalition.
        
        Args:
            n_players: Total number of players
            coalition_size: Size of coalition (excluding the player)
            
        Returns:
            Shapley weight
        """
        return (math.factorial(coalition_size) * 
                math.factorial(n_players - coalition_size - 1) / 
                math.factorial(n_players))


class FeatureShapley(ShapleyExplainer):
    """
    Feature-level Shapley values for latent factors.
    Explains which latent factors drive a recommendation.
    """
    
    def __init__(self, model, device: torch.device):
        """
        Initialize FeatureShapley explainer.
        
        Args:
            model: CausalShapGNN model
            device: Torch device
        """
        self.model = model
        self.device = device
        self.n_factors = model.cdm.n_factors
        self.factor_dim = model.cdm.factor_dim
        self.d_sep_analyzer = DSeparationAnalyzer(self.n_factors)
        self.population_means = None
    
    def _compute_population_means(self, user_emb: torch.Tensor, 
                                  item_emb: torch.Tensor):
        """Compute population mean embeddings for masking"""
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        factored = all_emb.view(-1, self.n_factors, self.factor_dim)
        self.population_means = factored.mean(dim=0)
    
    def _mask_factors(self, embedding: torch.Tensor, 
                      active_factors: Set[int]) -> torch.Tensor:
        """
        Mask factors not in active set with population mean.
        
        Args:
            embedding: Node embedding [embed_dim]
            active_factors: Set of factor indices to keep
            
        Returns:
            Masked embedding
        """
        factored = embedding.view(self.n_factors, self.factor_dim).clone()
        
        for k in range(self.n_factors):
            if k not in active_factors:
                factored[k] = self.population_means[k]
        
        return factored.view(-1)
    
    def _value_function(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                        coalition: Set[int]) -> float:
        """
        Compute value function v(S) for a coalition of factors.
        
        Args:
            user_emb: User embedding
            item_emb: Item embedding
            coalition: Set of active factor indices
            
        Returns:
            Prediction score using only factors in coalition
        """
        masked_user = self._mask_factors(user_emb, coalition)
        masked_item = self._mask_factors(item_emb, coalition)
        return (masked_user * masked_item).sum().item()
    
    def compute_exact(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                      player_set: Optional[Set[int]] = None) -> np.ndarray:
        """
        Compute exact Shapley values for factors.
        
        Args:
            user_emb: User embedding
            item_emb: Item embedding
            player_set: Set of factors to consider (default: all)
            
        Returns:
            Array of Shapley values for each factor
        """
        if player_set is None:
            player_set = set(range(self.n_factors))
        
        players = list(player_set)
        n_players = len(players)
        shapley = np.zeros(self.n_factors)
        
        for i, player in enumerate(players):
            other_players = [p for p in players if p != player]
            
            for size in range(n_players):
                for coalition in combinations(other_players, size):
                    coalition_set = set(coalition)
                    
                    v_without = self._value_function(user_emb, item_emb, coalition_set)
                    v_with = self._value_function(user_emb, item_emb, 
                                                  coalition_set | {player})
                    
                    weight = self.shapley_weight(n_players, size)
                    shapley[player] += weight * (v_with - v_without)
        
        return shapley
    
    def compute_factorized(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                           cliques: List[Set[int]]) -> np.ndarray:
        """
        Compute Shapley values using d-separation factorization.
        
        Args:
            user_emb: User embedding
            item_emb: Item embedding
            cliques: List of d-separated cliques
            
        Returns:
            Array of Shapley values for each factor
        """
        shapley = np.zeros(self.n_factors)
        
        # Compute within-clique Shapley values
        for clique in cliques:
            clique_shapley = self.compute_exact(user_emb, item_emb, clique)
            for factor in clique:
                shapley[factor] = clique_shapley[factor]
        
        # Add inter-clique corrections
        shapley = self._inter_clique_corrections(shapley, cliques, 
                                                  user_emb, item_emb)
        
        return shapley
    
    def _inter_clique_corrections(self, shapley: np.ndarray,
                                   cliques: List[Set[int]],
                                   user_emb: torch.Tensor,
                                   item_emb: torch.Tensor) -> np.ndarray:
        """Add corrections for inter-clique interactions"""
        n_cliques = len(cliques)
        
        if n_cliques <= 1:
            return shapley
        
        for i in range(n_cliques):
            for j in range(i + 1, n_cliques):
                v_both = self._value_function(user_emb, item_emb, 
                                              cliques[i] | cliques[j])
                v_i = self._value_function(user_emb, item_emb, cliques[i])
                v_j = self._value_function(user_emb, item_emb, cliques[j])
                v_empty = self._value_function(user_emb, item_emb, set())
                
                interaction = v_both - v_i - v_j + v_empty
                
                all_factors = list(cliques[i] | cliques[j])
                correction = interaction / len(all_factors)
                
                for f in all_factors:
                    shapley[f] += correction
        
        return shapley
    
    def compute(self, user_idx: int, item_idx: int,
                user_emb: torch.Tensor, item_emb: torch.Tensor,
                use_factorization: bool = True) -> np.ndarray:
        """
        Compute feature-level Shapley values.
        
        Args:
            user_idx: User index
            item_idx: Item index
            user_emb: All user embeddings
            item_emb: All item embeddings
            use_factorization: Whether to use d-sep factorization
            
        Returns:
            Array of Shapley values for each factor
        """
        if self.population_means is None:
            self._compute_population_means(user_emb, item_emb)
        
        u_emb = user_emb[user_idx]
        i_emb = item_emb[item_idx]
        
        if use_factorization:
            # Learn causal structure and find cliques
            self.d_sep_analyzer.learn_structure_from_gates(
                self.model.cdm.causal_gates
            )
            cliques = self.d_sep_analyzer.find_d_separated_cliques()
            
            # Use factorization if beneficial
            max_clique_size = max(len(c) for c in cliques) if cliques else self.n_factors
            
            if max_clique_size < self.n_factors:
                return self.compute_factorized(u_emb, i_emb, cliques)
        
        return self.compute_exact(u_emb, i_emb)


class PathShapley(ShapleyExplainer):
    """
    Path-level Shapley values.
    Explains which interaction paths carry causal signal.
    """
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.n_users = model.cdm.n_users
    
    def extract_paths(self, user_idx: int, item_idx: int,
                      adj_matrix: torch.sparse.Tensor,
                      max_depth: int = 4,
                      max_paths: int = 100) -> List[Tuple[int, ...]]:
        """
        Extract L-hop paths from user to item.
        
        Args:
            user_idx: Source user index
            item_idx: Target item index
            adj_matrix: Sparse adjacency matrix
            max_depth: Maximum path length
            max_paths: Maximum number of paths to extract
            
        Returns:
            List of paths (each path is a tuple of node indices)
        """
        target = self.n_users + item_idx
        
        # Build edge dictionary
        indices = adj_matrix.coalesce().indices()
        edges = defaultdict(list)
        for i in range(indices.size(1)):
            src, dst = indices[0, i].item(), indices[1, i].item()
            edges[src].append(dst)
        
        # BFS to find paths
        paths = []
        queue = [(user_idx, (user_idx,))]
        
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
    
    def compute_exact(self, paths: List[Tuple[int, ...]],
                      path_scores: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        Compute exact Shapley values for paths.
        
        Args:
            paths: List of paths
            path_scores: Score contribution of each path
            
        Returns:
            Dictionary mapping paths to Shapley values
        """
        n_paths = len(paths)
        shapley = {path: 0.0 for path in paths}
        
        if n_paths == 0:
            return shapley
        
        for i, path in enumerate(paths):
            other_paths = [p for p in paths if p != path]
            
            for size in range(n_paths):
                for coalition in combinations(range(len(other_paths)), size):
                    coalition_paths = {other_paths[j] for j in coalition}
                    
                    v_without = sum(path_scores.get(p, 0) for p in coalition_paths)
                    v_with = v_without + path_scores.get(path, 0)
                    
                    weight = self.shapley_weight(n_paths, size)
                    shapley[path] += weight * (v_with - v_without)
        
        return shapley
    
    def compute_approximate(self, paths: List[Tuple[int, ...]],
                           path_scores: Dict[Tuple, float],
                           n_samples: int = 1000) -> Dict[Tuple, float]:
        """
        Approximate Shapley values using Monte Carlo sampling.
        
        Args:
            paths: List of paths
            path_scores: Score contribution of each path
            n_samples: Number of permutation samples
            
        Returns:
            Dictionary mapping paths to Shapley values
        """
        n_paths = len(paths)
        shapley = {path: 0.0 for path in paths}
        
        if n_paths == 0:
            return shapley
        
        for _ in range(n_samples):
            perm = list(range(n_paths))
            random.shuffle(perm)
            
            cumulative_value = 0.0
            
            for i, idx in enumerate(perm):
                path = paths[idx]
                new_value = cumulative_value + path_scores.get(path, 0)
                marginal = new_value - cumulative_value
                shapley[path] += marginal / n_samples
                cumulative_value = new_value
        
        return shapley
    
    def compute(self, user_idx: int, item_idx: int,
                adj_matrix: torch.sparse.Tensor,
                user_emb: torch.Tensor, item_emb: torch.Tensor,
                max_paths: int = 100) -> Dict[Tuple, float]:
        """
        Compute path-level Shapley values.
        
        Args:
            user_idx: User index
            item_idx: Item index
            adj_matrix: Adjacency matrix
            user_emb: User embeddings
            item_emb: Item embeddings
            max_paths: Maximum paths to consider
            
        Returns:
            Dictionary mapping paths to Shapley values
        """
        paths = self.extract_paths(user_idx, item_idx, adj_matrix, 
                                   max_paths=max_paths)
        
        if not paths:
            return {}
        
        # Compute path scores based on intermediate node embeddings
        path_scores = {}
        for path in paths:
            score = 0.0
            for i in range(len(path) - 1):
                if path[i] < self.n_users:
                    node_emb = user_emb[path[i]]
                else:
                    node_emb = item_emb[path[i] - self.n_users]
                
                if path[i + 1] < self.n_users:
                    next_emb = user_emb[path[i + 1]]
                else:
                    next_emb = item_emb[path[i + 1] - self.n_users]
                
                score += (node_emb * next_emb).sum().item()
            
            path_scores[path] = score / len(path)
        
        # Choose computation method based on number of paths
        if len(paths) <= 15:
            return self.compute_exact(paths, path_scores)
        else:
            return self.compute_approximate(paths, path_scores)


class UserProfileShapley:
    """
    User-level Shapley profile.
    Aggregates factor importance across all recommendations for a user.
    """
    
    def __init__(self, feature_explainer: FeatureShapley):
        """
        Initialize UserProfileShapley.
        
        Args:
            feature_explainer: FeatureShapley instance
        """
        self.feature_explainer = feature_explainer
    
    def compute(self, user_idx: int, recommended_items: List[int],
                user_emb: torch.Tensor, item_emb: torch.Tensor) -> np.ndarray:
        """
        Compute user preference profile.
        
        Ψ_u(k) = (1/|R_u|) Σ_{i∈R_u} φ_k(u, i)
        
        Args:
            user_idx: User index
            recommended_items: List of recommended item indices
            user_emb: User embeddings
            item_emb: Item embeddings
            
        Returns:
            User profile array [n_factors]
        """
        n_factors = self.feature_explainer.n_factors
        profile = np.zeros(n_factors)
        
        if not recommended_items:
            return profile
        
        for item_idx in recommended_items:
            shapley = self.feature_explainer.compute(
                user_idx, item_idx, user_emb, item_emb
            )
            profile += shapley
        
        profile /= len(recommended_items)
        
        return profile
    
    def compute_batch(self, user_indices: List[int],
                      user_recommendations: Dict[int, List[int]],
                      user_emb: torch.Tensor,
                      item_emb: torch.Tensor) -> Dict[int, np.ndarray]:
        """
        Compute profiles for multiple users.
        
        Args:
            user_indices: List of user indices
            user_recommendations: Dict mapping user to recommended items
            user_emb: User embeddings
            item_emb: Item embeddings
            
        Returns:
            Dictionary mapping user indices to profiles
        """
        profiles = {}
        
        for user_idx in user_indices:
            items = user_recommendations.get(user_idx, [])
            profiles[user_idx] = self.compute(user_idx, items, user_emb, item_emb)
        
        return profiles