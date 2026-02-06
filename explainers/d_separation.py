"""
D-Separation Analysis for Causal Graph Structure
"""

import numpy as np
import torch
from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict


class CausalGraphBuilder:
    """
    Builds causal graph structure from learned model parameters.
    """
    
    def __init__(self, n_factors: int):
        """
        Initialize CausalGraphBuilder.
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.adjacency = np.zeros((n_factors, n_factors))
        self.edge_weights = np.zeros((n_factors, n_factors))
    
    def build_from_gates(self, causal_gates: List[List[torch.nn.Parameter]],
                         threshold: float = 0.3) -> np.ndarray:
        """
        Build causal graph from gating parameters.
        
        Args:
            causal_gates: List of gating parameters per layer
            threshold: Correlation threshold for edge creation
            
        Returns:
            Adjacency matrix of causal graph
        """
        # Extract gate values across layers
        gate_values = []
        for layer_gates in causal_gates:
            layer_vals = []
            for g in layer_gates:
                with torch.no_grad():
                    layer_vals.append(torch.sigmoid(g).mean().item())
            gate_values.append(layer_vals)
        
        gate_matrix = np.array(gate_values)  # [n_layers, n_factors]
        
        # Compute correlations between factors
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                if gate_matrix.shape[0] > 1:
                    corr = np.corrcoef(gate_matrix[:, i], gate_matrix[:, j])[0, 1]
                else:
                    corr = 0.0
                
                if not np.isnan(corr):
                    self.edge_weights[i, j] = abs(corr)
                    self.edge_weights[j, i] = abs(corr)
                    
                    if abs(corr) > threshold:
                        self.adjacency[i, j] = 1
                        self.adjacency[j, i] = 1
        
        return self.adjacency
    
    def build_from_attention(self, attention_weights: torch.Tensor,
                            threshold: float = 0.1) -> np.ndarray:
        """
        Build causal graph from attention weights.
        
        Args:
            attention_weights: Attention weight matrix [n_factors, n_factors]
            threshold: Weight threshold for edge creation
            
        Returns:
            Adjacency matrix
        """
        with torch.no_grad():
            weights = attention_weights.cpu().numpy()
        
        self.edge_weights = np.abs(weights)
        self.adjacency = (self.edge_weights > threshold).astype(float)
        
        return self.adjacency
    
    def get_moral_graph(self) -> np.ndarray:
        """
        Compute moral graph (for d-separation analysis).
        Moralization connects all parents of each node.
        
        Returns:
            Moral graph adjacency matrix
        """
        moral = self.adjacency.copy()
        
        # For each node, connect all its parents
        for node in range(self.n_factors):
            parents = np.where(self.adjacency[:, node] > 0)[0]
            for i, p1 in enumerate(parents):
                for p2 in parents[i+1:]:
                    moral[p1, p2] = 1
                    moral[p2, p1] = 1
        
        return moral


class DSeparationAnalyzer:
    """
    Analyzes d-separation structure in causal graphs.
    Used for factorizing Shapley value computation.
    """
    
    def __init__(self, n_factors: int):
        """
        Initialize DSeparationAnalyzer.
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.graph_builder = CausalGraphBuilder(n_factors)
        self.adjacency = None
        self.moral_graph = None
    
    def learn_structure_from_gates(self, causal_gates: List[List[torch.nn.Parameter]],
                                   threshold: float = 0.3):
        """
        Learn causal structure from gating parameters.
        
        Args:
            causal_gates: Gating parameters
            threshold: Correlation threshold
        """
        self.adjacency = self.graph_builder.build_from_gates(causal_gates, threshold)
        self.moral_graph = self.graph_builder.get_moral_graph()
    
    def learn_structure_from_attention(self, attention_weights: torch.Tensor,
                                       threshold: float = 0.1):
        """
        Learn causal structure from attention weights.
        
        Args:
            attention_weights: Attention matrix
            threshold: Weight threshold
        """
        self.adjacency = self.graph_builder.build_from_attention(
            attention_weights, threshold
        )
        self.moral_graph = self.graph_builder.get_moral_graph()
    
    def find_d_separated_cliques(self) -> List[Set[int]]:
        """
        Find maximal d-separated cliques.
        
        Uses connected components of the moral graph to identify
        conditionally independent factor groups.
        
        Returns:
            List of d-separated cliques (each clique is a set of factor indices)
        """
        if self.moral_graph is None:
            return [set(range(self.n_factors))]
        
        visited = set()
        cliques = []
        
        for start in range(self.n_factors):
            if start in visited:
                continue
            
            clique = self._bfs_component(start, visited)
            if clique:
                cliques.append(clique)
        
        return cliques if cliques else [set(range(self.n_factors))]
    
    def _bfs_component(self, start: int, visited: Set[int]) -> Set[int]:
        """Find connected component using BFS"""
        component = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            for neighbor in range(self.n_factors):
                if self.moral_graph[node, neighbor] > 0 and neighbor not in visited:
                    queue.append(neighbor)
        
        return component
    
    def check_d_separated(self, x: Set[int], y: Set[int], 
                          z: Set[int]) -> bool:
        """
        Check if X and Y are d-separated given Z.
        
        Uses the Bayes-Ball algorithm.
        
        Args:
            x: First set of nodes
            y: Second set of nodes
            z: Conditioning set
            
        Returns:
            True if X ⊥ Y | Z in the graph
        """
        if self.adjacency is None:
            return False
        
        # Simplified d-separation check using reachability
        reachable = self._find_reachable(x, z)
        return len(reachable & y) == 0
    
    def _find_reachable(self, sources: Set[int], blocked: Set[int]) -> Set[int]:
        """Find nodes reachable from sources avoiding blocked nodes"""
        reachable = set()
        queue = list(sources)
        
        while queue:
            node = queue.pop(0)
            if node in reachable or node in blocked:
                continue
            
            reachable.add(node)
            
            for neighbor in range(self.n_factors):
                if self.adjacency[node, neighbor] > 0:
                    queue.append(neighbor)
        
        return reachable - sources
    
    def compute_complexity_reduction(self) -> Dict[str, float]:
        """
        Compute theoretical complexity reduction from factorization.
        
        Returns:
            Dictionary with complexity metrics
        """
        cliques = self.find_d_separated_cliques()
        
        n = self.n_factors
        exact_complexity = 2 ** n
        
        factored_complexity = sum(2 ** len(c) for c in cliques)
        inter_clique = len(cliques) ** 2 * max(len(c) for c in cliques)
        total_factored = factored_complexity + inter_clique
        
        return {
            'n_factors': n,
            'n_cliques': len(cliques),
            'max_clique_size': max(len(c) for c in cliques),
            'exact_complexity': exact_complexity,
            'factored_complexity': total_factored,
            'speedup': exact_complexity / total_factored if total_factored > 0 else 1.0
        }
    
    def get_clique_structure(self) -> Dict:
        """
        Get detailed clique structure information.
        
        Returns:
            Dictionary with clique details
        """
        cliques = self.find_d_separated_cliques()
        
        return {
            'cliques': [list(c) for c in cliques],
            'n_cliques': len(cliques),
            'clique_sizes': [len(c) for c in cliques],
            'adjacency': self.adjacency.tolist() if self.adjacency is not None else None,
            'moral_graph': self.moral_graph.tolist() if self.moral_graph is not None else None
        }