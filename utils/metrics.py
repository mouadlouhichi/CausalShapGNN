"""
Evaluation Metrics for CausalShapGNN
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Compute Recall@K.
    
    Args:
        recommended: List of recommended items (ranked)
        ground_truth: Set of relevant items
        k: Cutoff
        
    Returns:
        Recall@K value
    """
    if not ground_truth:
        return 0.0
    
    top_k = set(recommended[:k])
    hits = len(top_k & ground_truth)
    
    return hits / min(k, len(ground_truth))


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Compute NDCG@K.
    
    Args:
        recommended: List of recommended items (ranked)
        ground_truth: Set of relevant items
        k: Cutoff
        
    Returns:
        NDCG@K value
    """
    if not ground_truth:
        return 0.0
    
    # DCG
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 2)
    
    # IDCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: List[int], ground_truth: Set[int]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        recommended: List of recommended items
        ground_truth: Set of relevant items
        
    Returns:
        MRR value
    """
    for idx, item in enumerate(recommended):
        if item in ground_truth:
            return 1.0 / (idx + 1)
    return 0.0


def hit_rate_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Compute Hit Rate@K (whether any relevant item is in top-k).
    
    Args:
        recommended: List of recommended items
        ground_truth: Set of relevant items
        k: Cutoff
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = set(recommended[:k])
    return 1.0 if len(top_k & ground_truth) > 0 else 0.0


def precision_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Compute Precision@K.
    
    Args:
        recommended: List of recommended items
        ground_truth: Set of relevant items
        k: Cutoff
        
    Returns:
        Precision@K value
    """
    top_k = set(recommended[:k])
    hits = len(top_k & ground_truth)
    return hits / k


def map_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Compute Mean Average Precision@K.
    
    Args:
        recommended: List of recommended items
        ground_truth: Set of relevant items
        k: Cutoff
        
    Returns:
        MAP@K value
    """
    if not ground_truth:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    
    for idx, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            sum_precision += hits / (idx + 1)
    
    return sum_precision / min(k, len(ground_truth))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for measuring inequality.
    Lower values indicate more equal distribution.
    
    Args:
        values: Array of values (e.g., item recommendation counts)
        
    Returns:
        Gini coefficient in [0, 1]
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0.0
    
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values) - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))


def coverage_at_k(all_recommendations: List[List[int]], 
                  n_items: int, k: int) -> float:
    """
    Compute catalog coverage at K.
    
    Args:
        all_recommendations: List of recommendation lists for all users
        n_items: Total number of items
        k: Cutoff
        
    Returns:
        Fraction of items recommended at least once
    """
    recommended_items = set()
    
    for recs in all_recommendations:
        recommended_items.update(recs[:k])
    
    return len(recommended_items) / n_items


def novelty_at_k(recommendations: List[int], item_popularity: Dict[int, int],
                 n_total_interactions: int, k: int) -> float:
    """
    Compute novelty (self-information) at K.
    
    Args:
        recommendations: List of recommended items
        item_popularity: Dict mapping items to interaction counts
        n_total_interactions: Total number of interactions
        k: Cutoff
        
    Returns:
        Average novelty of top-k recommendations
    """
    if not recommendations or k == 0:
        return 0.0
    
    novelty = 0.0
    for item in recommendations[:k]:
        pop = item_popularity.get(item, 1)
        prob = pop / n_total_interactions
        novelty += -np.log2(prob + 1e-10)
    
    return novelty / k


def diversity_at_k(item_embeddings: np.ndarray, 
                   recommendations: List[int], k: int) -> float:
    """
    Compute intra-list diversity at K (average pairwise distance).
    
    Args:
        item_embeddings: Item embedding matrix [n_items, dim]
        recommendations: List of recommended items
        k: Cutoff
        
    Returns:
        Average pairwise cosine distance
    """
    top_k = recommendations[:k]
    
    if len(top_k) < 2:
        return 0.0
    
    embeddings = item_embeddings[top_k]
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    # Compute pairwise cosine similarity
    similarity_matrix = normalized @ normalized.T
    
    # Average distance (1 - similarity) for off-diagonal elements
    n = len(top_k)
    total_distance = n * n - np.sum(similarity_matrix)  # Off-diagonal sum
    
    return total_distance / (n * (n - 1))


class ExplanationMetrics:
    """Metrics for evaluating explanation quality"""
    
    @staticmethod
    def fidelity_plus(original_score: float, masked_score: float) -> float:
        """
        Compute Fidelity+ (drop when top features removed).
        Higher is better (explanations identify important features).
        
        Args:
            original_score: Original prediction score
            masked_score: Score with top Shapley features masked
            
        Returns:
            Fidelity+ value
        """
        return original_score - masked_score
    
    @staticmethod
    def fidelity_minus(original_score: float, masked_score: float) -> float:
        """
        Compute Fidelity- (drop when bottom features removed).
        Lower is better (unimportant features correctly identified).
        
        Args:
            original_score: Original prediction score
            masked_score: Score with bottom Shapley features masked
            
        Returns:
            Fidelity- value
        """
        return original_score - masked_score
    
    @staticmethod
    def sufficiency(original_score: float, 
                    top_k_only_score: float) -> float:
        """
        Compute sufficiency (prediction using only top-k features).
        
        Args:
            original_score: Original prediction score
            top_k_only_score: Score using only top-k Shapley features
            
        Returns:
            Sufficiency ratio
        """
        if original_score == 0:
            return 1.0 if top_k_only_score == 0 else 0.0
        return top_k_only_score / original_score
    
    @staticmethod
    def stability(shapley_1: np.ndarray, shapley_2: np.ndarray) -> float:
        """
        Compute stability (consistency of explanations under perturbation).
        
        Args:
            shapley_1: Shapley values for original input
            shapley_2: Shapley values for perturbed input
            
        Returns:
            Cosine similarity between explanations
        """
        norm1 = np.linalg.norm(shapley_1)
        norm2 = np.linalg.norm(shapley_2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0
        
        return np.dot(shapley_1, shapley_2) / (norm1 * norm2)


def compute_all_metrics(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_list: List[int] = [10, 20, 50],
    item_popularity: Optional[Dict[int, int]] = None,
    n_items: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all recommendation metrics.
    
    Args:
        recommendations: Dict mapping user IDs to recommendation lists
        ground_truth: Dict mapping user IDs to ground truth item sets
        k_list: List of K values for metrics
        item_popularity: Optional item popularity dict
        n_items: Optional total number of items
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    for k in k_list:
        recall_scores = []
        ndcg_scores = []
        hit_scores = []
        precision_scores = []
        
        for user, recs in recommendations.items():
            gt = ground_truth.get(user, set())
            if not gt:
                continue
            
            recall_scores.append(recall_at_k(recs, gt, k))
            ndcg_scores.append(ndcg_at_k(recs, gt, k))
            hit_scores.append(hit_rate_at_k(recs, gt, k))
            precision_scores.append(precision_at_k(recs, gt, k))
        
        metrics[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        metrics[f'hit@{k}'] = np.mean(hit_scores) if hit_scores else 0.0
        metrics[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
    
    # MRR
    mrr_scores = []
    for user, recs in recommendations.items():
        gt = ground_truth.get(user, set())
        if gt:
            mrr_scores.append(mrr(recs, gt))
    metrics['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0
    
    # Coverage
    if n_items:
        all_recs = list(recommendations.values())
        for k in k_list:
            metrics[f'coverage@{k}'] = coverage_at_k(all_recs, n_items, k)
    
    return metrics