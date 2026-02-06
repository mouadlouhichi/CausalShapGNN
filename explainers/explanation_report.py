"""
Explanation Report Generation for CausalShapGNN
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class FactorExplanation:
    """Explanation for a single factor"""
    factor_idx: int
    factor_name: str
    shapley_value: float
    contribution_pct: float
    direction: str  # 'positive' or 'negative'


@dataclass
class PathExplanation:
    """Explanation for a single path"""
    path: Tuple[int, ...]
    shapley_value: float
    path_description: str


@dataclass
class RecommendationExplanation:
    """Complete explanation for a recommendation"""
    user_idx: int
    item_idx: int
    predicted_score: float
    factor_explanations: List[FactorExplanation]
    path_explanations: List[PathExplanation]
    top_factor: str
    explanation_text: str


class ExplanationReport:
    """
    Generates human-readable explanation reports.
    """
    
    DEFAULT_FACTOR_NAMES = [
        "Genre/Category Preference",
        "Price Sensitivity",
        "Social Influence",
        "Recency Preference",
        "Quality Focus",
        "Novelty Seeking",
        "Brand/Source Loyalty",
        "Seasonal/Temporal Trend"
    ]
    
    def __init__(self, model, device: torch.device,
                 factor_names: Optional[List[str]] = None):
        """
        Initialize ExplanationReport generator.
        
        Args:
            model: CausalShapGNN model
            device: Torch device
            factor_names: Custom names for latent factors
        """
        self.model = model
        self.device = device
        self.n_factors = model.cdm.n_factors
        
        if factor_names is None:
            self.factor_names = self.DEFAULT_FACTOR_NAMES[:self.n_factors]
        else:
            self.factor_names = factor_names
    
    def generate_recommendation_explanation(
        self,
        user_idx: int,
        item_idx: int,
        feature_shapley: np.ndarray,
        path_shapley: Dict[Tuple, float],
        predicted_score: float
    ) -> RecommendationExplanation:
        """
        Generate explanation for a single recommendation.
        
        Args:
            user_idx: User index
            item_idx: Item index
            feature_shapley: Feature-level Shapley values
            path_shapley: Path-level Shapley values
            predicted_score: Model prediction score
            
        Returns:
            RecommendationExplanation object
        """
        # Process factor explanations
        total_abs = np.abs(feature_shapley).sum()
        factor_explanations = []
        
        for idx in range(self.n_factors):
            value = feature_shapley[idx]
            pct = (value / total_abs * 100) if total_abs > 0 else 0
            
            factor_explanations.append(FactorExplanation(
                factor_idx=idx,
                factor_name=self.factor_names[idx],
                shapley_value=value,
                contribution_pct=abs(pct),
                direction='positive' if value >= 0 else 'negative'
            ))
        
        # Sort by absolute contribution
        factor_explanations.sort(key=lambda x: abs(x.shapley_value), reverse=True)
        
        # Process path explanations
        path_explanations = []
        for path, value in sorted(path_shapley.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:5]:
            path_explanations.append(PathExplanation(
                path=path,
                shapley_value=value,
                path_description=self._describe_path(path)
            ))
        
        # Generate text explanation
        top_factor = factor_explanations[0] if factor_explanations else None
        explanation_text = self._generate_text_explanation(
            user_idx, item_idx, factor_explanations, path_explanations
        )
        
        return RecommendationExplanation(
            user_idx=user_idx,
            item_idx=item_idx,
            predicted_score=predicted_score,
            factor_explanations=factor_explanations,
            path_explanations=path_explanations,
            top_factor=top_factor.factor_name if top_factor else "Unknown",
            explanation_text=explanation_text
        )
    
    def _describe_path(self, path: Tuple[int, ...]) -> str:
        """Generate human-readable path description"""
        n_users = self.model.cdm.n_users
        
        descriptions = []
        for node in path:
            if node < n_users:
                descriptions.append(f"User_{node}")
            else:
                descriptions.append(f"Item_{node - n_users}")
        
        return " → ".join(descriptions)
    
    def _generate_text_explanation(
        self,
        user_idx: int,
        item_idx: int,
        factor_explanations: List[FactorExplanation],
        path_explanations: List[PathExplanation]
    ) -> str:
        """Generate natural language explanation"""
        
        if not factor_explanations:
            return "Unable to generate explanation."
        
        top_factors = factor_explanations[:3]
        
        text = f"Item {item_idx} is recommended to User {user_idx} primarily because of:\n"
        
        for i, factor in enumerate(top_factors, 1):
            direction = "increases" if factor.direction == 'positive' else "decreases"
            text += f"  {i}. {factor.factor_name} ({factor.contribution_pct:.1f}%): "
            text += f"This factor {direction} the recommendation score.\n"
        
        if path_explanations:
            text += "\nKey interaction paths:\n"
            for i, path_exp in enumerate(path_explanations[:3], 1):
                text += f"  {i}. {path_exp.path_description}\n"
        
        return text
    
    def generate_user_profile_report(
        self,
        user_idx: int,
        user_profile: np.ndarray,
        recommended_items: List[int]
    ) -> str:
        """
        Generate user preference profile report.
        
        Args:
            user_idx: User index
            user_profile: User profile Shapley values
            recommended_items: List of recommended items
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"USER PREFERENCE PROFILE: User {user_idx}")
        lines.append("=" * 60)
        lines.append("")
        
        # Normalize profile
        total = np.abs(user_profile).sum()
        
        lines.append("Factor Importance Breakdown:")
        lines.append("-" * 40)
        
        # Sort by importance
        sorted_indices = np.argsort(np.abs(user_profile))[::-1]
        
        for idx in sorted_indices:
            value = user_profile[idx]
            pct = (value / total * 100) if total > 0 else 0
            bar_length = int(abs(pct) / 5)
            bar = "█" * bar_length
            sign = "+" if value >= 0 else "-"
            
            lines.append(
                f"  {self.factor_names[idx]:<25} {sign}{abs(pct):>5.1f}% {bar}"
            )
        
        lines.append("")
        lines.append("-" * 40)
        lines.append(f"Total recommendations analyzed: {len(recommended_items)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_comparison_report(
        self,
        explanations: List[RecommendationExplanation]
    ) -> str:
        """
        Generate comparison report for multiple recommendations.
        
        Args:
            explanations: List of recommendation explanations
            
        Returns:
            Formatted comparison report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RECOMMENDATION COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Header
        lines.append(f"{'Item':<10} {'Score':<10} {'Top Factor':<25} {'Contribution':<15}")
        lines.append("-" * 70)
        
        for exp in explanations:
            top = exp.factor_explanations[0] if exp.factor_explanations else None
            if top:
                lines.append(
                    f"{exp.item_idx:<10} {exp.predicted_score:<10.4f} "
                    f"{top.factor_name:<25} {top.contribution_pct:<15.1f}%"
                )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_json(self, explanation: RecommendationExplanation) -> str:
        """
        Convert explanation to JSON format.
        
        Args:
            explanation: RecommendationExplanation object
            
        Returns:
            JSON string
        """
        data = {
            'user_idx': explanation.user_idx,
            'item_idx': explanation.item_idx,
            'predicted_score': explanation.predicted_score,
            'top_factor': explanation.top_factor,
            'explanation_text': explanation.explanation_text,
            'factors': [
                {
                    'name': f.factor_name,
                    'shapley_value': float(f.shapley_value),
                    'contribution_pct': float(f.contribution_pct),
                    'direction': f.direction
                }
                for f in explanation.factor_explanations
            ],
            'paths': [
                {
                    'path': list(p.path),
                    'shapley_value': float(p.shapley_value),
                    'description': p.path_description
                }
                for p in explanation.path_explanations
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        return json.dumps(data, indent=2)


class ExplanationVisualizer:
    """
    Visualizes explanations using matplotlib.
    """
    
    def __init__(self, factor_names: Optional[List[str]] = None):
        """
        Initialize visualizer.
        
        Args:
            factor_names: Names for latent factors
        """
        self.factor_names = factor_names or ExplanationReport.DEFAULT_FACTOR_NAMES
    
    def plot_factor_importance(
        self,
        shapley_values: np.ndarray,
        title: str = "Factor Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot factor importance bar chart.
        
        Args:
            shapley_values: Shapley values for factors
            title: Plot title
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            n_factors = len(shapley_values)
            names = self.factor_names[:n_factors]
            
            colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in shapley_values]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_pos = np.arange(n_factors)
            ax.barh(y_pos, shapley_values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Shapley Value')
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            print("matplotlib not available for visualization")
    
    def plot_user_profile_heatmap(
        self,
        user_profiles: Dict[int, np.ndarray],
        title: str = "User Preference Profiles",
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of multiple user profiles.
        
        Args:
            user_profiles: Dictionary mapping user IDs to profiles
            title: Plot title
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            user_ids = list(user_profiles.keys())
            profiles = np.array([user_profiles[u] for u in user_ids])
            
            n_factors = profiles.shape[1]
            names = self.factor_names[:n_factors]
            
            fig, ax = plt.subplots(figsize=(12, max(6, len(user_ids) * 0.5)))
            
            sns.heatmap(
                profiles,
                xticklabels=names,
                yticklabels=[f"User {u}" for u in user_ids],
                cmap='RdBu_r',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Factor Importance'}
            )
            
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            print("matplotlib/seaborn not available for visualization")
    
    def plot_path_importance(
        self,
        path_shapley: Dict[Tuple, float],
        n_users: int,
        title: str = "Path Importance",
        save_path: Optional[str] = None,
        top_k: int = 10
    ):
        """
        Plot path importance.
        
        Args:
            path_shapley: Path Shapley values
            n_users: Number of users (for path labeling)
            title: Plot title
            save_path: Path to save figure
            top_k: Number of top paths to show
        """
        try:
            import matplotlib.pyplot as plt
            
            # Sort and get top paths
            sorted_paths = sorted(path_shapley.items(), 
                                 key=lambda x: abs(x[1]), reverse=True)[:top_k]
            
            if not sorted_paths:
                print("No paths to visualize")
                return
            
            paths, values = zip(*sorted_paths)
            
            # Create path labels
            labels = []
            for path in paths:
                parts = []
                for node in path:
                    if node < n_users:
                        parts.append(f"U{node}")
                    else:
                        parts.append(f"I{node - n_users}")
                labels.append("→".join(parts))
            
            colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
            
            fig, ax = plt.subplots(figsize=(10, max(4, len(paths) * 0.4)))
            
            y_pos = np.arange(len(paths))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Shapley Value')
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            print("matplotlib not available for visualization")