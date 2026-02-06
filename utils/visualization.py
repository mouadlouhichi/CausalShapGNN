"""
Visualization utilities for CausalShapGNN
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os


def plot_training_curves(
    metrics_history: List[Dict],
    metrics_to_plot: List[str] = ['total', 'bpr', 'cdm'],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Plot training loss curves.
    
    Args:
        metrics_history: List of metric dictionaries per epoch
        metrics_to_plot: Which metrics to include
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(metrics_history) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics_to_plot:
            values = [m.get(metric, 0) for m in metrics_history]
            ax.plot(epochs, values, label=metric, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def plot_validation_metrics(
    metrics_history: List[Dict],
    metrics_to_plot: List[str] = ['val_recall@20', 'val_ndcg@20'],
    save_path: Optional[str] = None,
    title: str = "Validation Metrics"
):
    """
    Plot validation metric curves.
    
    Args:
        metrics_history: List of metric dictionaries
        metrics_to_plot: Which metrics to include
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(metrics_history) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics_to_plot:
            values = [m.get(metric, 0) for m in metrics_history]
            if any(v > 0 for v in values):
                ax.plot(epochs, values, label=metric, linewidth=2, marker='o', 
                       markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def plot_popularity_distribution(
    recommendation_counts: np.ndarray,
    item_popularity: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Recommendation vs. Original Popularity"
):
    """
    Plot recommendation frequency vs. original item popularity.
    
    Args:
        recommendation_counts: How often each item was recommended
        item_popularity: Original training popularity of items
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of recommendation counts
        ax1 = axes[0]
        ax1.hist(recommendation_counts, bins=50, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Recommendation Count')
        ax1.set_ylabel('Number of Items')
        ax1.set_title('Distribution of Recommendation Counts')
        ax1.set_yscale('log')
        
        # Scatter plot: popularity vs recommendation
        ax2 = axes[1]
        ax2.scatter(item_popularity, recommendation_counts, alpha=0.3, s=10)
        ax2.set_xlabel('Original Popularity (Training)')
        ax2.set_ylabel('Recommendation Count')
        ax2.set_title('Popularity Correlation')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Add correlation coefficient
        valid_mask = (item_popularity > 0) & (recommendation_counts > 0)
        if valid_mask.sum() > 0:
            corr = np.corrcoef(
                np.log(item_popularity[valid_mask] + 1),
                np.log(recommendation_counts[valid_mask] + 1)
            )[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_samples: int = 5000,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    title: str = "Embedding Visualization (t-SNE)"
):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding matrix [n_nodes, dim]
        labels: Optional labels for coloring
        n_samples: Number of samples to visualize
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Sample if too many points
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]
        
        # Run t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if labels is not None:
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], 
                               c=labels, cmap='tab10', alpha=0.5, s=10)
            plt.colorbar(scatter)
        else:
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=10)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("matplotlib or sklearn not available")


def plot_factor_correlations(
    causal_gates: List,
    factor_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Factor Correlation Matrix"
):
    """
    Plot correlation matrix of learned factor gates.
    
    Args:
        causal_gates: List of gating parameters
        factor_names: Names for factors
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import torch
        
        # Extract gate values
        gate_values = []
        for layer_gates in causal_gates:
            layer_vals = []
            for g in layer_gates:
                with torch.no_grad():
                    layer_vals.append(torch.sigmoid(g).mean().item())
            gate_values.append(layer_vals)
        
        gate_matrix = np.array(gate_values)
        n_factors = gate_matrix.shape[1]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(gate_matrix.T)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im)
        
        if factor_names:
            names = factor_names[:n_factors]
        else:
            names = [f'Factor {i}' for i in range(n_factors)]
        
        ax.set_xticks(range(n_factors))
        ax.set_yticks(range(n_factors))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        ax.set_title(title)
        
        # Add correlation values
        for i in range(n_factors):
            for j in range(n_factors):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def plot_explanation_comparison(
    explanations: List[Dict],
    factor_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Explanation Comparison"
):
    """
    Compare explanations across multiple recommendations.
    
    Args:
        explanations: List of explanation dictionaries
        factor_names: Names for factors
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        n_items = len(explanations)
        n_factors = len(factor_names)
        
        shapley_matrix = np.zeros((n_items, n_factors))
        
        for i, exp in enumerate(explanations):
            shapley_matrix[i] = exp.get('feature_shapley', np.zeros(n_factors))
        
        fig, ax = plt.subplots(figsize=(12, max(4, n_items * 0.5)))
        
        im = ax.imshow(shapley_matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, label='Shapley Value')
        
        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_names, rotation=45, ha='right')
        ax.set_yticks(range(n_items))
        ax.set_yticklabels([f"Item {exp.get('item_idx', i)}" 
                           for i, exp in enumerate(explanations)])
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def create_summary_report(
    experiment_dir: str,
    metrics: Dict[str, float],
    config: Dict,
    training_time: float
):
    """
    Create a summary report for an experiment.
    
    Args:
        experiment_dir: Directory containing experiment files
        metrics: Final test metrics
        config: Configuration dictionary
        training_time: Total training time in seconds
    """
    report_path = os.path.join(experiment_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CausalShapGNN EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("FINAL METRICS\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(metrics.items()):
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("TRAINING INFO\n")
        f.write("-" * 40 + "\n")
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        seconds = training_time % 60
        f.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"Summary report saved to {report_path}")