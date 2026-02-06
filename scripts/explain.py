#!/usr/bin/env python
"""
Generate explanations for recommendations using CausalShapGNN
"""

import argparse
import os
import sys
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, get_default_config
from data import DataPreprocessor, BipartiteGraphProcessor
from models import CausalShapGNN
from explainers import ShapleyExplainer, FeatureShapley, PathShapley, UserProfileShapley
from explainers import ExplanationReport, ExplanationVisualizer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Generate explanations')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--user_id', type=int, required=True, help='User ID to explain')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--output', type=str, default='./explanations', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    # Load config and data
    config = get_default_config()
    config.update(load_config(args.config))
    
    preprocessor = DataPreprocessor(config['dataset']['path'], config['dataset']['name'])
    graph_data = preprocessor.load_data()
    
    config['n_users'] = graph_data.n_users
    config['n_items'] = graph_data.n_items
    
    graph_processor = BipartiteGraphProcessor(
        graph_data.n_users, graph_data.n_items,
        graph_data.train_interactions, device
    )
    
    # Load model
    model = CausalShapGNN(config, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print(f"Generating explanations for User {args.user_id}")
    
    # Get recommendations
    with torch.no_grad():
        user_emb, item_emb, _ = model(graph_processor.norm_adj, use_causal_only=True)
    
    user_e = user_emb[args.user_id]
    scores = torch.matmul(user_e, item_emb.t())
    
    # Mask training items
    train_items = list(graph_processor.train_user_items[args.user_id])
    if train_items:
        scores[train_items] = -float('inf')
    
    top_scores, top_items = torch.topk(scores, args.top_k)
    top_items = top_items.cpu().numpy().tolist()
    top_scores = top_scores.cpu().numpy().tolist()
    
    print(f"\nTop {args.top_k} Recommendations:")
    for i, (item, score) in enumerate(zip(top_items, top_scores)):
        print(f"  {i+1}. Item {item} (score: {score:.4f})")
    
    # Initialize explainers
    feature_explainer = FeatureShapley(model, device)
    feature_explainer._compute_population_means(user_emb, item_emb)
    
    path_explainer = PathShapley(model, device)
    profile_explainer = UserProfileShapley(feature_explainer)
    
    report_generator = ExplanationReport(model, device)
    visualizer = ExplanationVisualizer()
    
    # Generate explanations
    print("\nGenerating explanations...")
    
    explanations = []
    
    for item_idx, score in zip(top_items, top_scores):
        # Feature-level Shapley
        feature_shapley = feature_explainer.compute(
            args.user_id, item_idx, user_emb, item_emb
        )
        
        # Path-level Shapley
        path_shapley = path_explainer.compute(
            args.user_id, item_idx, graph_processor.norm_adj,
            user_emb, item_emb
        )
        
        # Generate explanation
        explanation = report_generator.generate_recommendation_explanation(
            args.user_id, item_idx, feature_shapley, path_shapley, score
        )
        explanations.append(explanation)
        
        print(f"\nItem {item_idx}:")
        print(explanation.explanation_text)
    
    # Generate user profile
    user_profile = profile_explainer.compute(
        args.user_id, top_items, user_emb, item_emb
    )
    
    profile_report = report_generator.generate_user_profile_report(
        args.user_id, user_profile, top_items
    )
    print("\n" + profile_report)
    
    # Save explanations
    for exp in explanations:
        json_str = report_generator.to_json(exp)
        filepath = os.path.join(args.output, f'user{args.user_id}_item{exp.item_idx}.json')
        with open(filepath, 'w') as f:
            f.write(json_str)
    
    # Generate visualizations
    visualizer.plot_factor_importance(
        user_profile,
        title=f"User {args.user_id} Preference Profile",
        save_path=os.path.join(args.output, f'user{args.user_id}_profile.png')
    )
    
    print(f"\nExplanations saved to {args.output}")


if __name__ == "__main__":
    main()