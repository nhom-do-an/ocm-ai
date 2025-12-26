"""
Visualization and Comparison of Recommendation Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Project root
project_root = Path(__file__).parent.parent.parent


def load_results():
    """Load all model results - only 3 models: Popularity (baseline), ItemKNN, NeuMF"""
    results_path = project_root / 'results' / 'recommendation'
    
    # Load sklearn results
    sklearn_results = pd.read_csv(results_path / 'all_stores_sklearn_results.csv')
    
    # Filter to only 3 models: Popularity (baseline), ItemKNN, NeuMF
    # Map model names to standard names
    model_mapping = {
        'Popularity': 'Popularity',
        'ItemKNN_k20': 'ItemKNN',
        'ItemKNN': 'ItemKNN',
        'NeuMF': 'NeuMF'
    }
    
    # Filter and rename
    filtered_results = sklearn_results[sklearn_results['model_name'].isin(model_mapping.keys())].copy()
    filtered_results['model_name'] = filtered_results['model_name'].map(model_mapping)
    
    # Load NeuMF results if exists
    neumf_file = results_path / 'neumf_results.csv'
    if neumf_file.exists():
        neumf_results = pd.read_csv(neumf_file)
        neumf_results['model_name'] = 'NeuMF'
        filtered_results = pd.concat([filtered_results, neumf_results], ignore_index=True)
    
    # Add model type
    filtered_results['model_type'] = filtered_results['model_name'].map({
        'Popularity': 'Baseline',
        'ItemKNN': 'Collaborative Filtering',
        'NeuMF': 'Deep Learning'
    })
    
    return filtered_results


def create_comparison_plots(results):
    """Create comparison visualizations"""
    output_dir = project_root / 'results' / 'recommendation' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate average metrics across stores
    avg_results = results.groupby('model_name').agg({
        'precision@10': 'mean',
        'recall@10': 'mean',
        'ndcg@10': 'mean',
        'hit_rate@10': 'mean',
        'train_time': 'mean',
        'eval_time': 'mean',
        'model_type': 'first'  # Keep model_type (same for all rows of same model)
    }).reset_index()
    
    # Sort by NDCG
    avg_results = avg_results.sort_values('ndcg@10', ascending=False)
    
    # 1. Bar plot of main metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10']
    titles = ['Precision@10', 'Recall@10', 'NDCG@10', 'Hit Rate@10']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Color by model type
        color_map = {
            'Baseline': '#3498db',
            'Collaborative Filtering': '#2ecc71',
            'Deep Learning': '#e74c3c'
        }
        colors = [color_map.get(avg_results.loc[i, 'model_type'], 'gray') 
                 for i in avg_results.index]
        
        bars = ax.barh(avg_results['model_name'], avg_results[metric], color=colors)
        
        # Highlight the best model with gold border
        best_idx = avg_results[metric].idxmax()
        best_bar_idx = avg_results.index.get_loc(best_idx)
        bars[best_bar_idx].set_edgecolor('gold')
        bars[best_bar_idx].set_linewidth(3)
        bars[best_bar_idx].set_alpha(0.9)
        
        ax.set_xlabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison (🥇 Best: {avg_results.loc[best_idx, "model_name"]})', 
                    fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_results[metric]):
            label = f' {v:.4f}'
            if i == best_bar_idx:
                label = f' 🥇{v:.4f}'
            ax.text(v, i, label, va='center', fontweight='bold' if i == best_bar_idx else 'normal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")
    plt.close()
    
    # 2. Precision-Recall tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        avg_results['recall@10'], 
        avg_results['precision@10'],
        s=avg_results['ndcg@10'] * 1000,  # Size by NDCG
        alpha=0.6,
        c=range(len(avg_results)),
        cmap='viridis'
    )
    
    # Add labels
    for idx, row in avg_results.iterrows():
        ax.annotate(
            row['model_name'], 
            (row['recall@10'], row['precision@10']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('Precision@10')
    ax.set_title('Precision-Recall Tradeoff (bubble size = NDCG@10)')
    ax.grid(alpha=0.3)
    
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'precision_recall_tradeoff.png'}")
    plt.close()
    
    # 3. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_by_time = avg_results.sort_values('train_time')
    bars = ax.barh(sorted_by_time['model_name'], sorted_by_time['train_time'])
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Model Training Time Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sorted_by_time['train_time']):
        ax.text(v, i, f' {v:.3f}s', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_time_comparison.png'}")
    plt.close()
    
    # 4. Performance across stores (heatmap)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot for heatmap
    heatmap_data = results.pivot_table(
        values='ndcg@10',
        index='model_name',
        columns='store_id'
    )
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'NDCG@10'}
    )
    
    ax.set_title('NDCG@10 Performance Across Stores')
    ax.set_xlabel('Store ID')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_across_stores.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_across_stores.png'}")
    plt.close()
    
    # 5. Radar chart for best models
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Take top 5 models by NDCG
    top_models = avg_results.nlargest(5, 'ndcg@10')
    
    categories = ['Precision@10', 'Recall@10', 'NDCG@10', 'Hit Rate@10']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, row in top_models.iterrows():
        values = [
            row['precision@10'],
            row['recall@10'],
            row['ndcg@10'],
            row['hit_rate@10']
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(top_models[['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10']].max()) * 1.1)
    ax.set_title('Top 5 Models - Performance Radar Chart', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_models_radar.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'top_models_radar.png'}")
    plt.close()


def create_summary_report(results):
    """Create a markdown summary report"""
    output_dir = project_root / 'results' / 'recommendation'
    
    # Calculate statistics
    avg_results = results.groupby('model_name').agg({
        'precision@10': ['mean', 'std'],
        'recall@10': ['mean', 'std'],
        'ndcg@10': ['mean', 'std'],
        'hit_rate@10': ['mean', 'std'],
        'train_time': ['mean', 'std'],
        'eval_time': ['mean', 'std'],
        'n_evaluated_users': 'mean'
    }).round(4)
    
    # Find best models
    best_precision = avg_results[('precision@10', 'mean')].idxmax()
    best_recall = avg_results[('recall@10', 'mean')].idxmax()
    best_ndcg = avg_results[('ndcg@10', 'mean')].idxmax()
    best_hitrate = avg_results[('hit_rate@10', 'mean')].idxmax()
    
    # Create report
    report = f"""# Recommendation Models Comparison Report

## Overview

This report compares {len(avg_results)} different recommendation models trained on {len(results['store_id'].unique())} stores.

**Models Evaluated:**
{chr(10).join([f'- {model}' for model in avg_results.index])}

**Evaluation Metrics:**
- Precision@10: Proportion of recommended items that are relevant
- Recall@10: Proportion of relevant items that are recommended
- NDCG@10: Normalized Discounted Cumulative Gain (ranking quality)
- Hit Rate@10: Proportion of users with at least one relevant item in top-10

## Best Performing Models

| Metric | Best Model | Score |
|--------|-----------|-------|
| Precision@10 | **{best_precision}** | {avg_results.loc[best_precision, ('precision@10', 'mean')]:.4f} ± {avg_results.loc[best_precision, ('precision@10', 'std')]:.4f} |
| Recall@10 | **{best_recall}** | {avg_results.loc[best_recall, ('recall@10', 'mean')]:.4f} ± {avg_results.loc[best_recall, ('recall@10', 'std')]:.4f} |
| NDCG@10 | **{best_ndcg}** | {avg_results.loc[best_ndcg, ('ndcg@10', 'mean')]:.4f} ± {avg_results.loc[best_ndcg, ('ndcg@10', 'std')]:.4f} |
| Hit Rate@10 | **{best_hitrate}** | {avg_results.loc[best_hitrate, ('hit_rate@10', 'mean')]:.4f} ± {avg_results.loc[best_hitrate, ('hit_rate@10', 'std')]:.4f} |

## Detailed Results

### Average Performance Across All Stores

"""
    
    # Add detailed table
    report += "| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Train Time (s) |\n"
    report += "|-------|--------------|-----------|---------|-------------|----------------|\n"
    
    for model in avg_results.index:
        report += f"| {model} | "
        report += f"{avg_results.loc[model, ('precision@10', 'mean')]:.4f} ± {avg_results.loc[model, ('precision@10', 'std')]:.4f} | "
        report += f"{avg_results.loc[model, ('recall@10', 'mean')]:.4f} ± {avg_results.loc[model, ('recall@10', 'std')]:.4f} | "
        report += f"{avg_results.loc[model, ('ndcg@10', 'mean')]:.4f} ± {avg_results.loc[model, ('ndcg@10', 'std')]:.4f} | "
        report += f"{avg_results.loc[model, ('hit_rate@10', 'mean')]:.4f} ± {avg_results.loc[model, ('hit_rate@10', 'std')]:.4f} | "
        report += f"{avg_results.loc[model, ('train_time', 'mean')]:.3f} ± {avg_results.loc[model, ('train_time', 'std')]:.3f} |\n"
    
    report += f"""

## Key Insights

1. **{best_ndcg}** achieves the highest NDCG@10 ({avg_results.loc[best_ndcg, ('ndcg@10', 'mean')]:.4f}), indicating the best ranking quality.

2. **{best_precision}** has the highest precision ({avg_results.loc[best_precision, ('precision@10', 'mean')]:.4f}), meaning its recommendations are most accurate.

3. **{best_recall}** achieves the highest recall ({avg_results.loc[best_recall, ('recall@10', 'mean')]:.4f}), capturing more relevant items.

4. Training times range from {avg_results[('train_time', 'mean')].min():.3f}s to {avg_results[('train_time', 'mean')].max():.3f}s.

## Recommendations

Based on the evaluation:

- **For Production**: Use **{best_ndcg}** for the best overall ranking quality (NDCG@10: {avg_results.loc[best_ndcg, ('ndcg@10', 'mean')]:.4f})
- **For Fast Training**: Use **Popularity** baseline (fastest, simple, but decent performance)
- **For Personalization**: Matrix factorization methods (NMF, SVD) provide better personalization

## Visualizations

See the `visualizations/` folder for:
- Metrics comparison bar charts
- Precision-Recall tradeoff scatter plot
- Training time comparison
- Performance across stores heatmap
- Top models radar chart

## Next Steps

1. Implement deep learning models (Neural Collaborative Filtering, Graph Neural Networks)
2. Optimize hyperparameters for best models
3. Implement ensemble methods
4. A/B testing in production environment
"""
    
    # Save report
    report_file = output_dir / 'COMPARISON_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSaved: {report_file}")
    return report


def main():
    print("=" * 60)
    print("Recommendation Models Comparison")
    print("=" * 60)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    
    print(f"Loaded results for {len(results)} model-store combinations")
    print(f"Models: {results['model_name'].unique().tolist()}")
    print(f"Stores: {results['store_id'].unique().tolist()}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_plots(results)
    
    # Create summary report
    print("\nGenerating summary report...")
    report = create_summary_report(results)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
    
    # Print top 3 models
    avg_results = results.groupby('model_name')['ndcg@10'].mean().sort_values(ascending=False)
    print("\nTop 3 Models by NDCG@10:")
    for i, (model, score) in enumerate(avg_results.head(3).items(), 1):
        print(f"{i}. {model}: {score:.4f}")


if __name__ == '__main__':
    main()
