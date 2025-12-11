"""
Visualization of Next Item Prediction Model Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Project root
project_root = Path(__file__).parent.parent.parent


def load_results(store_id=1):
    """Load comparison results"""
    results_path = project_root / 'results' / 'next_item' / f'store_{store_id}_comparison.csv'
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Please run compare_next_item_models.py first")
        sys.exit(1)
    
    results = pd.read_csv(results_path)
    return results


def create_metrics_comparison(results, store_id):
    """Create bar plot comparing main metrics"""
    output_dir = project_root / 'results' / 'next_item' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by NDCG@10
    results = results.sort_values('ndcg@10', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    metrics = ['hit_rate@10', 'mrr@10', 'ndcg@10', 'hit_rate@5']
    titles = ['Hit Rate@10', 'MRR@10', 'NDCG@10', 'Hit Rate@5']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        bars = ax.barh(results['model'], results[metric])
        
        # Color the best model
        best_idx = results[metric].idxmax()
        colors = ['#2ecc71' if i == best_idx else '#3498db' for i in results.index]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(title)
        ax.set_title(f'{title} Comparison')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(results.iterrows()):
            ax.text(row[metric], i, f' {row[metric]:.4f}', va='center', fontsize=9)
    
    plt.suptitle(f'Next Item Prediction Models - Store {store_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'store_{store_id}_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_performance_overview(results, store_id):
    """Create comprehensive performance overview"""
    output_dir = project_root / 'results' / 'next_item' / 'visualizations'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sort by NDCG@10
    results = results.sort_values('ndcg@10', ascending=False)
    
    # 1. Hit Rate comparison across different k values
    ax = axes[0]
    k_values = [5, 10, 20]
    hit_rate_cols = [f'hit_rate@{k}' for k in k_values if f'hit_rate@{k}' in results.columns]
    
    x = np.arange(len(results))
    width = 0.25
    
    for i, col in enumerate(hit_rate_cols):
        offset = (i - len(hit_rate_cols)/2 + 0.5) * width
        ax.bar(x + offset, results[col], width, label=col.replace('hit_rate@', 'k='))
    
    ax.set_ylabel('Hit Rate')
    ax.set_title('Hit Rate at Different k Values')
    ax.set_xticks(x)
    ax.set_xticklabels(results['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. NDCG comparison
    ax = axes[1]
    ndcg_cols = [f'ndcg@{k}' for k in k_values if f'ndcg@{k}' in results.columns]
    
    for i, col in enumerate(ndcg_cols):
        offset = (i - len(ndcg_cols)/2 + 0.5) * width
        ax.bar(x + offset, results[col], width, label=col.replace('ndcg@', 'k='))
    
    ax.set_ylabel('NDCG')
    ax.set_title('NDCG at Different k Values')
    ax.set_xticks(x)
    ax.set_xticklabels(results['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Training time comparison
    ax = axes[2]
    bars = ax.barh(results['model'], results['train_time'])
    
    # Color by time (faster = green, slower = red)
    norm = plt.Normalize(vmin=results['train_time'].min(), vmax=results['train_time'].max())
    colors = plt.cm.RdYlGn_r(norm(results['train_time']))
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results.iterrows()):
        ax.text(row['train_time'], i, f' {row["train_time"]:.2f}s', va='center', fontsize=9)
    
    plt.suptitle(f'Performance Overview - Store {store_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'store_{store_id}_performance_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_accuracy_vs_speed(results, store_id):
    """Create accuracy vs speed tradeoff plot"""
    output_dir = project_root / 'results' / 'next_item' / 'visualizations'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot: NDCG@10 vs Training Time
    scatter = ax.scatter(
        results['train_time'],
        results['ndcg@10'],
        s=results['hit_rate@10'] * 500,  # Size by hit rate
        alpha=0.6,
        c=range(len(results)),
        cmap='viridis'
    )
    
    # Add labels
    for idx, row in results.iterrows():
        ax.annotate(
            row['model'],
            (row['train_time'], row['ndcg@10']),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('NDCG@10', fontsize=12)
    ax.set_title(f'Accuracy vs Training Speed\n(bubble size = Hit Rate@10) - Store {store_id}', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add quadrant lines
    median_time = results['train_time'].median()
    median_ndcg = results['ndcg@10'].median()
    
    ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(median_ndcg, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Annotate quadrants
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Fast & Accurate', 
            fontsize=10, alpha=0.5, ha='left', va='top')
    ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'Slow but Accurate', 
            fontsize=10, alpha=0.5, ha='right', va='top')
    ax.text(ax.get_xlim()[0], ax.get_ylim()[0], 'Fast but Less Accurate', 
            fontsize=10, alpha=0.5, ha='left', va='bottom')
    
    output_path = output_dir / f'store_{store_id}_accuracy_vs_speed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(results, store_id):
    """Create formatted summary table"""
    output_dir = project_root / 'results' / 'next_item' / 'visualizations'
    
    # Select key metrics
    summary = results[['model', 'hit_rate@10', 'mrr@10', 'ndcg@10', 'train_time']].copy()
    
    # Sort by NDCG@10
    summary = summary.sort_values('ndcg@10', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(summary) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for table
    table_data = []
    headers = ['Rank', 'Model', 'Hit Rate@10', 'MRR@10', 'NDCG@10', 'Train Time']
    
    for rank, (idx, row) in enumerate(summary.iterrows(), 1):
        table_data.append([
            rank,
            row['model'],
            f"{row['hit_rate@10']:.4f}",
            f"{row['mrr@10']:.4f}",
            f"{row['ndcg@10']:.4f}",
            f"{row['train_time']:.2f}s"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best model
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#2ecc71')
        table[(1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title(f'Next Item Prediction Models - Summary Table (Store {store_id})', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = output_dir / f'store_{store_id}_summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations"""
    store_id = 1
    
    print(f"\n{'='*70}")
    print(f"NEXT ITEM PREDICTION MODEL COMPARISON - VISUALIZATION")
    print(f"Store ID: {store_id}")
    print(f"{'='*70}\n")
    
    # Load results
    results = load_results(store_id)
    
    print(f"Loaded results for {len(results)} models\n")
    
    # Create visualizations
    print("Creating visualizations...")
    
    create_metrics_comparison(results, store_id)
    create_performance_overview(results, store_id)
    create_accuracy_vs_speed(results, store_id)
    create_summary_table(results, store_id)
    
    print(f"\n{'='*70}")
    print("All visualizations created successfully!")
    print(f"Location: {project_root / 'results' / 'next_item' / 'visualizations'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
