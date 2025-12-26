"""
Create comprehensive comparison visualizations for all three tasks:
1. Recommendation System
2. Trending Prediction
3. Next Item Prediction

This script generates a summary visualization comparing the best models from each task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Project root
project_root = Path(__file__).parent.parent.parent


def load_best_models():
    """Load best model from each task"""
    
    best_models = []
    
    # 1. Recommendation System
    rec_file = project_root / 'results' / 'recommendation' / 'all_stores_sklearn_results.csv'
    if rec_file.exists():
        rec_df = pd.read_csv(rec_file)
        # Filter to only 3 models
        rec_df = rec_df[rec_df['model_name'].isin(['Popularity', 'ItemKNN_k20', 'NeuMF'])]
        rec_df['model_name'] = rec_df['model_name'].replace({'ItemKNN_k20': 'ItemKNN'})
        
        # Get best by NDCG@10
        rec_avg = rec_df.groupby('model_name')['ndcg@10'].mean().reset_index()
        rec_best = rec_avg.loc[rec_avg['ndcg@10'].idxmax()]
        best_models.append({
            'task': 'Recommendation',
            'model': rec_best['model_name'],
            'metric': 'NDCG@10',
            'value': rec_best['ndcg@10'],
            'type': 'Baseline' if rec_best['model_name'] == 'Popularity' 
                   else 'Collaborative Filtering' if rec_best['model_name'] == 'ItemKNN'
                   else 'Deep Learning'
        })
    
    # 2. Trending Prediction
    trending_file = project_root / 'results' / 'trending' / 'lightgbm_results.csv'
    if trending_file.exists():
        trending_df = pd.read_csv(trending_file)
        trending_df['model_name'] = 'LightGBM'
        trending_avg = trending_df.groupby('model_name')['mae'].mean().reset_index()
        trending_best = trending_avg.loc[trending_avg['mae'].idxmin()]
        best_models.append({
            'task': 'Trending',
            'model': trending_best['model_name'],
            'metric': 'MAE',
            'value': trending_best['mae'],
            'type': 'Gradient Boosting'
        })
    
    # 3. Next Item Prediction
    next_item_file = project_root / 'results' / 'next_item' / 'store_1_comparison.csv'
    if next_item_file.exists():
        next_item_df = pd.read_csv(next_item_file)
        # Filter to only 3 models
        next_item_df = next_item_df[next_item_df['model'].isin(['Markov Chain', 'ItemKNN', 'GRU4Rec'])]
        next_item_best = next_item_df.loc[next_item_df['hit_rate@10'].idxmax()]
        best_models.append({
            'task': 'Next Item',
            'model': next_item_best['model'],
            'metric': 'Hit Rate@10',
            'value': next_item_best['hit_rate@10'],
            'type': 'Baseline' if next_item_best['model'] == 'Markov Chain'
                   else 'Collaborative Filtering' if next_item_best['model'] == 'ItemKNN'
                   else 'Deep Learning'
        })
    
    return pd.DataFrame(best_models)


def create_summary_comparison():
    """Create summary visualization comparing best models from all tasks"""
    
    output_dir = project_root / 'results' / 'comparison_summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_models_df = load_best_models()
    
    if best_models_df.empty:
        print("⚠ No results found! Please run model comparisons first.")
        return
    
    print("\n" + "="*80)
    print("CREATING SUMMARY COMPARISON OF ALL TASKS")
    print("="*80)
    print(f"\nBest Models Found:")
    print(best_models_df.to_string(index=False))
    
    # 1. Summary table visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in best_models_df.iterrows():
        table_data.append([
            row['task'],
            row['model'],
            row['type'],
            row['metric'],
            f"{row['value']:.4f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Task', 'Best Model', 'Type', 'Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.15, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best models
    color_map = {
        'Baseline': '#3498db',
        'Collaborative Filtering': '#2ecc71',
        'Deep Learning': '#e74c3c',
        'Gradient Boosting': '#9b59b6'
    }
    
    for i, (_, row) in enumerate(best_models_df.iterrows(), 1):
        for j in range(5):
            table[(i, j)].set_facecolor(color_map.get(row['type'], 'white'))
            if j == 1:  # Model name column
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Best Models Summary - All Tasks', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'best_models_summary.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'best_models_summary.png'}")
    plt.close()
    
    # 2. Performance comparison (normalized)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize values for comparison (higher is better for all)
    normalized_values = []
    for _, row in best_models_df.iterrows():
        if row['metric'] == 'MAE':
            # For MAE, lower is better, so invert
            normalized_values.append(1 / (row['value'] + 0.001))  # Add small value to avoid division by zero
        else:
            # For NDCG and Hit Rate, higher is better
            normalized_values.append(row['value'])
    
    best_models_df['normalized_value'] = normalized_values
    best_models_df = best_models_df.sort_values('normalized_value', ascending=False)
    
    # Color by model type
    colors = [color_map.get(t, 'gray') for t in best_models_df['type']]
    
    bars = ax.barh(best_models_df['task'], best_models_df['normalized_value'], color=colors)
    
    # Highlight best overall
    best_overall_idx = 0
    bars[best_overall_idx].set_edgecolor('gold')
    bars[best_overall_idx].set_linewidth(3)
    
    ax.set_xlabel('Normalized Performance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Best Models Performance Comparison (Normalized)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(best_models_df.iterrows()):
        label = f" {row['model']}\n ({row['metric']}: {row['value']:.4f})"
        if i == best_overall_idx:
            label = f" 🥇{label}"
        ax.text(best_models_df['normalized_value'].iloc[i], i, label, 
               va='center', fontweight='bold' if i == best_overall_idx else 'normal')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#2ecc71', label='Collaborative Filtering'),
        Patch(facecolor='#e74c3c', label='Deep Learning'),
        Patch(facecolor='#9b59b6', label='Gradient Boosting')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_comparison.png'}")
    plt.close()
    
    # 3. Model type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    type_counts = best_models_df['type'].value_counts()
    colors_pie = [color_map.get(t, 'gray') for t in type_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    ax.set_title('Best Models by Type Distribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_type_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_type_distribution.png'}")
    plt.close()
    
    print("\n" + "="*80)
    print(f"All summary visualizations saved to: {output_dir}")
    print("="*80 + "\n")


def main():
    """Main function"""
    create_summary_comparison()


if __name__ == '__main__':
    main()



