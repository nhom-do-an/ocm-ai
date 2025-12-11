"""
Comprehensive comparison of all trending prediction models
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
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / 'results' / 'trending'


def load_all_results():
    """Load results from all trending models"""
    
    print("Loading results from all trending models...\n")
    
    # Load baseline models
    baseline_file = results_dir / 'baseline_models_all_stores.csv'
    if baseline_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        print(f"✓ Baseline models: {len(baseline_df)} models")
    else:
        print(f"✗ Baseline models not found: {baseline_file}")
        baseline_df = pd.DataFrame()
    
    # Load LSTM results
    lstm_file = results_dir / 'lstm_models_results.csv'
    if lstm_file.exists():
        lstm_df = pd.read_csv(lstm_file)
        print(f"✓ LSTM models: {len(lstm_df)} models")
    else:
        print(f"✗ LSTM models not found: {lstm_file}")
        lstm_df = pd.DataFrame()
    
    # Load LightGBM results
    lgb_file = results_dir / 'lightgbm_results.csv'
    if lgb_file.exists():
        lgb_df = pd.read_csv(lgb_file)
        print(f"✓ LightGBM models: {len(lgb_df)} models")
    else:
        print(f"✗ LightGBM models not found: {lgb_file}")
        lgb_df = pd.DataFrame()
    
    # Combine all results
    all_results = []
    
    if not baseline_df.empty:
        baseline_df['model_type'] = 'Baseline'
        all_results.append(baseline_df)
    
    if not lstm_df.empty:
        lstm_df['model_type'] = 'Deep Learning'
        all_results.append(lstm_df)
    
    if not lgb_df.empty:
        lgb_df['model_type'] = 'Gradient Boosting'
        all_results.append(lgb_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal models: {len(combined_df)}")
        return combined_df
    else:
        print("\n⚠ No results found!")
        return pd.DataFrame()


def create_mae_comparison(df, output_dir):
    """Create MAE comparison plot"""
    
    plt.figure(figsize=(14, 8))
    
    # Sort by MAE
    df_sorted = df.sort_values('mae')
    
    # Create color map by model type
    colors = []
    for model_type in df_sorted['model_type']:
        if model_type == 'Baseline':
            colors.append('#3498db')
        elif model_type == 'Deep Learning':
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')
    
    # Plot
    plt.barh(range(len(df_sorted)), df_sorted['mae'], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['model_name'])
    plt.xlabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Trending Prediction Models - MAE Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (mae, model_type) in enumerate(zip(df_sorted['mae'], df_sorted['model_type'])):
        plt.text(mae, i, f' {mae:.4f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e74c3c', label='Deep Learning'),
        Patch(facecolor='#2ecc71', label='Gradient Boosting')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mae_comparison.png")
    plt.close()


def create_mape_comparison(df, output_dir):
    """Create MAPE comparison plot"""
    
    plt.figure(figsize=(14, 8))
    
    # Sort by MAPE
    df_sorted = df.sort_values('mape')
    
    # Create color map by model type
    colors = []
    for model_type in df_sorted['model_type']:
        if model_type == 'Baseline':
            colors.append('#3498db')
        elif model_type == 'Deep Learning':
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')
    
    # Plot
    plt.barh(range(len(df_sorted)), df_sorted['mape'], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['model_name'])
    plt.xlabel('Mean Absolute Percentage Error (MAPE %)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Trending Prediction Models - MAPE Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, mape in enumerate(df_sorted['mape']):
        plt.text(mape, i, f' {mape:.2f}%', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e74c3c', label='Deep Learning'),
        Patch(facecolor='#2ecc71', label='Gradient Boosting')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mape_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mape_comparison.png")
    plt.close()


def create_top_k_comparison(df, output_dir):
    """Create Top-K accuracy comparison plot"""
    
    # Find Top-K column
    top_k_col = None
    for col in df.columns:
        if 'top_' in col.lower() and 'accuracy' in col.lower():
            top_k_col = col
            break
    
    if top_k_col is None:
        print("⚠ No Top-K accuracy column found")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Sort by Top-K accuracy
    df_sorted = df.sort_values(top_k_col, ascending=False)
    
    # Create color map by model type
    colors = []
    for model_type in df_sorted['model_type']:
        if model_type == 'Baseline':
            colors.append('#3498db')
        elif model_type == 'Deep Learning':
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')
    
    # Plot
    plt.barh(range(len(df_sorted)), df_sorted[top_k_col], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['model_name'])
    plt.xlabel('Top-K Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Trending Prediction Models - Top-K Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, acc in enumerate(df_sorted[top_k_col]):
        plt.text(acc, i, f' {acc:.4f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e74c3c', label='Deep Learning'),
        Patch(facecolor='#2ecc71', label='Gradient Boosting')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_k_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: top_k_comparison.png")
    plt.close()


def create_train_time_comparison(df, output_dir):
    """Create training time comparison plot"""
    
    # Find training time column
    time_col = None
    for col in ['train_time', 'avg_train_time', 'avg_train_time_per_item', 'total_time']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("⚠ No training time column found")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Sort by training time
    df_sorted = df.sort_values(time_col)
    
    # Create color map by model type
    colors = []
    for model_type in df_sorted['model_type']:
        if model_type == 'Baseline':
            colors.append('#3498db')
        elif model_type == 'Deep Learning':
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')
    
    # Plot
    plt.barh(range(len(df_sorted)), df_sorted[time_col], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['model_name'])
    plt.xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Trending Prediction Models - Training Time Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, time_val in enumerate(df_sorted[time_col]):
        if time_val < 1:
            label = f' {time_val*1000:.2f}ms'
        else:
            label = f' {time_val:.2f}s'
        plt.text(time_val, i, label, va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e74c3c', label='Deep Learning'),
        Patch(facecolor='#2ecc71', label='Gradient Boosting')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'train_time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: train_time_comparison.png")
    plt.close()


def create_scatter_mae_vs_time(df, output_dir):
    """Create scatter plot: MAE vs Training Time"""
    
    # Find training time column
    time_col = None
    for col in ['train_time', 'avg_train_time', 'avg_train_time_per_item', 'total_time']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("⚠ No training time column found for scatter plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot by model type
    for model_type, color, marker in [
        ('Baseline', '#3498db', 'o'),
        ('Deep Learning', '#e74c3c', 's'),
        ('Gradient Boosting', '#2ecc71', '^')
    ]:
        mask = df['model_type'] == model_type
        if mask.any():
            plt.scatter(
                df[mask][time_col],
                df[mask]['mae'],
                c=color,
                s=150,
                alpha=0.7,
                marker=marker,
                label=model_type
            )
            
            # Add labels
            for _, row in df[mask].iterrows():
                plt.annotate(
                    row['model_name'],
                    (row[time_col], row['mae']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
    
    plt.xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    plt.title('Trending Models: Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_vs_time_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mae_vs_time_scatter.png")
    plt.close()


def generate_final_report(df, output_dir):
    """Generate final Markdown report"""
    
    report_file = output_dir / 'TRENDING_MODELS_FINAL_REPORT.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Trending Prediction Models - Final Comparison Report\n\n")
        
        # Overview
        f.write("## Executive Summary\n\n")
        f.write(f"This report compares **{len(df)} trending prediction models** across multiple categories:\n\n")
        
        baseline_count = len(df[df['model_type'] == 'Baseline'])
        dl_count = len(df[df['model_type'] == 'Deep Learning'])
        gb_count = len(df[df['model_type'] == 'Gradient Boosting'])
        
        f.write(f"- **Baseline Models**: {baseline_count} (Statistical methods)\n")
        f.write(f"- **Deep Learning Models**: {dl_count} (LSTM)\n")
        f.write(f"- **Gradient Boosting Models**: {gb_count} (LightGBM)\n\n")
        
        # Top performers
        f.write("## Top Performers\n\n")
        
        # Best MAE
        best_mae = df.loc[df['mae'].idxmin()]
        f.write(f"### 🥇 Best Accuracy (Lowest MAE)\n\n")
        f.write(f"**{best_mae['model_name']}** ({best_mae['model_type']})\n\n")
        f.write(f"- MAE: **{best_mae['mae']:.4f}**\n")
        f.write(f"- RMSE: {best_mae['rmse']:.4f}\n")
        f.write(f"- MAPE: {best_mae['mape']:.2f}%\n\n")
        
        # Find Top-K column
        top_k_col = None
        for col in df.columns:
            if 'top_' in col.lower() and 'accuracy' in col.lower():
                top_k_col = col
                break
        
        if top_k_col:
            best_topk = df.loc[df[top_k_col].idxmax()]
            f.write(f"### 🎯 Best Top-K Accuracy\n\n")
            f.write(f"**{best_topk['model_name']}** ({best_topk['model_type']})\n\n")
            f.write(f"- Top-K Accuracy: **{best_topk[top_k_col]:.4f}**\n")
            f.write(f"- MAE: {best_topk['mae']:.4f}\n\n")
        
        # Fastest
        time_col = None
        for col in ['train_time', 'avg_train_time', 'avg_train_time_per_item', 'total_time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            fastest = df.loc[df[time_col].idxmin()]
            f.write(f"### ⚡ Fastest Training\n\n")
            f.write(f"**{fastest['model_name']}** ({fastest['model_type']})\n\n")
            f.write(f"- Training Time: **{fastest[time_col]:.6f}s**\n")
            f.write(f"- MAE: {fastest['mae']:.4f}\n\n")
        
        # Detailed results
        f.write("## Detailed Results\n\n")
        
        # Sort by MAE
        df_sorted = df.sort_values('mae')
        
        f.write("| Rank | Model | Type | MAE | RMSE | MAPE (%) |\n")
        f.write("|------|-------|------|-----|------|----------|\n")
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"
            f.write(f"| {medal} | {row['model_name']} | {row['model_type']} | "
                   f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['mape']:.2f} |\n")
        
        f.write("\n")
        
        # Analysis by model type
        f.write("## Performance by Model Type\n\n")
        
        for model_type in ['Gradient Boosting', 'Deep Learning', 'Baseline']:
            type_df = df[df['model_type'] == model_type]
            if not type_df.empty:
                f.write(f"### {model_type}\n\n")
                avg_mae = type_df['mae'].mean()
                avg_mape = type_df['mape'].mean()
                f.write(f"- Average MAE: {avg_mae:.4f}\n")
                f.write(f"- Average MAPE: {avg_mape:.2f}%\n")
                f.write(f"- Models: {len(type_df)}\n\n")
        
        # Recommendations
        f.write("## Production Recommendations\n\n")
        
        f.write("### For Real-Time Trending Prediction\n\n")
        
        if best_mae['model_type'] == 'Gradient Boosting':
            f.write(f"**Recommended: {best_mae['model_name']}**\n\n")
            f.write("**Rationale:**\n")
            f.write(f"- Outstanding accuracy (MAE: {best_mae['mae']:.4f})\n")
            f.write("- Fast inference time\n")
            f.write("- Handles time series features effectively\n")
            f.write("- Robust to missing data\n\n")
        
        # Baseline fallback
        baseline_best = df[df['model_type'] == 'Baseline'].sort_values('mae').iloc[0]
        f.write(f"### Baseline Fallback: {baseline_best['model_name']}\n\n")
        f.write("**Use Case:** Cold-start items with limited history\n\n")
        f.write(f"- MAE: {baseline_best['mae']:.4f}\n")
        f.write("- Extremely fast\n")
        f.write("- No training required\n\n")
        
        # Key insights
        f.write("## Key Insights\n\n")
        
        mae_improvement = ((baseline_best['mae'] - best_mae['mae']) / baseline_best['mae']) * 100
        f.write(f"1. **{best_mae['model_name']}** outperforms baseline by **{mae_improvement:.1f}%**\n")
        
        if top_k_col and best_topk[top_k_col] == 1.0:
            f.write(f"2. **Perfect Top-K prediction**: {best_topk['model_name']} achieved 100% Top-K accuracy\n")
        
        f.write(f"3. **Feature importance**: Time series features (velocity, lags) are critical\n")
        f.write(f"4. **Trade-off**: Deep learning offers good accuracy but requires more training time\n")
        
        f.write("\n## Conclusion\n\n")
        f.write(f"The **{best_mae['model_name']}** model demonstrates superior performance for trending prediction, ")
        f.write("achieving state-of-the-art accuracy while maintaining reasonable training times. ")
        f.write("For production deployment, we recommend a hybrid approach: ")
        f.write(f"use **{best_mae['model_name']}** for items with sufficient history, ")
        f.write(f"and fall back to **{baseline_best['model_name']}** for cold-start scenarios.\n\n")
        
        f.write("---\n")
        f.write("*Report generated automatically by compare_trending_models.py*\n")
    
    print(f"✓ Saved: {report_file.name}")


def create_comprehensive_comparison():
    """Create comprehensive comparison of all trending models"""
    
    print("\n" + "="*80)
    print("TRENDING PREDICTION MODELS - COMPREHENSIVE COMPARISON")
    print("="*80 + "\n")
    
    # Load results
    df = load_all_results()
    
    if df.empty:
        print("No results to compare!")
        return
    
    # Create output directory
    output_dir = results_dir / 'trending_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating comparison visualizations...\n")
    
    # Create plots
    create_mae_comparison(df, output_dir)
    create_mape_comparison(df, output_dir)
    create_top_k_comparison(df, output_dir)
    create_train_time_comparison(df, output_dir)
    create_scatter_mae_vs_time(df, output_dir)
    
    # Generate report
    print("\nGenerating final report...\n")
    generate_final_report(df, output_dir)
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80 + "\n")
    
    # Print summary
    print("\n" + "="*80)
    print("TOP 5 MODELS BY MAE")
    print("="*80)
    
    top5 = df.nsmallest(5, 'mae')[['model_name', 'model_type', 'mae', 'rmse', 'mape']]
    print(top5.to_string(index=False))
    
    print("\n" + "="*80)


def main():
    """Main function"""
    create_comprehensive_comparison()


if __name__ == '__main__':
    main()
