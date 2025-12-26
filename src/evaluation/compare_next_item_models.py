"""
Compare Next Item Prediction Models
Evaluates and compares different sequential recommendation approaches
Only compares 3 models: Markov Chain (baseline), ItemKNN, GRU4Rec
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'api'))

from config import config
from database import get_db_connection
from psycopg2.extras import RealDictCursor

# Import models
from src.models.next_item.train_item_knn import ItemKNN
from src.models.next_item.train_fpmc import FPMC
from src.models.next_item.train_gru4rec import GRU4Rec, TORCH_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NextItemComparison:
    """Compare different next item prediction models"""
    
    def __init__(self, store_id=1):
        self.store_id = store_id
        self.results_dir = Path(__file__).parent.parent.parent / 'results' / 'next_item'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_sequences(self):
        """Extract purchase sequences from database"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Extracting sequences for store {self.store_id}...")
        logger.info(f"{'='*70}\n")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT 
                    o.customer_id,
                    o.id as order_id,
                    o.created_at,
                    li.variant_id as item_id
                FROM orders o
                JOIN line_items li ON o.id = li.reference_id AND li.reference_type = 'order'
                WHERE o.store_id = %s
                ORDER BY o.customer_id, o.created_at, li.id
            """, (self.store_id,))
            
            data = cur.fetchall()
            logger.info(f"Retrieved {len(data)} order-item records")
            
            # Group by customer
            customer_sequences = defaultdict(list)
            
            for row in data:
                customer_id = row['customer_id']
                item_id = row['item_id']
                customer_sequences[customer_id].append(item_id)
            
            # Convert to list format
            sequences = [(cid, items) for cid, items in customer_sequences.items()]
            
            logger.info(f"Total customers: {len(sequences)}")
            logger.info(f"Avg sequence length: {np.mean([len(items) for _, items in sequences]):.2f}")
            
            return sequences
            
        finally:
            cur.close()
            conn.close()
    
    def split_sequences(self, sequences, test_size=0.2):
        """
        Split sequences into train/test
        Use last N items as test set for each user
        """
        train_sequences = []
        test_data = []
        
        for customer_id, items in sequences:
            if len(items) < 3:  # Need at least 3 items (2 for history, 1 for test)
                continue
            
            # Calculate split point
            n_test = max(1, int(len(items) * test_size))
            split_idx = len(items) - n_test
            
            # Train: all items up to split
            train_items = items[:split_idx]
            train_sequences.append((customer_id, train_items))
            
            # Test: predict each item in test set given history
            for i in range(split_idx, len(items)):
                history = items[:i]
                target = items[i]
                test_data.append((customer_id, history, target))
        
        logger.info(f"\nData split:")
        logger.info(f"  Train sequences: {len(train_sequences)}")
        logger.info(f"  Test cases: {len(test_data)}")
        
        return train_sequences, test_data
    
    def evaluate_model(self, model, test_data, model_name, k_values=[5, 10, 20]):
        """
        Evaluate a model on test data
        
        Args:
            model: Trained model with predict() method
            test_data: List of (customer_id, history, target) tuples
            model_name: Name of the model
            k_values: List of k values for top-k metrics
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        metrics = {k: {'hit': 0, 'mrr': 0, 'ndcg': 0} for k in k_values}
        total_cases = len(test_data)
        
        start_time = time.time()
        
        for idx, (customer_id, history, target) in enumerate(test_data):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Progress: {idx + 1}/{total_cases}")
            
            # Get predictions
            try:
                if hasattr(model, 'predict') and 'user_id' in model.predict.__code__.co_varnames:
                    # FPMC needs user_id
                    predictions = model.predict(customer_id, history, top_k=max(k_values))
                else:
                    predictions = model.predict(history, top_k=max(k_values))
                
                predicted_items = [item for item, score in predictions]
                
                # Calculate metrics for each k
                for k in k_values:
                    top_k_items = predicted_items[:k]
                    
                    # Hit Rate
                    if target in top_k_items:
                        metrics[k]['hit'] += 1
                        
                        # MRR (Mean Reciprocal Rank)
                        rank = top_k_items.index(target) + 1
                        metrics[k]['mrr'] += 1.0 / rank
                        
                        # NDCG
                        metrics[k]['ndcg'] += 1.0 / np.log2(rank + 1)
                
            except Exception as e:
                logger.warning(f"Prediction failed for customer {customer_id}: {str(e)}")
                continue
        
        eval_time = time.time() - start_time
        
        # Average metrics
        results = {'model': model_name}
        
        for k in k_values:
            results[f'hit_rate@{k}'] = metrics[k]['hit'] / total_cases
            results[f'mrr@{k}'] = metrics[k]['mrr'] / total_cases
            results[f'ndcg@{k}'] = metrics[k]['ndcg'] / total_cases
        
        results['eval_time'] = eval_time
        results['test_cases'] = total_cases
        
        return results
    
    def run_comparison(self):
        """Run complete model comparison"""
        logger.info(f"\n{'='*70}")
        logger.info(f"NEXT ITEM PREDICTION MODEL COMPARISON")
        logger.info(f"Store ID: {self.store_id}")
        logger.info(f"{'='*70}\n")
        
        # 1. Extract sequences
        sequences = self.extract_sequences()
        
        # 2. Split data
        train_sequences, test_data = self.split_sequences(sequences)
        
        # 3. Define models to compare (only 3 models: baseline + 2 others)
        models_config = [
            {
                'name': 'Markov Chain',
                'class': 'transition',  # Baseline: Pattern Mining
                'params': {},
                'type': 'Baseline'
            },
            {
                'name': 'ItemKNN',
                'class': ItemKNN,
                'params': {'k': 20},
                'type': 'Collaborative Filtering'
            },
        ]
        
        # Add GRU4Rec if PyTorch is available
        if TORCH_AVAILABLE:
            models_config.append({
                'name': 'GRU4Rec',
                'class': GRU4Rec,
                'params': {'hidden_size': 200, 'n_epochs': 10},
                'type': 'Deep Learning'
            })
        else:
            logger.warning("PyTorch not available. Skipping GRU4Rec. Using FPMC instead.")
            models_config.append({
                'name': 'FPMC',
                'class': FPMC,
                'params': {'n_factors': 64, 'n_epochs': 20},
                'type': 'Deep Learning'
            })
        
        # 4. Train and evaluate each model
        all_results = []
        
        for config in models_config:
            model_name = config['name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*70}")
            
            try:
                # Train model
                start_time = time.time()
                
                if config['class'] == 'transition':
                    # Use our original transition-based model
                    model = self._train_transition_model(train_sequences)
                else:
                    model = config['class'](**config['params'])
                    model.fit(train_sequences)
                
                train_time = time.time() - start_time
                
                logger.info(f"Training time: {train_time:.2f}s")
                
                # Evaluate model
                results = self.evaluate_model(model, test_data, model_name)
                results['train_time'] = train_time
                results['model_type'] = config.get('type', 'Unknown')
                
                all_results.append(results)
                
                # Save model (skip for TransitionModel as it's a local class and can't be pickled)
                if config['class'] != 'transition':
                    model_path = self.results_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Model saved: {model_path}")
                else:
                    logger.info(f"Skipping model save for {model_name} (local class, can't pickle)")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 5. Save and display results
        results_df = pd.DataFrame(all_results)
        
        # Sort by NDCG@10
        results_df = results_df.sort_values('ndcg@10', ascending=False)
        
        # Save to CSV
        csv_path = self.results_dir / f'store_{self.store_id}_comparison.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")
        
        # Display results
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARISON RESULTS")
        logger.info(f"{'='*70}\n")
        
        # Format for display
        display_df = results_df.copy()
        
        # Round numeric columns
        numeric_cols = [col for col in display_df.columns if col != 'model' and col != 'model_type']
        for col in numeric_cols:
            if 'time' in col:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{float(x):.2f}s" if pd.notna(x) and isinstance(x, (int, float, np.number)) else str(x) if pd.notna(x) else "N/A"
                )
            elif 'cases' in col:
                display_df[col] = display_df[col].apply(
                    lambda x: int(x) if pd.notna(x) and isinstance(x, (int, float, np.number)) else x
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else str(x) if pd.notna(x) else "N/A"
                )
        
        print(display_df.to_string(index=False))
        
        # Highlight best model
        best_model = results_df.iloc[0]['model']
        best_ndcg = results_df.iloc[0]['ndcg@10']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"NDCG@10: {best_ndcg:.4f}")
        logger.info(f"{'='*70}\n")
        
        # Create visualizations
        logger.info("Creating visualizations...")
        self.create_visualizations(results_df)
        
        return results_df
    
    def _train_transition_model(self, sequences):
        """Train our original transition-based model"""
        from collections import Counter
        
        class TransitionModel:
            def __init__(self):
                self.item_transitions = defaultdict(Counter)
                
            def fit(self, sequences):
                for customer_id, items in sequences:
                    for i in range(len(items) - 1):
                        curr_item = items[i]
                        next_item = items[i + 1]
                        self.item_transitions[curr_item][next_item] += 1
                return self
            
            def predict(self, history, top_k=10):
                if not history:
                    return []
                
                last_item = history[-1]
                
                if last_item not in self.item_transitions:
                    return []
                
                # Get transitions
                transitions = self.item_transitions[last_item]
                total = sum(transitions.values())
                
                # Calculate probabilities
                predictions = [
                    (item, count / total)
                    for item, count in transitions.most_common(top_k)
                ]
                
                return predictions
        
        model = TransitionModel()
        model.fit(sequences)
        return model
    
    def create_visualizations(self, results_df):
        """Create comparison visualizations"""
        viz_dir = self.results_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        # 1. Bar plot of main metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['hit_rate@10', 'ndcg@10', 'mrr@10', 'train_time']
        titles = ['Hit Rate@10', 'NDCG@10', 'MRR@10', 'Training Time (s)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Sort by metric value
            sorted_df = results_df.sort_values(metric, ascending=(metric != 'train_time'))
            
            # Color by model type
            colors = []
            for model_type in sorted_df['model_type']:
                if model_type == 'Baseline':
                    colors.append('#3498db')
                elif model_type == 'Collaborative Filtering':
                    colors.append('#2ecc71')
                else:
                    colors.append('#e74c3c')
            
            bars = ax.barh(sorted_df['model'], sorted_df[metric], color=colors)
            
            # Highlight best (lowest for train_time, highest for others)
            if metric == 'train_time':
                best_idx = sorted_df[metric].idxmin()
            else:
                best_idx = sorted_df[metric].idxmax()
            bars[sorted_df.index.get_loc(best_idx)].set_edgecolor('gold')
            bars[sorted_df.index.get_loc(best_idx)].set_linewidth(3)
            
            ax.set_xlabel(title)
            ax.set_title(f'{title} Comparison')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(sorted_df[metric]):
                if metric == 'train_time':
                    label = f' {v:.2f}s' if v >= 0.01 else f' {v*1000:.2f}ms'
                else:
                    label = f' {v:.4f}'
                ax.text(v, i, label, va='center')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {viz_dir / 'metrics_comparison.png'}")
        plt.close()
        
        # 2. Performance vs Speed scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by model type
        color_map = {
            'Baseline': '#3498db',
            'Collaborative Filtering': '#2ecc71',
            'Deep Learning': '#e74c3c'
        }
        
        for model_type in results_df['model_type'].unique():
            mask = results_df['model_type'] == model_type
            ax.scatter(
                results_df[mask]['train_time'],
                results_df[mask]['hit_rate@10'],
                c=color_map.get(model_type, 'gray'),
                s=200,
                alpha=0.7,
                label=model_type
            )
            
            # Add labels
            for _, row in results_df[mask].iterrows():
                ax.annotate(
                    row['model'],
                    (row['train_time'], row['hit_rate@10']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold'
                )
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hit Rate@10', fontsize=12, fontweight='bold')
        ax.set_title('Next Item Models: Performance vs Speed Trade-off', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_vs_speed.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {viz_dir / 'performance_vs_speed.png'}")
        plt.close()
        
        # 3. Radar chart for all models
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['Hit Rate@10', 'NDCG@10', 'MRR@10']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Normalize metrics to 0-1 for radar chart
        metrics_to_plot = ['hit_rate@10', 'ndcg@10', 'mrr@10']
        normalized_df = results_df.copy()
        for metric in metrics_to_plot:
            max_val = normalized_df[metric].max()
            min_val = normalized_df[metric].min()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 1.0
        
        color_map_radar = {
            'Baseline': '#3498db',
            'Collaborative Filtering': '#2ecc71',
            'Deep Learning': '#e74c3c'
        }
        
        for _, row in results_df.iterrows():
            values = [
                normalized_df.loc[row.name, 'hit_rate@10_norm'],
                normalized_df.loc[row.name, 'ndcg@10_norm'],
                normalized_df.loc[row.name, 'mrr@10_norm']
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['model'], 
                   color=color_map_radar.get(row['model_type'], 'gray'))
            ax.fill(angles, values, alpha=0.1, color=color_map_radar.get(row['model_type'], 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)
        ax.set_title('Next Item Models - Performance Radar Chart', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {viz_dir / 'radar_chart.png'}")
        plt.close()
        
        logger.info(f"All visualizations saved to: {viz_dir}")


def main():
    """Main function"""
    comparison = NextItemComparison(store_id=1)
    results = comparison.run_comparison()
    
    logger.info("\nComparison completed successfully!")


if __name__ == '__main__':
    main()
