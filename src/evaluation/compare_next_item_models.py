"""
Compare Next Item Prediction Models
Evaluates and compares different sequential recommendation approaches
"""

import pandas as pd
import numpy as np
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
        
        # 3. Define models to compare
        models_config = [
            {
                'name': 'Markov Chain (Transition)',
                'class': 'transition',  # Our original model
                'params': {}
            },
            {
                'name': 'Item-KNN (k=20)',
                'class': ItemKNN,
                'params': {'k': 20}
            },
            {
                'name': 'Item-KNN (k=50)',
                'class': ItemKNN,
                'params': {'k': 50}
            },
            {
                'name': 'FPMC (factors=32)',
                'class': FPMC,
                'params': {'n_factors': 32, 'n_epochs': 20}
            },
            {
                'name': 'FPMC (factors=64)',
                'class': FPMC,
                'params': {'n_factors': 64, 'n_epochs': 20}
            },
        ]
        
        # Add GRU4Rec if PyTorch is available
        if TORCH_AVAILABLE:
            models_config.extend([
                {
                    'name': 'GRU4Rec (hidden=100)',
                    'class': GRU4Rec,
                    'params': {'hidden_size': 100, 'n_epochs': 10}
                },
                {
                    'name': 'GRU4Rec (hidden=200)',
                    'class': GRU4Rec,
                    'params': {'hidden_size': 200, 'n_epochs': 10}
                },
            ])
        else:
            logger.warning("PyTorch not available. Skipping GRU4Rec models.")
        
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
                
                all_results.append(results)
                
                # Save model
                model_path = self.results_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved: {model_path}")
                
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
        numeric_cols = [col for col in display_df.columns if col != 'model']
        for col in numeric_cols:
            if 'time' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}s")
            elif 'cases' in col:
                display_df[col] = display_df[col].astype(int)
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        # Highlight best model
        best_model = results_df.iloc[0]['model']
        best_ndcg = results_df.iloc[0]['ndcg@10']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"NDCG@10: {best_ndcg:.4f}")
        logger.info(f"{'='*70}\n")
        
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


def main():
    """Main function"""
    comparison = NextItemComparison(store_id=1)
    results = comparison.run_comparison()
    
    logger.info("\nComparison completed successfully!")


if __name__ == '__main__':
    main()
