"""
Next Item Prediction Trainer
Sequential recommendation using item transition patterns
"""

import pandas as pd
import numpy as np
import logging
import time
import pickle
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database import get_db_connection
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class NextItemTrainer:
    """Trains next item prediction model using sequential patterns"""
    
    def __init__(self, store_id):
        self.store_id = store_id
        self.order = config.NEXT_ITEM_ORDER
        self.min_support = config.NEXT_ITEM_MIN_SUPPORT
        self.smoothing = config.NEXT_ITEM_SMOOTHING
        
    def extract_purchase_sequences(self):
        """Extract sequential purchase data for each customer"""
        logger.info(f"\nStep 1: Extracting purchase sequences for store {self.store_id}...")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT 
                    o.customer_id,
                    o.id as order_id,
                    o.created_at,
                    li.variant_id as item_id,
                    v.product_id,
                    p.name as product_name,
                    pt.name as product_type,
                    li.quantity,
                    li.price
                FROM orders o
                JOIN line_items li ON li.reference_id = o.id AND li.reference_type = 'order'
                LEFT JOIN variants v ON v.id = li.variant_id
                LEFT JOIN products p ON p.id = v.product_id
                LEFT JOIN product_types pt ON pt.id = p.product_type_id
                WHERE o.store_id = %s
                    AND o.status IN ('completed', 'confirmed')
                    AND o.financial_status IN ('paid', 'partial_paid')
                    AND o.deleted_at IS NULL
                    AND li.variant_id IS NOT NULL
                ORDER BY o.customer_id, o.created_at, li.variant_id
            """, (self.store_id,))
            
            data = cur.fetchall()
            df = pd.DataFrame(data)
            
            if len(df) == 0:
                raise ValueError(f"No purchase data found for store {self.store_id}")
            
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            logger.info(f"  ✅ Extracted {len(df):,} purchase records")
            logger.info(f"  ✅ Customers: {df['customer_id'].nunique():,}")
            logger.info(f"  ✅ Unique items: {df['item_id'].nunique():,}")
            logger.info(f"  ✅ Orders: {df['order_id'].nunique():,}")
            
            return df
            
        finally:
            cur.close()
            conn.close()
    
    def build_item_sequences(self, df):
        """Build item sequences for each customer"""
        logger.info(f"\nStep 2: Building item sequences...")
        
        sequences = []
        sequence_df = df.groupby('customer_id').apply(
            lambda x: x.sort_values('created_at')[['item_id', 'product_name', 'product_type', 'created_at']].values.tolist()
        ).reset_index()
        sequence_df.columns = ['customer_id', 'sequence']
        
        # Extract just item IDs for pattern mining
        for idx, row in sequence_df.iterrows():
            customer_id = row['customer_id']
            full_sequence = row['sequence']
            item_sequence = [item[0] for item in full_sequence]  # item_id
            
            if len(item_sequence) >= 2:  # Need at least 2 items for pattern
                sequences.append({
                    'customer_id': customer_id,
                    'items': item_sequence,
                    'length': len(item_sequence)
                })
        
        logger.info(f"  ✅ Built {len(sequences):,} sequences")
        logger.info(f"  ✅ Avg sequence length: {np.mean([s['length'] for s in sequences]):.1f} items")
        logger.info(f"  ✅ Max sequence length: {max([s['length'] for s in sequences])} items")
        
        return sequences
    
    def mine_transition_patterns(self, sequences, min_support=2):
        """Mine item-to-item transition patterns"""
        logger.info(f"\nStep 3: Mining transition patterns (min_support={min_support}, smoothing={self.smoothing})...")
        
        # Count transitions: item_i → item_j
        transitions = defaultdict(lambda: defaultdict(int))
        item_counts = Counter()
        
        for seq_data in sequences:
            seq = seq_data['items']
            
            # Count item frequencies
            for item in seq:
                item_counts[item] += 1
            
            # Count transitions (bigrams)
            for i in range(len(seq) - 1):
                current_item = seq[i]
                next_item = seq[i + 1]
                
                if current_item != next_item:  # Skip same item transitions
                    transitions[current_item][next_item] += 1
        
        # Calculate transition probabilities with Laplace smoothing
        transition_probs = {}
        vocab_size = len(item_counts)  # Number of unique items
        
        for item, next_items in transitions.items():
            total_transitions = sum(next_items.values())
            
            if total_transitions >= min_support:
                transition_probs[item] = {}
                for next_item, count in next_items.items():
                    if count >= min_support:
                        # Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                        prob = (count + self.smoothing) / (total_transitions + self.smoothing * vocab_size)
                        transition_probs[item][next_item] = {
                            'probability': prob,
                            'count': count,
                            'confidence': count / total_transitions  # Raw confidence without smoothing
                        }
        
        logger.info(f"  ✅ Found {len(transition_probs):,} items with transitions")
        
        # Count total patterns
        total_patterns = sum(len(next_items) for next_items in transition_probs.values())
        logger.info(f"  ✅ Total transition patterns: {total_patterns:,}")
        
        # Show top patterns
        logger.info(f"\n  Top 10 strongest transitions:")
        all_transitions = []
        for item, next_items in transition_probs.items():
            for next_item, stats in next_items.items():
                all_transitions.append((item, next_item, stats['probability'], stats['count']))
        
        all_transitions.sort(key=lambda x: (x[3], x[2]), reverse=True)  # Sort by count, then probability
        
        for i, (item, next_item, prob, count) in enumerate(all_transitions[:10], 1):
            logger.info(f"    {i}. Item {item} → Item {next_item}: {prob:.2%} ({count} times)")
        
        return transition_probs, item_counts
    
    def mine_product_type_patterns(self, df, min_support=3):
        """Mine category-level transition patterns"""
        logger.info(f"\nStep 4: Mining product type patterns...")
        
        # Build sequences at category level
        category_sequences = df.groupby('customer_id').apply(
            lambda x: x.sort_values('created_at')['product_type'].dropna().tolist()
        ).reset_index()[0].values
        
        # Count category transitions
        category_transitions = defaultdict(lambda: defaultdict(int))
        
        for seq in category_sequences:
            if len(seq) < 2:
                continue
            
            for i in range(len(seq) - 1):
                current = seq[i]
                next_cat = seq[i + 1]
                
                if current and next_cat and current != next_cat:
                    category_transitions[current][next_cat] += 1
        
        # Calculate probabilities
        category_probs = {}
        for cat, next_cats in category_transitions.items():
            total = sum(next_cats.values())
            
            if total >= min_support:
                category_probs[cat] = {}
                for next_cat, count in next_cats.items():
                    if count >= min_support:
                        category_probs[cat][next_cat] = {
                            'probability': count / total,
                            'count': count
                        }
        
        logger.info(f"  ✅ Found {len(category_probs):,} category transitions")
        
        # Show top category patterns
        logger.info(f"\n  Top 5 category transitions:")
        all_cat_trans = []
        for cat, next_cats in category_probs.items():
            for next_cat, stats in next_cats.items():
                all_cat_trans.append((cat, next_cat, stats['probability'], stats['count']))
        
        all_cat_trans.sort(key=lambda x: x[3], reverse=True)
        
        for i, (cat, next_cat, prob, count) in enumerate(all_cat_trans[:5], 1):
            logger.info(f"    {i}. {cat} → {next_cat}: {prob:.2%} ({count} times)")
        
        return category_probs
    
    def predict_next_items(self, item_history, transition_probs, top_k=10):
        """Predict next items based on purchase history"""
        if not item_history:
            return []
        
        # Use last item for prediction
        last_item = item_history[-1]
        
        if last_item not in transition_probs:
            return []
        
        # Get predictions
        predictions = transition_probs[last_item]
        
        # Sort by probability
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: (x[1]['probability'], x[1]['count']),
            reverse=True
        )
        
        results = []
        for item_id, stats in sorted_predictions[:top_k]:
            results.append({
                'item_id': int(item_id),
                'probability': float(stats['probability']),
                'confidence': float(stats['confidence']),
                'support': int(stats['count'])
            })
        
        return results
    
    def train(self):
        """Train next item prediction model"""
        logger.info(f"=" * 60)
        logger.info(f"NEXT ITEM PREDICTION - Store {self.store_id}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Extract purchase sequences
        df = self.extract_purchase_sequences()
        
        # Build item sequences
        sequences = self.build_item_sequences(df)
        
        # Mine transition patterns
        transition_probs, item_counts = self.mine_transition_patterns(sequences, min_support=self.min_support)
        
        # Mine category patterns
        category_probs = self.mine_product_type_patterns(df)
        
        training_time = time.time() - start_time
        
        logger.info(f"\n✅ Next item prediction model trained in {training_time:.2f}s")
        
        # Prepare results
        result = {
            'transition_patterns': transition_probs,
            'category_patterns': category_probs,
            'item_counts': dict(item_counts),
            'metrics': {
                'num_customers': int(df['customer_id'].nunique()),
                'num_sequences': len(sequences),
                'num_items': len(item_counts),
                'num_transitions': sum(len(next_items) for next_items in transition_probs.values()),
                'num_category_transitions': sum(len(next_cats) for next_cats in category_probs.values()),
                'avg_sequence_length': float(np.mean([s['length'] for s in sequences])),
                'training_time': int(training_time)
            },
            'store_id': self.store_id,
            'trained_at': datetime.now().isoformat()
        }
        
        return result
