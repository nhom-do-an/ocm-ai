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
        """Build item sequences for each customer (grouped by order)"""
        logger.info(f"\nStep 2: Building item sequences (grouped by order)...")
        
        sequences = []
        
        # Group by customer first
        for customer_id, customer_data in df.groupby('customer_id'):
            # Group by order_id to handle multiple items in same order
            order_groups = []
            
            for order_id, order_data in customer_data.groupby('order_id'):
                # Get all items in this order (sorted by variant_id for consistency)
                items_in_order = sorted(order_data['item_id'].unique().tolist())
                order_groups.append({
                    'order_id': order_id,
                    'created_at': order_data['created_at'].iloc[0],
                    'items': items_in_order
                })
            
            # Sort orders by created_at
            order_groups.sort(key=lambda x: x['created_at'])
            
            # Build sequence: flatten items but keep track of order boundaries
            item_sequence = []
            for order_group in order_groups:
                item_sequence.extend(order_group['items'])
            
            if len(item_sequence) >= 2:  # Need at least 2 items for pattern
                sequences.append({
                    'customer_id': customer_id,
                    'items': item_sequence,
                    'order_groups': order_groups,  # Keep for pattern mining
                    'length': len(item_sequence)
                })
        
        logger.info(f"  ✅ Built {len(sequences):,} sequences")
        avg_length = np.mean([s['length'] for s in sequences]) if sequences else 0
        max_length = max([s['length'] for s in sequences]) if sequences else 0
        logger.info(f"  ✅ Avg sequence length: {avg_length:.1f} items")
        logger.info(f"  ✅ Max sequence length: {max_length} items")
        
        return sequences
    
    def mine_transition_patterns(self, sequences, min_support=2):
        """Mine both sequential and co-occurrence patterns"""
        logger.info(f"\nStep 3: Mining transition patterns (min_support={min_support}, smoothing={self.smoothing})...")
        logger.info(f"  Strategy: Hybrid (sequential between orders + co-occurrence within orders)")
        
        # Sequential transitions (between orders)
        sequential_transitions = defaultdict(lambda: defaultdict(int))
        
        # Co-occurrence patterns (within same order)
        cooccurrence_patterns = defaultdict(lambda: defaultdict(int))
        
        item_counts = Counter()
        
        for seq_data in sequences:
            order_groups = seq_data.get('order_groups', [])
            
            if not order_groups:
                # Fallback: use items sequence if order_groups not available
                seq = seq_data['items']
                for item in seq:
                    item_counts[item] += 1
                
                # Only learn transitions between different positions
                for i in range(len(seq) - 1):
                    current_item = seq[i]
                    next_item = seq[i + 1]
                    if current_item != next_item:
                        sequential_transitions[current_item][next_item] += 1
            else:
                # Count item frequencies
                for item in seq_data['items']:
                    item_counts[item] += 1
                
                # 1. Learn sequential transitions (between orders)
                for i in range(len(order_groups) - 1):
                    current_order_items = order_groups[i]['items']
                    next_order_items = order_groups[i + 1]['items']
                    
                    # Transition from last item of current order to first item of next order
                    last_item_current = current_order_items[-1]
                    first_item_next = next_order_items[0]
                    
                    if last_item_current != first_item_next:
                        sequential_transitions[last_item_current][first_item_next] += 1
                
                # 2. Learn co-occurrence patterns (within same order)
                for order_group in order_groups:
                    items_in_order = order_group['items']
                    
                    # All pairs of items in the same order
                    for i, item1 in enumerate(items_in_order):
                        for item2 in items_in_order[i+1:]:
                            if item1 != item2:
                                # Bidirectional: A co-occurs with B means B co-occurs with A
                                cooccurrence_patterns[item1][item2] += 1
                                cooccurrence_patterns[item2][item1] += 1
        
        # Combine both patterns
        transition_probs = {}
        vocab_size = len(item_counts)  # Number of unique items
        cooccurrence_weight = 0.5  # Co-occurrence has lower weight than sequential
        
        # Process sequential transitions (higher priority)
        sequential_count = 0
        for item, next_items in sequential_transitions.items():
            if item not in transition_probs:
                transition_probs[item] = {}
            
            total = sum(next_items.values())
            for next_item, count in next_items.items():
                if count >= min_support:
                    prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
                    transition_probs[item][next_item] = {
                        'probability': prob,
                        'count': count,
                        'confidence': count / total,
                        'type': 'sequential'
                    }
                    sequential_count += 1
        
        # Process co-occurrence patterns (with lower weight)
        cooccurrence_count = 0
        for item, co_items in cooccurrence_patterns.items():
            if item not in transition_probs:
                transition_probs[item] = {}
            
            total = sum(co_items.values())
            for co_item, count in co_items.items():
                if count >= min_support:
                    prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
                    prob_weighted = prob * cooccurrence_weight  # Lower weight
                    
                    # If already exists from sequential, combine (take max)
                    if co_item in transition_probs[item]:
                        # Sequential has priority, but boost if also co-occurs
                        existing_prob = transition_probs[item][co_item]['probability']
                        transition_probs[item][co_item]['probability'] = max(existing_prob, prob_weighted)
                        transition_probs[item][co_item]['type'] = 'both'
                    else:
                        transition_probs[item][co_item] = {
                            'probability': prob_weighted,
                            'count': count,
                            'confidence': count / total,
                            'type': 'cooccurrence'
                        }
                        cooccurrence_count += 1
        
        logger.info(f"  ✅ Found {len(transition_probs):,} items with transitions")
        total_patterns = sum(len(next_items) for next_items in transition_probs.values())
        logger.info(f"  ✅ Total transition patterns: {total_patterns:,}")
        logger.info(f"    - Sequential patterns: {sequential_count:,}")
        logger.info(f"    - Co-occurrence patterns: {cooccurrence_count:,}")
        
        # Show top patterns
        logger.info(f"\n  Top 10 strongest transitions:")
        all_transitions = []
        for item, next_items in transition_probs.items():
            for next_item, stats in next_items.items():
                all_transitions.append((item, next_item, stats['probability'], stats['count'], stats.get('type', 'unknown')))
        
        all_transitions.sort(key=lambda x: (x[3], x[2]), reverse=True)  # Sort by count, then probability
        
        for i, (item, next_item, prob, count, pattern_type) in enumerate(all_transitions[:10], 1):
            logger.info(f"    {i}. Item {item} → Item {next_item}: {prob:.2%} ({count} times, {pattern_type})")
        
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
        
        # Step: Temporal train/val/test split (70/15/15)
        logger.info(f"\nStep 2.5: Temporal train/val/test split (70/15/15)...")
        
        # Sort sequences by first order date (temporal split)
        sequences_with_dates = []
        for seq in sequences:
            if seq.get('order_groups'):
                first_order_date = seq['order_groups'][0]['created_at']
            else:
                # Fallback: use customer's first purchase date from df
                customer_data = df[df['customer_id'] == seq['customer_id']]
                if len(customer_data) > 0:
                    first_order_date = customer_data['created_at'].min()
                else:
                    first_order_date = datetime.now()
            sequences_with_dates.append((first_order_date, seq))
        
        # Sort by date
        sequences_with_dates.sort(key=lambda x: x[0])
        
        # Split indices: 70% train, 15% val, 15% test
        split1_idx = int(len(sequences_with_dates) * 0.7)
        split2_idx = int(len(sequences_with_dates) * 0.85)
        
        train_sequences = [seq for _, seq in sequences_with_dates[:split1_idx]]
        val_sequences = [seq for _, seq in sequences_with_dates[split1_idx:split2_idx]]
        test_sequences = [seq for _, seq in sequences_with_dates[split2_idx:]]
        
        logger.info(f"  ✅ Train: {len(train_sequences):,} sequences ({len(train_sequences)/len(sequences)*100:.1f}%)")
        logger.info(f"  ✅ Validation: {len(val_sequences):,} sequences ({len(val_sequences)/len(sequences)*100:.1f}%)")
        logger.info(f"  ✅ Test: {len(test_sequences):,} sequences ({len(test_sequences)/len(sequences)*100:.1f}%)")
        
        # Mine transition patterns from TRAIN set only
        logger.info(f"\nStep 3: Mining transition patterns from TRAIN set...")
        transition_probs, item_counts = self.mine_transition_patterns(train_sequences, min_support=self.min_support)
        
        # Mine category patterns from TRAIN set only
        train_df = df[df['customer_id'].isin([s['customer_id'] for s in train_sequences])]
        category_probs = self.mine_product_type_patterns(train_df)
        
        training_time = time.time() - start_time
        
        logger.info(f"\n✅ Next item prediction model trained in {training_time:.2f}s")
        logger.info(f"  ✅ Trained on {len(train_sequences):,} sequences")
        logger.info(f"  ✅ Validation set: {len(val_sequences):,} sequences")
        logger.info(f"  ✅ Test set: {len(test_sequences):,} sequences")
        
        # Prepare results
        result = {
            'transition_patterns': transition_probs,
            'category_patterns': category_probs,
            'item_counts': dict(item_counts),
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'metrics': {
                'num_customers': int(df['customer_id'].nunique()),
                'num_sequences': len(sequences),
                'num_train_sequences': len(train_sequences),
                'num_val_sequences': len(val_sequences),
                'num_test_sequences': len(test_sequences),
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
