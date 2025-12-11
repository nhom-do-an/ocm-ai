"""
Data Preprocessing Pipeline - Step 4
Split data into train/validation/test sets (temporal split)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Paths
PROCESSED_DIR = Path("d:/ai-feature/data/processed")
FEATURES_DIR = Path("d:/ai-feature/data/features")
SPLITS_DIR = Path("d:/ai-feature/data/splits")
REPORT_DIR = Path("d:/ai-feature/reports/data")

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load interactions and features"""
    print("\n📂 Loading data...")
    
    interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    
    user_features = pd.read_csv(FEATURES_DIR / "user_features.csv")
    item_features = pd.read_csv(FEATURES_DIR / "item_features.csv")
    
    print(f"  ✅ Interactions: {len(interactions):,}")
    print(f"  ✅ User features: {len(user_features):,}")
    print(f"  ✅ Item features: {len(item_features):,}")
    
    return interactions, user_features, item_features


def temporal_split(interactions, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data temporally to avoid data leakage
    
    Important for time-series data:
    - Training set: oldest data
    - Validation set: middle data
    - Test set: most recent data
    
    References:
    - Time-based CV: https://neptune.ai/blog/train-test-split-strategies-for-time-series-data
    """
    print("\n⏰ Performing temporal split...")
    print(f"  Ratios: Train {train_ratio}, Val {val_ratio}, Test {test_ratio}")
    
    # Sort by timestamp
    interactions = interactions.sort_values('timestamp').reset_index(drop=True)
    
    # Get time range
    min_time = interactions['timestamp'].min()
    max_time = interactions['timestamp'].max()
    duration = (max_time - min_time).total_seconds()
    
    print(f"\n  Time range: {min_time} to {max_time}")
    print(f"  Duration: {(max_time - min_time).days} days")
    
    # Calculate split points
    train_end_time = min_time + pd.Timedelta(seconds=duration * train_ratio)
    val_end_time = min_time + pd.Timedelta(seconds=duration * (train_ratio + val_ratio))
    
    print(f"\n  Train period: {min_time} to {train_end_time}")
    print(f"  Val period: {train_end_time} to {val_end_time}")
    print(f"  Test period: {val_end_time} to {max_time}")
    
    # Split data
    train_df = interactions[interactions['timestamp'] <= train_end_time].copy()
    val_df = interactions[(interactions['timestamp'] > train_end_time) & 
                         (interactions['timestamp'] <= val_end_time)].copy()
    test_df = interactions[interactions['timestamp'] > val_end_time].copy()
    
    print(f"\n  Train: {len(train_df):,} ({len(train_df)/len(interactions)*100:.1f}%)")
    print(f"  Val: {len(val_df):,} ({len(val_df)/len(interactions)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ({len(test_df)/len(interactions)*100:.1f}%)")
    
    return train_df, val_df, test_df, train_end_time, val_end_time


def analyze_split_quality(train_df, val_df, test_df):
    """
    Analyze split quality
    - Check for cold-start problems
    - Verify no data leakage
    - Check distribution consistency
    """
    print("\n🔍 Analyzing split quality...")
    
    analysis = {
        'train': {
            'num_interactions': len(train_df),
            'num_users': train_df['user_id'].nunique(),
            'num_items': train_df['item_id'].nunique(),
            'num_stores': train_df['store_id'].nunique(),
            'sparsity': 1 - (len(train_df) / (train_df['user_id'].nunique() * train_df['item_id'].nunique()))
        },
        'val': {
            'num_interactions': len(val_df),
            'num_users': val_df['user_id'].nunique(),
            'num_items': val_df['item_id'].nunique(),
            'num_stores': val_df['store_id'].nunique(),
        },
        'test': {
            'num_interactions': len(test_df),
            'num_users': test_df['user_id'].nunique(),
            'num_items': test_df['item_id'].nunique(),
            'num_stores': test_df['store_id'].nunique(),
        }
    }
    
    # Cold-start analysis
    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['item_id'].unique())
    
    val_users = set(val_df['user_id'].unique())
    val_items = set(val_df['item_id'].unique())
    
    test_users = set(test_df['user_id'].unique())
    test_items = set(test_df['item_id'].unique())
    
    analysis['cold_start'] = {
        'val_new_users': len(val_users - train_users),
        'val_new_items': len(val_items - train_items),
        'test_new_users': len(test_users - train_users),
        'test_new_items': len(test_items - train_items),
    }
    
    # User overlap
    analysis['overlap'] = {
        'train_val_users': len(train_users & val_users),
        'train_test_users': len(train_users & test_users),
        'val_test_users': len(val_users & test_users),
        'train_val_items': len(train_items & val_items),
        'train_test_items': len(train_items & test_items),
        'val_test_items': len(val_items & test_items),
    }
    
    # Print summary
    print("\n  📊 Split Statistics:")
    print(f"    Train: {analysis['train']['num_users']:,} users, {analysis['train']['num_items']:,} items")
    print(f"    Val:   {analysis['val']['num_users']:,} users, {analysis['val']['num_items']:,} items")
    print(f"    Test:  {analysis['test']['num_users']:,} users, {analysis['test']['num_items']:,} items")
    
    print(f"\n  🆕 Cold-Start Analysis:")
    print(f"    Val new users: {analysis['cold_start']['val_new_users']:,}")
    print(f"    Val new items: {analysis['cold_start']['val_new_items']:,}")
    print(f"    Test new users: {analysis['cold_start']['test_new_users']:,}")
    print(f"    Test new items: {analysis['cold_start']['test_new_items']:,}")
    
    if analysis['cold_start']['test_new_users'] > 0 or analysis['cold_start']['test_new_items'] > 0:
        print(f"\n    ⚠️  Cold-start problem exists in test set")
        print(f"    → Will need cold-start strategy (content-based, popularity fallback)")
    else:
        print(f"\n    ✅ No cold-start problem (all test users/items in train)")
    
    return analysis


def create_recbole_format(train_df, val_df, test_df, user_features, item_features):
    """
    Create RecBole format files (.inter, .user, .item)
    
    RecBole documentation: https://recbole.io/docs/user_guide/data/data_format.html
    """
    print("\n📦 Creating RecBole format...")
    
    recbole_dir = SPLITS_DIR / "recbole"
    recbole_dir.mkdir(exist_ok=True)
    
    # Interaction files - per store
    stores = train_df['store_id'].unique()
    
    for store_id in stores:
        store_dir = recbole_dir / f"store_{store_id}"
        store_dir.mkdir(exist_ok=True)
        
        # Filter by store
        store_train = train_df[train_df['store_id'] == store_id].copy()
        store_val = val_df[val_df['store_id'] == store_id].copy()
        store_test = test_df[test_df['store_id'] == store_id].copy()
        
        if len(store_train) == 0:
            continue
        
        # Prepare interaction format
        for split_name, split_df in [('train', store_train), ('val', store_val), ('test', store_test)]:
            inter_df = split_df[['user_id', 'item_id', 'timestamp_unix', 'weight']].copy()
            inter_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float', 'weight:float']
            
            output_path = store_dir / f"mydata.{split_name}.inter"
            inter_df.to_csv(output_path, sep='\t', index=False)
        
        # Combined .inter file (for full training)
        all_inter = pd.concat([store_train, store_val, store_test])
        all_inter = all_inter[['user_id', 'item_id', 'timestamp_unix', 'weight']].copy()
        all_inter.columns = ['user_id:token', 'item_id:token', 'timestamp:float', 'weight:float']
        all_inter.to_csv(store_dir / "mydata.inter", sep='\t', index=False)
    
    print(f"  ✅ Created RecBole format for {len(stores)} stores")
    
    # User features (global)
    user_recbole = user_features[['user_id', 'gender', 'age_group', 'rfm_segment']].copy()
    user_recbole.columns = ['user_id:token', 'gender:token', 'age_group:token', 'rfm_segment:token']
    user_recbole = user_recbole.fillna('unknown')
    user_recbole.to_csv(recbole_dir / "mydata.user", sep='\t', index=False)
    
    # Item features (global)
    item_recbole = item_features[['item_id', 'vendor', 'product_type', 'price_range']].copy()
    item_recbole.columns = ['item_id:token', 'vendor:token', 'product_type:token', 'price_range:token']
    item_recbole = item_recbole.fillna('unknown')
    item_recbole.to_csv(recbole_dir / "mydata.item", sep='\t', index=False)
    
    print(f"  ✅ Created user features: {len(user_recbole):,} users")
    print(f"  ✅ Created item features: {len(item_recbole):,} items")


def save_splits(train_df, val_df, test_df, user_features, item_features, analysis):
    """Save all split files"""
    print("\n💾 Saving split files...")
    
    # Save CSV splits
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    
    print(f"  ✅ {SPLITS_DIR / 'train.csv'}")
    print(f"  ✅ {SPLITS_DIR / 'val.csv'}")
    print(f"  ✅ {SPLITS_DIR / 'test.csv'}")
    
    # Save features (copy to splits dir for convenience)
    user_features.to_csv(SPLITS_DIR / "user_features.csv", index=False)
    item_features.to_csv(SPLITS_DIR / "item_features.csv", index=False)
    
    # Save analysis report
    report_path = REPORT_DIR / "data_split_analysis.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  ✅ Analysis report: {report_path}")


def main():
    """Main data splitting pipeline"""
    print("\n" + "="*60)
    print("🚀 DATA PREPROCESSING PIPELINE - STEP 4")
    print("   TEMPORAL DATA SPLITTING")
    print("="*60)
    
    # Load data
    interactions, user_features, item_features = load_data()
    
    # Temporal split
    train_df, val_df, test_df, train_end_time, val_end_time = temporal_split(interactions)
    
    # Analyze split quality
    analysis = analyze_split_quality(train_df, val_df, test_df)
    
    # Create RecBole format
    create_recbole_format(train_df, val_df, test_df, user_features, item_features)
    
    # Save splits
    save_splits(train_df, val_df, test_df, user_features, item_features, analysis)
    
    print("\n" + "="*60)
    print("✅ DATA SPLITTING COMPLETE")
    print("="*60)
    print(f"\nSplits saved to: {SPLITS_DIR}")
    print(f"  - train.csv: {len(train_df):,} interactions")
    print(f"  - val.csv: {len(val_df):,} interactions")
    print(f"  - test.csv: {len(test_df):,} interactions")
    print(f"  - RecBole format: {SPLITS_DIR / 'recbole'}")
    print("\n" + "="*60)
    print("🎉 DATA PREPROCESSING PIPELINE COMPLETE!")
    print("="*60)
    print("\n💡 Next steps:")
    print("  1. Train Recommendation Models: notebooks/recommendation/")
    print("  2. Train Trending Prediction: notebooks/forecasting/")
    print("  3. Customer Segmentation: notebooks/segmentation/")


if __name__ == "__main__":
    main()
