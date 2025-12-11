"""
Display all model metrics and configurations
"""
import torch
import pickle
import os

# Get the correct path
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

print('='*100)
print('RECOMMENDATION MODEL (NeuMF) - Store 1 - FULL DETAILS')
print('='*100)

# Load recommendation model
rec_path = os.path.join(results_dir, 'store_1', 'recommendation', 'neumf_model.pth')
rec = torch.load(rec_path, map_location='cpu', weights_only=False)

print('\n📊 TRAINING METRICS:')
print('-' * 100)
for k, v in sorted(rec['metrics'].items()):
    if isinstance(v, float):
        print(f'  {k:30s}: {v:.6f}')
    else:
        print(f'  {k:30s}: {v:,}' if isinstance(v, int) else f'  {k:30s}: {v}')

print('\n🏗️ MODEL ARCHITECTURE:')
print('-' * 100)
print(f'  Total Users (n_users):        {rec["n_users"]:,}')
print(f'  Total Items (n_items):        {rec["n_items"]:,}')
print(f'  Embedding Dimension:          32')
print(f'  MLP Hidden Layers:            [64, 32, 16]')
print(f'  Optimizer:                    Adam')
print(f'  Learning Rate:                0.001')
print(f'  Batch Size:                   256')
print(f'  Loss Function:                MSE (Mean Squared Error)')

print('\n📈 DATA STATISTICS:')
print('-' * 100)
sparsity = (1 - rec['metrics']['num_interactions'] / (rec['n_users'] * rec['n_items'])) * 100
print(f'  Matrix Size:                  {rec["n_users"]:,} x {rec["n_items"]:,} = {rec["n_users"] * rec["n_items"]:,} cells')
print(f'  Filled Interactions:          {rec["metrics"]["num_interactions"]:,}')
print(f'  Sparsity:                     {sparsity:.2f}%')
if 'num_order_interactions' in rec['metrics']:
    print(f'  Order Interactions:           {rec["metrics"]["num_order_interactions"]:,} (weight=1.0)')
if 'num_cart_interactions' in rec['metrics']:
    print(f'  Cart Interactions:            {rec["metrics"]["num_cart_interactions"]:,} (weight=0.3)')

print('\n⚙️ PREPROCESSING:')
print('-' * 100)
print(f'  Pipeline Version:             {rec.get("preprocessing_version", "N/A")}')
print(f'  Data Cleaning:                ✓ (status=completed/confirmed, paid orders only)')
print(f'  Interaction Types:            Orders (1.0) + Carts (0.3)')
print(f'  Temporal Split:               80% Train / 20% Test')

print('\n💾 MODEL SIZE:')
print('-' * 100)
model_path = os.path.join(results_dir, 'store_1', 'recommendation', 'neumf_model.pth')
model_size = os.path.getsize(model_path)
print(f'  File Size:                    {model_size:,} bytes ({model_size/1024:.2f} KB)')

print('\n' + '='*100)
print('TRENDING MODEL (LightGBM) - Store 1 - FULL DETAILS')
print('='*100)

# Load trending model
meta_path = os.path.join(results_dir, 'store_1', 'trending', 'metadata.pkl')
meta = pickle.load(open(meta_path, 'rb'))

print('\n📊 TRAINING METRICS:')
print('-' * 100)
for k, v in sorted(meta['metrics'].items()):
    if k == 'top_features':
        continue
    if isinstance(v, float):
        print(f'  {k:30s}: {v:.6f}')
    else:
        print(f'  {k:30s}: {v:,}' if isinstance(v, int) else f'  {k:30s}: {v}')

print('\n🏗️ MODEL CONFIGURATION:')
print('-' * 100)
print(f'  Algorithm:                    LightGBM (Gradient Boosting)')
print(f'  Objective:                    Regression')
print(f'  Boosting Type:                GBDT')
print(f'  Max Depth:                    6')
print(f'  Num Leaves:                   31')
print(f'  Learning Rate:                0.05')
print(f'  Feature Fraction:             0.9')
print(f'  Bagging Fraction:             0.8')
print(f'  Bagging Frequency:            5')
print(f'  Max Rounds:                   1000')
print(f'  Early Stopping:               50 rounds')
print(f'  Metrics:                      MAE, RMSE')

print('\n📈 DATA STATISTICS:')
print('-' * 100)
print(f'  Products:                     {meta["metrics"]["num_products"]:,}')
print(f'  Time Period (days):           {meta["metrics"]["num_days"]:,}')
print(f'  Total Records:                {meta["metrics"]["num_records"]:,}')
print(f'  Train/Test Split:             80% / 20%')
train_size = int(meta['metrics']['num_records'] * 0.8)
test_size = meta['metrics']['num_records'] - train_size
print(f'  Training Samples:             {train_size:,}')
print(f'  Test Samples:                 {test_size:,}')

print(f'\n🔧 FEATURES USED ({len(meta["feature_cols"])} total):')
print('-' * 100)
print('\n  Temporal Features:')
temporal = [f for f in meta['feature_cols'] if any(x in f for x in ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'weekend'])]
for f in temporal:
    print(f'    - {f}')

print('\n  Lag Features:')
lag = [f for f in meta['feature_cols'] if 'lag' in f]
for f in lag:
    print(f'    - {f}')

print('\n  Rolling Statistics:')
rolling = [f for f in meta['feature_cols'] if 'rolling' in f]
for f in rolling:
    print(f'    - {f}')

print('\n  Velocity & Growth:')
velocity = [f for f in meta['feature_cols'] if any(x in f for x in ['velocity', 'acceleration', 'growth'])]
for f in velocity:
    print(f'    - {f}')

print('\n  Other Metrics:')
other = [f for f in meta['feature_cols'] if f not in temporal + lag + rolling + velocity]
for f in other:
    print(f'    - {f}')

if 'top_features' in meta['metrics']:
    print('\n🌟 TOP 5 MOST IMPORTANT FEATURES:')
    print('-' * 100)
    for i, (feat, imp) in enumerate(sorted(meta['metrics']['top_features'].items(), key=lambda x: x[1], reverse=True), 1):
        print(f'  {i}. {feat:30s}: {imp:,.0f}')

print('\n⚙️ PREPROCESSING:')
print('-' * 100)
print(f'  Pipeline Version:             {meta.get("preprocessing_version", "N/A")}')
print(f'  Data Source:                  PostgreSQL (orders table)')
print(f'  Aggregation:                  Daily sales per product')
print(f'  Feature Engineering:          24 features (temporal, lag, rolling, velocity)')
print(f'  Missing Value Handling:       Fill with 0')

print('\n💾 MODEL SIZE:')
print('-' * 100)
lgb_model_path = os.path.join(results_dir, 'store_1', 'trending', 'lightgbm_model.txt')
lgb_meta_path = os.path.join(results_dir, 'store_1', 'trending', 'metadata.pkl')
model_size = os.path.getsize(lgb_model_path)
meta_size = os.path.getsize(lgb_meta_path)
print(f'  Model File:                   {model_size:,} bytes ({model_size/1024:.2f} KB)')
print(f'  Metadata File:                {meta_size:,} bytes ({meta_size/1024:.2f} KB)')
print(f'  Total Size:                   {model_size + meta_size:,} bytes ({(model_size + meta_size)/1024:.2f} KB)')

print('\n' + '='*100)
print('✅ ALL MODEL DETAILS EXTRACTED SUCCESSFULLY')
print('='*100)
