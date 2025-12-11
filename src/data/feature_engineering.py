"""
Data Preprocessing Pipeline - Step 3
Feature Engineering for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Paths
CLEANED_DATA_DIR = Path("d:/ai-feature/data/processed/cleaned")
PROCESSED_DIR = Path("d:/ai-feature/data/processed")
FEATURES_DIR = Path("d:/ai-feature/data/features")
REPORT_DIR = Path("d:/ai-feature/reports/data")

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load necessary data"""
    print("\n📂 Loading data...")
    
    interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    
    customers = pd.read_csv(CLEANED_DATA_DIR / "customers.csv")
    customers['created_at'] = pd.to_datetime(customers['created_at'])
    customers['dob'] = pd.to_datetime(customers['dob'], errors='coerce')
    
    products = pd.read_csv(CLEANED_DATA_DIR / "products.csv")
    variants = pd.read_csv(CLEANED_DATA_DIR / "variants.csv")
    orders = pd.read_csv(CLEANED_DATA_DIR / "orders.csv")
    
    print(f"  ✅ Interactions: {len(interactions):,}")
    print(f"  ✅ Customers: {len(customers):,}")
    print(f"  ✅ Products: {len(products):,}")
    
    return interactions, customers, products, variants, orders


def extract_user_features(interactions, customers):
    """
    Extract user features including RFM analysis
    
    References:
    - RFM: https://en.wikipedia.org/wiki/RFM_(market_research)
    - Customer Segmentation: Fader & Hardie (2009)
    """
    print("\n👤 Extracting user features...")
    
    # Calculate age from dob
    current_date = datetime.now()
    customers['age'] = (current_date - customers['dob']).dt.days / 365.25
    customers['age'] = customers['age'].fillna(-1).astype(int)
    
    # Age groups
    customers['age_group'] = pd.cut(
        customers['age'],
        bins=[-np.inf, 0, 18, 25, 35, 45, 55, 100],
        labels=['unknown', 'under_18', '18-25', '26-35', '36-45', '46-55', '55+']
    )
    
    # Customer tenure
    customers['customer_tenure_days'] = (current_date - customers['created_at']).dt.days
    
    print("  ✅ Basic demographics extracted")
    
    # RFM Analysis - only from 'order' interactions
    order_interactions = interactions[interactions['action_type'] == 'order'].copy()
    
    # Calculate reference date (last interaction date + 1 day)
    reference_date = order_interactions['timestamp'].max() + timedelta(days=1)
    
    # Group by user
    rfm = order_interactions.groupby('user_id').agg({
        'timestamp': lambda x: (reference_date - x.max()).days,  # Recency
        'item_id': 'count',  # Frequency
        'price': lambda x: (x * order_interactions.loc[x.index, 'quantity']).sum()  # Monetary
    }).reset_index()
    
    rfm.columns = ['user_id', 'recency_days', 'frequency', 'monetary']
    
    # Calculate quantiles for RFM scoring (1-5 scale)
    rfm['r_score'] = pd.qcut(rfm['recency_days'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # RFM Score (concatenated)
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    # RFM Segmentation
    def rfm_segment(row):
        r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        elif r >= 4 and f <= 1:
            return 'New Customers'
        else:
            return 'Need Attention'
    
    rfm['rfm_segment'] = rfm.apply(rfm_segment, axis=1)
    
    print("  ✅ RFM analysis completed")
    
    # Additional user metrics
    user_stats = interactions.groupby('user_id').agg({
        'timestamp': ['min', 'max', 'count'],
        'price': lambda x: (x * interactions.loc[x.index, 'quantity']).sum(),
        'store_id': 'nunique'
    }).reset_index()
    
    user_stats.columns = ['user_id', 'first_interaction', 'last_interaction', 
                          'total_interactions', 'total_spent', 'num_stores']
    
    user_stats['avg_order_value'] = user_stats['total_spent'] / user_stats['total_interactions']
    user_stats['active_days'] = (user_stats['last_interaction'] - user_stats['first_interaction']).dt.days + 1
    
    # Merge all user features
    user_features = customers[['id', 'store_id', 'gender', 'age', 'age_group', 'customer_tenure_days']].copy()
    user_features = user_features.rename(columns={'id': 'user_id'})
    
    user_features = user_features.merge(rfm, on='user_id', how='left')
    user_features = user_features.merge(user_stats, on='user_id', how='left')
    
    # Fill missing values for users without orders
    user_features['recency_days'] = user_features['recency_days'].fillna(999)
    user_features['frequency'] = user_features['frequency'].fillna(0)
    user_features['monetary'] = user_features['monetary'].fillna(0)
    user_features['rfm_segment'] = user_features['rfm_segment'].fillna('No Purchase')
    user_features['total_interactions'] = user_features['total_interactions'].fillna(0)
    
    print(f"  ✅ Created features for {len(user_features):,} users")
    print(f"\n  RFM Segments:")
    for segment, count in user_features['rfm_segment'].value_counts().items():
        print(f"    {segment}: {count:,} ({count/len(user_features)*100:.1f}%)")
    
    return user_features


def extract_item_features(interactions, products, variants):
    """Extract item (product) features"""
    print("\n📦 Extracting item features...")
    
    # Product basic info
    item_features = products[['id', 'name', 'vendor_id', 'product_type_id', 'store_id', 'status']].copy()
    item_features = item_features.rename(columns={'id': 'item_id', 'name': 'product_name'})
    
    # Variant info - get price statistics
    variant_stats = variants.groupby('product_id').agg({
        'price': ['mean', 'min', 'max', 'count'],
        'id': 'count'
    }).reset_index()
    
    variant_stats.columns = ['item_id', 'avg_price', 'min_price', 'max_price', 
                              'price_count', 'num_variants']
    
    # Price range category
    variant_stats['price_range'] = pd.cut(
        variant_stats['avg_price'],
        bins=[0, 100000, 500000, 2000000, np.inf],
        labels=['budget', 'mid', 'premium', 'luxury']
    )
    
    item_features = item_features.merge(variant_stats, on='item_id', how='left')
    
    # Sales statistics from interactions
    item_sales = interactions[interactions['action_type'] == 'order'].groupby('item_id').agg({
        'user_id': 'count',  # Number of purchases
        'quantity': 'sum',   # Total quantity sold
        'price': lambda x: (x * interactions.loc[x.index, 'quantity']).sum(),  # Total revenue
        'timestamp': ['min', 'max']
    }).reset_index()
    
    item_sales.columns = ['item_id', 'num_purchases', 'total_sold', 'revenue_generated',
                          'first_sale', 'last_sale']
    
    item_sales['days_on_sale'] = (item_sales['last_sale'] - item_sales['first_sale']).dt.days + 1
    item_sales['avg_daily_sales'] = item_sales['total_sold'] / item_sales['days_on_sale']
    
    # Popularity score (log-scaled to handle outliers)
    item_sales['popularity_score'] = np.log1p(item_sales['num_purchases'])
    
    item_features = item_features.merge(item_sales, on='item_id', how='left')
    
    # Fill missing values for items without sales
    item_features['num_purchases'] = item_features['num_purchases'].fillna(0)
    item_features['total_sold'] = item_features['total_sold'].fillna(0)
    item_features['revenue_generated'] = item_features['revenue_generated'].fillna(0)
    item_features['popularity_score'] = item_features['popularity_score'].fillna(0)
    
    # Add product features from interactions (vendor, type, collection)
    product_attrs = interactions.groupby('item_id')[['vendor', 'product_type', 'collection_name']].first().reset_index()
    item_features = item_features.merge(product_attrs, on='item_id', how='left')
    
    print(f"  ✅ Created features for {len(item_features):,} items")
    print(f"\n  Price Range Distribution:")
    for range_cat, count in item_features['price_range'].value_counts().items():
        print(f"    {range_cat}: {count:,}")
    
    return item_features


def extract_time_series_features(interactions):
    """
    Extract time series features for trending prediction
    
    References:
    - Time Series Feature Engineering: Hyndman & Athanasopoulos (2021)
    - Trending Algorithms: Reddit Hot Ranking, HN Algorithm
    """
    print("\n📈 Extracting time series features...")
    
    # Focus on orders only
    orders = interactions[interactions['action_type'] == 'order'].copy()
    
    # Create date column
    orders['date'] = orders['timestamp'].dt.date
    
    # Daily aggregation
    daily_sales = orders.groupby(['date', 'item_id', 'store_id']).agg({
        'user_id': 'count',  # Number of orders
        'quantity': 'sum',   # Total quantity
        'price': lambda x: (x * orders.loc[x.index, 'quantity']).sum()  # Revenue
    }).reset_index()
    
    daily_sales.columns = ['date', 'item_id', 'store_id', 'daily_orders', 
                           'daily_sales', 'daily_revenue']
    
    print(f"  ✅ Created daily aggregation: {len(daily_sales):,} records")
    
    # Sort by date
    daily_sales = daily_sales.sort_values(['item_id', 'date'])
    
    # Moving averages (7, 14, 30 days)
    for window in [7, 14, 30]:
        daily_sales[f'sales_ma_{window}'] = daily_sales.groupby('item_id')['daily_sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Sales velocity (growth rate)
    daily_sales['sales_prev_7d'] = daily_sales.groupby('item_id')['daily_sales'].shift(7)
    daily_sales['sales_velocity'] = (
        (daily_sales['daily_sales'] - daily_sales['sales_prev_7d']) / 
        (daily_sales['sales_prev_7d'] + 1)  # Add 1 to avoid division by zero
    )
    
    # Sales acceleration (change in velocity)
    daily_sales['velocity_prev'] = daily_sales.groupby('item_id')['sales_velocity'].shift(1)
    daily_sales['sales_acceleration'] = daily_sales['sales_velocity'] - daily_sales['velocity_prev']
    
    # Exponentially weighted popularity score (decay factor = 0.9)
    daily_sales['popularity_score'] = daily_sales.groupby('item_id')['daily_sales'].transform(
        lambda x: x.ewm(alpha=0.1).mean()
    )
    
    # Week-over-week growth
    daily_sales['sales_prev_week'] = daily_sales.groupby('item_id')['daily_sales'].shift(7)
    daily_sales['week_over_week_growth'] = (
        (daily_sales['daily_sales'] - daily_sales['sales_prev_week']) / 
        (daily_sales['sales_prev_week'] + 1)
    )
    
    # Temporal features
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['day_of_month'] = daily_sales['date'].dt.day
    daily_sales['week_of_year'] = daily_sales['date'].dt.isocalendar().week
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['quarter'] = daily_sales['date'].dt.quarter
    daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
    
    # Season
    daily_sales['season'] = daily_sales['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    
    print("  ✅ Created time series features:")
    print(f"    - Moving averages (7, 14, 30 days)")
    print(f"    - Sales velocity & acceleration")
    print(f"    - Exponentially weighted popularity")
    print(f"    - Temporal features (day, week, month, season)")
    
    return daily_sales


def calculate_trending_score(time_series_features):
    """
    Calculate trending score for products
    
    Formula:
    trend_score = 0.4 * sales_velocity + 0.3 * popularity_score_norm + 
                  0.2 * sales_acceleration + 0.1 * week_over_week_growth
    """
    print("\n🔥 Calculating trending scores...")
    
    df = time_series_features.copy()
    
    # Normalize features to 0-1 range
    for col in ['sales_velocity', 'popularity_score', 'sales_acceleration', 'week_over_week_growth']:
        if col in df.columns:
            df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    
    # Calculate trend score
    df['trend_score'] = (
        0.4 * df['sales_velocity_norm'].fillna(0) +
        0.3 * df['popularity_score_norm'].fillna(0) +
        0.2 * df['sales_acceleration_norm'].fillna(0) +
        0.1 * df['week_over_week_growth_norm'].fillna(0)
    )
    
    print(f"  ✅ Trend scores calculated")
    print(f"  Range: [{df['trend_score'].min():.4f}, {df['trend_score'].max():.4f}]")
    
    return df


def save_features(user_features, item_features, time_series_features):
    """Save all feature sets"""
    print("\n💾 Saving features...")
    
    # User features
    user_path = FEATURES_DIR / "user_features.csv"
    user_features.to_csv(user_path, index=False)
    print(f"  ✅ {user_path}")
    
    # Item features
    item_path = FEATURES_DIR / "item_features.csv"
    item_features.to_csv(item_path, index=False)
    print(f"  ✅ {item_path}")
    
    # Time series features
    ts_path = FEATURES_DIR / "time_series_features.csv"
    time_series_features.to_csv(ts_path, index=False)
    print(f"  ✅ {ts_path}")
    
    # Generate summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'user_features': {
            'num_users': len(user_features),
            'num_features': len(user_features.columns),
            'features': list(user_features.columns)
        },
        'item_features': {
            'num_items': len(item_features),
            'num_features': len(item_features.columns),
            'features': list(item_features.columns)
        },
        'time_series_features': {
            'num_records': len(time_series_features),
            'num_features': len(time_series_features.columns),
            'date_range': {
                'min': time_series_features['date'].min().isoformat(),
                'max': time_series_features['date'].max().isoformat()
            },
            'features': list(time_series_features.columns)
        }
    }
    
    summary_path = REPORT_DIR / "feature_engineering_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  ✅ Summary: {summary_path}")


def main():
    """Main feature engineering pipeline"""
    print("\n" + "="*60)
    print("🚀 DATA PREPROCESSING PIPELINE - STEP 3")
    print("   FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    interactions, customers, products, variants, orders = load_data()
    
    # Extract user features
    user_features = extract_user_features(interactions, customers)
    
    # Extract item features
    item_features = extract_item_features(interactions, products, variants)
    
    # Extract time series features
    time_series_features = extract_time_series_features(interactions)
    
    # Calculate trending scores
    time_series_features = calculate_trending_score(time_series_features)
    
    # Save all features
    save_features(user_features, item_features, time_series_features)
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nFeatures saved to: {FEATURES_DIR}")
    print(f"  - User features: {len(user_features):,} users")
    print(f"  - Item features: {len(item_features):,} items")
    print(f"  - Time series: {len(time_series_features):,} records")
    print("\n💡 Next step: python src/data/split_data.py")


if __name__ == "__main__":
    main()
