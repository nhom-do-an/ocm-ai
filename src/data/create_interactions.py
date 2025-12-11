"""
Data Preprocessing Pipeline - Step 2
Create user-item interactions from orders and carts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Paths
CLEANED_DATA_DIR = Path("d:/ai-feature/data/processed/cleaned")
OUTPUT_DIR = Path("d:/ai-feature/data/processed")
REPORT_DIR = Path("d:/ai-feature/reports/data")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data():
    """Load cleaned data"""
    print("\n📂 Loading cleaned data...")
    
    tables = {}
    files = ['orders', 'line_items', 'carts', 'customers', 'products', 'variants',
             'collections', 'product_collections', 'vendors', 'product_types']
    
    for file in files:
        df = pd.read_csv(CLEANED_DATA_DIR / f"{file}.csv")
        
        # Convert datetime columns
        datetime_cols = [col for col in df.columns if 
                        any(x in col.lower() for x in ['_at', '_on', 'date'])]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        tables[file] = df
        print(f"  ✅ {file}: {len(df):,} rows")
    
    return tables


def create_order_interactions(tables):
    """
    Create interactions from orders
    Weight = 1.0 (strong signal - customer purchased)
    """
    print("\n🛍️ Creating order interactions...")
    
    orders = tables['orders']
    line_items = tables['line_items']
    variants = tables['variants']
    products = tables['products']
    
    # Filter valid orders
    # Status: completed OR confirmed
    # Financial status: paid OR partial_paid
    valid_orders = orders[
        (orders['status'].isin(['completed', 'confirmed'])) &
        (orders['financial_status'].isin(['paid', 'partial_paid']))
    ].copy()
    
    print(f"  Valid orders: {len(valid_orders):,} / {len(orders):,}")
    
    # Get line items for valid orders
    order_line_items = line_items[
        (line_items['reference_type'] == 'order') &
        (line_items['reference_id'].isin(valid_orders['id']))
    ].copy()
    
    print(f"  Order line items: {len(order_line_items):,}")
    
    # Join with orders to get customer_id, store_id, timestamp
    order_line_items = order_line_items.merge(
        valid_orders[['id', 'customer_id', 'store_id', 'completed_on', 'confirmed_on']],
        left_on='reference_id',
        right_on='id',
        how='left',
        suffixes=('', '_order')
    )
    
    # Use completed_on if available, else confirmed_on
    order_line_items['timestamp'] = order_line_items['completed_on'].fillna(
        order_line_items['confirmed_on']
    )
    
    # Join with variants to get product_id
    order_line_items = order_line_items.merge(
        variants[['id', 'product_id']],
        left_on='variant_id',
        right_on='id',
        how='left',
        suffixes=('', '_variant')
    )
    
    # Join with products to get product details
    order_line_items = order_line_items.merge(
        products[['id', 'name', 'vendor_id', 'product_type_id']],
        left_on='product_id',
        right_on='id',
        how='left',
        suffixes=('', '_product')
    )
    
    # Create interaction records
    order_interactions = pd.DataFrame({
        'user_id': order_line_items['customer_id'],
        'item_id': order_line_items['product_id'],
        'timestamp': order_line_items['timestamp'],
        'action_type': 'order',
        'weight': 1.0,
        'quantity': order_line_items['quantity'],
        'price': order_line_items['price'],
        'store_id': order_line_items['store_id'],
        'product_name': order_line_items['name'],
        'variant_title': order_line_items['variant_title']
    })
    
    # Remove rows with missing essential data
    order_interactions = order_interactions.dropna(subset=['user_id', 'item_id', 'timestamp'])
    
    print(f"  ✅ Created {len(order_interactions):,} order interactions")
    
    return order_interactions


def create_cart_interactions(tables):
    """
    Create interactions from carts
    Weight = 0.3 (weak signal - customer interested but not purchased)
    """
    print("\n🛒 Creating cart interactions...")
    
    carts = tables['carts']
    line_items = tables['line_items']
    variants = tables['variants']
    products = tables['products']
    
    # Filter active carts (not converted to orders)
    active_carts = carts[carts['status'] == 'active'].copy()
    
    print(f"  Active carts: {len(active_carts):,} / {len(carts):,}")
    
    # Get line items for active carts
    cart_line_items = line_items[
        (line_items['reference_type'] == 'cart') &
        (line_items['reference_id'].isin(active_carts['id']))
    ].copy()
    
    print(f"  Cart line items: {len(cart_line_items):,}")
    
    # Join with carts to get customer_id, store_id, timestamp
    cart_line_items = cart_line_items.merge(
        active_carts[['id', 'customer_id', 'store_id', 'updated_at']],
        left_on='reference_id',
        right_on='id',
        how='left',
        suffixes=('', '_cart')
    )
    
    cart_line_items['timestamp'] = cart_line_items['updated_at']
    
    # Join with variants to get product_id
    cart_line_items = cart_line_items.merge(
        variants[['id', 'product_id']],
        left_on='variant_id',
        right_on='id',
        how='left',
        suffixes=('', '_variant')
    )
    
    # Join with products to get product details
    cart_line_items = cart_line_items.merge(
        products[['id', 'name', 'vendor_id', 'product_type_id']],
        left_on='product_id',
        right_on='id',
        how='left',
        suffixes=('', '_product')
    )
    
    # Create interaction records
    cart_interactions = pd.DataFrame({
        'user_id': cart_line_items['customer_id'],
        'item_id': cart_line_items['product_id'],
        'timestamp': cart_line_items['timestamp'],
        'action_type': 'cart',
        'weight': 0.3,
        'quantity': cart_line_items['quantity'],
        'price': cart_line_items['price'],
        'store_id': cart_line_items['store_id'],
        'product_name': cart_line_items['name'],
        'variant_title': cart_line_items['variant_title']
    })
    
    # Remove rows with missing essential data
    cart_interactions = cart_interactions.dropna(subset=['user_id', 'item_id', 'timestamp'])
    
    print(f"  ✅ Created {len(cart_interactions):,} cart interactions")
    
    return cart_interactions


def enrich_interactions(interactions, tables):
    """Add additional product features to interactions"""
    print("\n🔗 Enriching interactions with product features...")
    
    products = tables['products']
    vendors = tables['vendors']
    product_types = tables['product_types']
    collections = tables['collections']
    product_collections = tables['product_collections']
    
    # Merge vendor names
    interactions = interactions.merge(
        products[['id', 'vendor_id']],
        left_on='item_id',
        right_on='id',
        how='left',
        suffixes=('', '_p')
    )
    
    interactions = interactions.merge(
        vendors[['id', 'name']].rename(columns={'name': 'vendor'}),
        left_on='vendor_id',
        right_on='id',
        how='left',
        suffixes=('', '_v')
    )
    
    # Merge product types
    interactions = interactions.merge(
        products[['id', 'product_type_id']],
        left_on='item_id',
        right_on='id',
        how='left',
        suffixes=('', '_pt1')
    )
    
    interactions = interactions.merge(
        product_types[['id', 'name']].rename(columns={'name': 'product_type'}),
        left_on='product_type_id',
        right_on='id',
        how='left',
        suffixes=('', '_pt2')
    )
    
    # Get first collection for each product (simplified)
    first_collections = product_collections.sort_values('position').groupby('product_id').first().reset_index()
    first_collections = first_collections.merge(
        collections[['id', 'name']].rename(columns={'name': 'collection_name'}),
        left_on='collection_id',
        right_on='id',
        how='left'
    )
    
    interactions = interactions.merge(
        first_collections[['product_id', 'collection_name']],
        left_on='item_id',
        right_on='product_id',
        how='left',
        suffixes=('', '_col')
    )
    
    # Keep only necessary columns
    final_cols = [
        'user_id', 'item_id', 'timestamp', 'action_type', 'weight',
        'quantity', 'price', 'store_id',
        'product_name', 'vendor', 'product_type', 'collection_name'
    ]
    
    interactions = interactions[[col for col in final_cols if col in interactions.columns]]
    
    # Fill missing categorical values
    interactions['vendor'] = interactions['vendor'].fillna('Unknown')
    interactions['product_type'] = interactions['product_type'].fillna('Unknown')
    interactions['collection_name'] = interactions['collection_name'].fillna('Unknown')
    
    print(f"  ✅ Enriched {len(interactions):,} interactions")
    
    return interactions


def analyze_interactions(interactions):
    """Analyze interaction data quality and statistics"""
    print("\n📊 Analyzing interactions...")
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_interactions': len(interactions),
        'unique_users': interactions['user_id'].nunique(),
        'unique_items': interactions['item_id'].nunique(),
        'unique_stores': interactions['store_id'].nunique(),
        'date_range': {
            'min': interactions['timestamp'].min().isoformat(),
            'max': interactions['timestamp'].max().isoformat(),
            'days': (interactions['timestamp'].max() - interactions['timestamp'].min()).days
        },
        'by_action_type': interactions['action_type'].value_counts().to_dict(),
        'by_store': interactions.groupby('store_id').size().describe().to_dict(),
        'sparsity': 1 - (len(interactions) / (interactions['user_id'].nunique() * interactions['item_id'].nunique())),
        'user_activity': {
            'mean': interactions.groupby('user_id').size().mean(),
            'median': interactions.groupby('user_id').size().median(),
            'min': interactions.groupby('user_id').size().min(),
            'max': interactions.groupby('user_id').size().max(),
        },
        'item_popularity': {
            'mean': interactions.groupby('item_id').size().mean(),
            'median': interactions.groupby('item_id').size().median(),
            'min': interactions.groupby('item_id').size().min(),
            'max': interactions.groupby('item_id').size().max(),
        }
    }
    
    # User activity distribution
    user_activity = interactions.groupby('user_id').size()
    stats['user_distribution'] = {
        'users_with_1_interaction': (user_activity == 1).sum(),
        'users_with_2-5_interactions': ((user_activity >= 2) & (user_activity <= 5)).sum(),
        'users_with_6-10_interactions': ((user_activity >= 6) & (user_activity <= 10)).sum(),
        'users_with_10+_interactions': (user_activity > 10).sum(),
    }
    
    # Item popularity distribution
    item_popularity = interactions.groupby('item_id').size().sort_values(ascending=False)
    stats['item_distribution'] = {
        'top_20_pct_items_interactions': item_popularity.head(int(len(item_popularity) * 0.2)).sum(),
        'total_interactions': len(interactions),
        'long_tail_percentage': 1 - (item_popularity.head(int(len(item_popularity) * 0.2)).sum() / len(interactions))
    }
    
    # Save report
    report_path = REPORT_DIR / "interactions_analysis.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Analysis saved: {report_path}")
    
    # Print key metrics
    print("\n" + "="*60)
    print("📈 INTERACTION STATISTICS")
    print("="*60)
    print(f"Total Interactions: {stats['total_interactions']:,}")
    print(f"Unique Users: {stats['unique_users']:,}")
    print(f"Unique Items: {stats['unique_items']:,}")
    print(f"Unique Stores: {stats['unique_stores']:,}")
    print(f"\nSparsity: {stats['sparsity']:.4%}")
    print(f"\nBy Action Type:")
    for action, count in stats['by_action_type'].items():
        print(f"  {action}: {count:,}")
    print(f"\nUser Activity (avg): {stats['user_activity']['mean']:.2f} interactions/user")
    print(f"Item Popularity (avg): {stats['item_popularity']['mean']:.2f} interactions/item")
    print(f"\nDate Range: {stats['date_range']['days']} days")
    print("="*60)
    
    return stats


def save_interactions(interactions):
    """Save interactions to CSV"""
    print("\n💾 Saving interactions...")
    
    # Convert timestamp to Unix timestamp for RecBole compatibility
    interactions['timestamp_unix'] = interactions['timestamp'].astype(np.int64) // 10**9
    
    output_path = OUTPUT_DIR / "interactions.csv"
    interactions.to_csv(output_path, index=False)
    
    print(f"  ✅ Saved: {output_path}")
    print(f"  Rows: {len(interactions):,}")
    print(f"  Columns: {len(interactions.columns)}")


def main():
    """Main interaction creation pipeline"""
    print("\n" + "="*60)
    print("🚀 DATA PREPROCESSING PIPELINE - STEP 2")
    print("   CREATE USER-ITEM INTERACTIONS")
    print("="*60)
    
    # Load data
    tables = load_cleaned_data()
    
    # Create order interactions
    order_interactions = create_order_interactions(tables)
    
    # Create cart interactions
    cart_interactions = create_cart_interactions(tables)
    
    # Combine interactions
    print("\n🔗 Combining interactions...")
    all_interactions = pd.concat([order_interactions, cart_interactions], ignore_index=True)
    print(f"  Total: {len(all_interactions):,} interactions")
    
    # Enrich with product features
    all_interactions = enrich_interactions(all_interactions, tables)
    
    # Sort by timestamp
    all_interactions = all_interactions.sort_values('timestamp').reset_index(drop=True)
    
    # Analyze
    stats = analyze_interactions(all_interactions)
    
    # Save
    save_interactions(all_interactions)
    
    print("\n" + "="*60)
    print("✅ INTERACTION CREATION COMPLETE")
    print("="*60)
    print(f"\nOutput: {OUTPUT_DIR / 'interactions.csv'}")
    print("\n💡 Next step: python src/data/feature_engineering.py")


if __name__ == "__main__":
    main()
