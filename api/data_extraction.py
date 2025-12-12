"""
Data extraction and preprocessing functions
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
from database import get_db_connection
from config import config

logger = logging.getLogger(__name__)


def extract_interactions(store_id):
    """
    Extract and preprocess user-item interactions from database
    Following the validated data pipeline:
    1. Data cleaning (remove invalid orders)
    2. Create interactions (orders + carts)
    3. Feature engineering
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        logger.info(f"Step 1: Extracting raw data for store {store_id}...")
        
        # Extract ORDER interactions (weight from config)
        cur.execute(f"""
            SELECT 
                o.customer_id as user_id,
                li.variant_id as item_id,
                v.product_id,
                o.store_id,
                COALESCE(o.completed_on, o.confirmed_on, o.created_at) as timestamp,
                li.quantity,
                li.price,
                'order' as action_type,
                {config.WEIGHT_ORDER} as weight,
                p.name as product_name,
                vn.name as vendor,
                pt.name as product_type
            FROM orders o
            JOIN line_items li ON li.reference_id = o.id AND li.reference_type = 'order'
            JOIN variants v ON v.id = li.variant_id
            JOIN products p ON p.id = v.product_id
            LEFT JOIN vendors vn ON vn.id = p.vendor_id
            LEFT JOIN product_types pt ON pt.id = p.product_type_id
            WHERE o.store_id = %s 
                AND o.status IN ('completed', 'confirmed')
                AND o.financial_status IN ('paid', 'partial_paid')
                AND o.customer_id IS NOT NULL
                AND li.variant_id IS NOT NULL
                AND o.deleted_at IS NULL
        """, (store_id,))
        
        order_interactions = cur.fetchall()
        
        # Extract CART interactions (weight from config)
        # Only get carts that haven't been checked out yet (no corresponding order)
        cur.execute(f"""
            SELECT 
                c.customer_id as user_id,
                li.variant_id as item_id,
                v.product_id,
                c.store_id,
                c.updated_at as timestamp,
                li.quantity,
                li.price,
                'cart' as action_type,
                {config.WEIGHT_CART} as weight,
                p.name as product_name,
                vn.name as vendor,
                pt.name as product_type
            FROM carts c
            JOIN line_items li ON li.reference_id = c.id AND li.reference_type = 'cart'
            JOIN variants v ON v.id = li.variant_id
            JOIN products p ON p.id = v.product_id
            LEFT JOIN vendors vn ON vn.id = p.vendor_id
            LEFT JOIN product_types pt ON pt.id = p.product_type_id
            WHERE c.store_id = %s
                AND c.customer_id IS NOT NULL
                AND li.variant_id IS NOT NULL
                AND c.deleted_at IS NULL
                AND NOT EXISTS (
                    SELECT 1 FROM orders o 
                    WHERE o.cart_token = c.token 
                    AND o.status IN ('completed', 'confirmed')
                )
        """, (store_id,))
        
        cart_interactions = cur.fetchall()
        
        # Combine interactions
        all_interactions = order_interactions + cart_interactions
        
        if len(all_interactions) == 0:
            raise ValueError(f"No interaction data found for store {store_id}")
        
        df = pd.DataFrame(all_interactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"  ✅ Order interactions: {len(order_interactions):,} (weight={config.WEIGHT_ORDER})")
        logger.info(f"  ✅ Cart interactions: {len(cart_interactions):,} (weight={config.WEIGHT_CART})")
        logger.info(f"  ✅ Total interactions: {len(df):,}")
        logger.info(f"  ✅ Unique users: {df['user_id'].nunique()}")
        logger.info(f"  ✅ Unique items: {df['item_id'].nunique()}")
        
        # Validate minimum requirements
        unique_users = df['user_id'].nunique()
        unique_items = df['item_id'].nunique()
        
        if unique_users < config.MIN_USERS_REQUIRED:
            raise ValueError(f"Insufficient users: need at least {config.MIN_USERS_REQUIRED}, got {unique_users}")
        
        if unique_items < config.MIN_ITEMS_REQUIRED:
            raise ValueError(f"Insufficient items: need at least {config.MIN_ITEMS_REQUIRED}, got {unique_items}")
        
        return df
        
    finally:
        cur.close()
        conn.close()


def extract_time_series_data(store_id, days_back=None):
    """
    Extract and engineer time series features for trending prediction
    Following the validated feature engineering pipeline
    """
    if days_back is None:
        days_back = config.TIME_SERIES_DAYS_BACK
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        start_date = datetime.now() - timedelta(days=days_back)
        
        logger.info(f"Step 1: Extracting raw time series data (last {days_back} days)...")
        
        cur.execute("""
            SELECT 
                v.product_id as item_id,
                li.variant_id,
                DATE(COALESCE(o.completed_on, o.confirmed_on, o.created_at)) as date,
                SUM(li.quantity) as daily_sales,
                SUM(li.quantity * li.price) as daily_revenue,
                COUNT(DISTINCT o.id) as order_count,
                AVG(li.price) as avg_price,
                p.name as product_name,
                vn.name as vendor,
                pt.name as product_type
            FROM orders o
            JOIN line_items li ON li.reference_id = o.id AND li.reference_type = 'order'
            JOIN variants v ON v.id = li.variant_id
            JOIN products p ON p.id = v.product_id
            LEFT JOIN vendors vn ON vn.id = p.vendor_id
            LEFT JOIN product_types pt ON pt.id = p.product_type_id
            WHERE o.store_id = %s 
                AND o.status IN ('completed', 'confirmed')
                AND o.financial_status IN ('paid', 'partial_paid')
                AND o.created_at >= %s
                AND li.variant_id IS NOT NULL
                AND o.deleted_at IS NULL
            GROUP BY v.product_id, li.variant_id, DATE(COALESCE(o.completed_on, o.confirmed_on, o.created_at)),
                     p.name, vn.name, pt.name
            ORDER BY v.product_id, date
        """, (store_id, start_date))
        
        data = cur.fetchall()
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise ValueError(f"No time series data found for store {store_id}")
        
        df['store_id'] = store_id
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"  ✅ Extracted {len(df):,} daily records")
        logger.info(f"  ✅ Products: {df['item_id'].nunique()}")
        logger.info(f"  ✅ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Feature Engineering
        logger.info(f"Step 2: Engineering time series features...")
        
        df = df.sort_values(['item_id', 'date'])
        
        # Temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 3, 7, 14]:
            df[f'sales_lag_{lag}'] = df.groupby('item_id')['daily_sales'].shift(lag)
        
        # Rolling statistics (Moving Averages)
        for window in [3, 7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df.groupby('item_id')['daily_sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Rolling standard deviation (volatility)
        for window in [7, 14]:
            df[f'sales_rolling_std_{window}'] = df.groupby('item_id')['daily_sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
        
        # Sales velocity (growth rate)
        df['sales_prev_7d'] = df.groupby('item_id')['daily_sales'].shift(7)
        df['sales_velocity'] = (
            (df['daily_sales'] - df['sales_prev_7d']) / 
            (df['sales_prev_7d'] + 1)
        )
        
        # Sales acceleration
        df['velocity_prev'] = df.groupby('item_id')['sales_velocity'].shift(1)
        df['sales_acceleration'] = df['sales_velocity'] - df['velocity_prev']
        
        # Exponentially weighted popularity
        df['popularity_score'] = df.groupby('item_id')['daily_sales'].transform(
            lambda x: x.ewm(alpha=0.1).mean()
        )
        
        # Week-over-week growth
        df['sales_prev_week'] = df.groupby('item_id')['daily_sales'].shift(7)
        df['week_over_week_growth'] = (
            (df['daily_sales'] - df['sales_prev_week']) / 
            (df['sales_prev_week'] + 1)
        )
        
        # Fill NaN values
        df = df.fillna(0)
        
        logger.info(f"  ✅ Features engineered:")
        logger.info(f"    - Temporal: day, week, month, quarter, weekend")
        logger.info(f"    - Lag features: 1, 3, 7, 14 days")
        logger.info(f"    - Moving averages: 3, 7, 14, 30 days")
        logger.info(f"    - Velocity & acceleration metrics")
        logger.info(f"    - Popularity score (exponentially weighted)")
        
        # Validate minimum data
        if len(df) < config.MIN_TIME_SERIES_RECORDS:
            raise ValueError(f"Insufficient time series data: need at least {config.MIN_TIME_SERIES_RECORDS} records, got {len(df)}")
        
        return df
        
    finally:
        cur.close()
        conn.close()
