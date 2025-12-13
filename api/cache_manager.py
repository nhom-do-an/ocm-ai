"""
Cache management for recommendations and trending predictions
"""

import json
import logging
import torch
from datetime import datetime
from database import get_db_connection
from config import config

logger = logging.getLogger(__name__)


def cache_recommendations(store_id, model, train_dataset, all_users, device):
    """Pre-compute and cache recommendations for all users"""
    logger.info(f"\nCaching recommendations for store {store_id}...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get next version number
        cur.execute("""
            SELECT COALESCE(MAX(CAST(SUBSTRING(model_version FROM 2) AS INTEGER)), 0) + 1
            FROM recommendation_cache
            WHERE store_id = %s AND model_version ~ '^v[0-9]+$'
        """, (store_id,))
        next_version_num = cur.fetchone()[0]
        model_version = f'v{next_version_num}'
        logger.info(f"  📌 Model version: {model_version}")
        
        # Clear old cache before inserting new predictions
        logger.info(f"  🗑️  Clearing old recommendation cache...")
        cur.execute("""
            DELETE FROM recommendation_cache 
            WHERE store_id = %s
        """, (store_id,))
        deleted_count = cur.rowcount
        conn.commit()
        logger.info(f"  ✅ Deleted {deleted_count:,} old cache entries")
        
        logger.info(f"  Generating recommendations for {len(all_users):,} users...")
        
        model.eval()
        cached_count = 0
        
        with torch.no_grad():
            for user_id in all_users:
                user_id = int(user_id)
                
                if user_id not in train_dataset.user_to_idx:
                    continue
                
                user_idx = train_dataset.user_to_idx[user_id]
                
                # Generate predictions for all items
                predictions = []
                for item_id, item_idx in train_dataset.item_to_idx.items():
                    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(device)
                    score = model(user_tensor, item_tensor).cpu().item()
                    predictions.append({
                        'item_id': int(item_id),
                        'variant_id': int(item_id),
                        'score': float(score),
                    })
                
                # Sort by score and take top 100
                predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)[:100]
                for i, pred in enumerate(predictions):
                    pred['rank'] = i + 1
                
                # Save to cache
                cur.execute("""
                    INSERT INTO recommendation_cache 
                    (store_id, user_id, recommendations, model_version, 
                     strategy, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '%s days')
                    ON CONFLICT (store_id, user_id, model_version) 
                    DO UPDATE SET 
                        recommendations = EXCLUDED.recommendations,
                        strategy = EXCLUDED.strategy,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at
                """, (int(store_id), int(user_id), json.dumps(predictions), model_version, 'ai_model', config.CACHE_EXPIRY_DAYS))
                
                cached_count += 1
        
        conn.commit()
        logger.info(f"  ✅ Cached recommendations for {cached_count:,} users")
        logger.info(f"  📦 Model version saved: {model_version}")
        return cached_count, model_version
        
    except Exception as e:
        logger.error(f"  ❌ Failed to cache recommendations: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def cache_trending(store_id, model, time_series_df, feature_cols, use_existing_version=False):
    """Pre-compute and cache trending predictions"""
    logger.info(f"\nCaching trending predictions for store {store_id}...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        if use_existing_version:
            # Use existing model version (for daily cache updates without retraining)
            cur.execute("""
                SELECT model_version
                FROM trending_cache
                WHERE store_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (store_id,))
            result = cur.fetchone()
            if result:
                model_version = result[0]
                logger.info(f"  📌 Using existing model version: {model_version}")
            else:
                # Fallback: get next version if no cache exists
                cur.execute("""
                    SELECT COALESCE(MAX(CAST(SUBSTRING(model_version FROM 2) AS INTEGER)), 0) + 1
                    FROM trending_cache
                    WHERE store_id = %s AND model_version ~ '^v[0-9]+$'
                """, (store_id,))
                next_version_num = cur.fetchone()[0]
                model_version = f'v{next_version_num}'
                logger.info(f"  📌 New model version: {model_version}")
        else:
            # Get next version number (for new model training)
            cur.execute("""
                SELECT COALESCE(MAX(CAST(SUBSTRING(model_version FROM 2) AS INTEGER)), 0) + 1
                FROM trending_cache
                WHERE store_id = %s AND model_version ~ '^v[0-9]+$'
            """, (store_id,))
            next_version_num = cur.fetchone()[0]
            model_version = f'v{next_version_num}'
            logger.info(f"  📌 New model version: {model_version}")
        
        # Only clear cache for today's date (not all cache)
        today = datetime.now().date()
        logger.info(f"  🗑️  Clearing today's trending cache (date: {today})...")
        cur.execute("""
            DELETE FROM trending_cache 
            WHERE store_id = %s AND prediction_date = %s
        """, (store_id, today))
        deleted_count = cur.rowcount
        conn.commit()
        logger.info(f"  ✅ Deleted {deleted_count:,} cache entries for today")
        
        # Get latest data for each item
        latest_data = time_series_df.groupby('item_id').last().reset_index()
        
        if len(latest_data) == 0:
            logger.warning("  ⚠️  No data available for trending cache")
            return 0
        
        X_latest = latest_data[feature_cols].values
        predicted_sales = model.predict(X_latest, num_iteration=model.best_iteration)
        
        # Create predictions list
        predictions = []
        for idx, row in latest_data.iterrows():
            growth_rate = 0
            if 'sales_rolling_mean_7' in row and row['sales_rolling_mean_7'] > 0:
                growth_rate = ((predicted_sales[idx] - row['sales_rolling_mean_7']) / row['sales_rolling_mean_7']) * 100
            
            trend_score = predicted_sales[idx] / (row.get('sales_rolling_mean_7', 1) + 1e-10)
            
            predictions.append({
                'item_id': int(row['item_id']),
                'variant_id': int(row['variant_id']) if 'variant_id' in row else int(row['item_id']),
                'product_name': str(row.get('product_name', '')),
                'predicted_sales': float(predicted_sales[idx]),
                'trend_score': float(trend_score),
                'growth_rate': float(growth_rate)
            })
        
        # Sort by predicted sales
        predictions = sorted(predictions, key=lambda x: x['predicted_sales'], reverse=True)
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        # Save to cache
        today = datetime.now().date()
        cur.execute("""
            INSERT INTO trending_cache 
            (store_id, prediction_date, trending_products, model_version, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (store_id, prediction_date, model_version)
            DO UPDATE SET 
                trending_products = EXCLUDED.trending_products,
                created_at = EXCLUDED.created_at
        """, (int(store_id), today, json.dumps(predictions), model_version))
        
        conn.commit()
        logger.info(f"  ✅ Cached trending for {len(predictions):,} products")
        logger.info(f"  📦 Model version saved: {model_version}")
        return len(predictions), model_version
        
    except Exception as e:
        logger.error(f"  ❌ Failed to cache trending: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()
