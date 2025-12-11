"""
Database connection and utility functions
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from config import config

logger = logging.getLogger(__name__)


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(**config.DB_CONFIG)


def update_training_log(store_id, model_type, status, metrics=None, error=None):
    """Update training log in database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        if status == 'running':
            cur.execute("""
                UPDATE ai_training_log 
                SET status = %s, started_at = NOW()
                WHERE id = (
                    SELECT id FROM ai_training_log
                    WHERE store_id = %s AND model_type = %s AND status = 'pending'
                    ORDER BY created_at DESC LIMIT 1
                )
            """, (status, store_id, model_type))
        elif status in ['success', 'failed']:
            metrics_json = json.dumps(metrics) if metrics else None
            cur.execute("""
                UPDATE ai_training_log 
                SET status = %s, 
                    completed_at = NOW(), 
                    metrics = %s, 
                    error_message = %s,
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER
                WHERE id = (
                    SELECT id FROM ai_training_log
                    WHERE store_id = %s AND model_type = %s AND status = 'running'
                    ORDER BY created_at DESC LIMIT 1
                )
            """, (status, metrics_json, error, store_id, model_type))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update training log: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def update_store_ai_status(store_id, model_type, metrics):
    """Update store AI status after successful training"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        if model_type == 'recommendation':
            cur.execute("""
                UPDATE store_ai_status 
                SET recommendation_model_status = 'ready',
                    recommendation_model_version = 'v1.0',
                    recommendation_model_trained_at = NOW(),
                    recommendation_model_metrics = %s,
                    current_strategy = CASE 
                        WHEN trending_model_status = 'ready' THEN 'full_ai'
                        ELSE 'hybrid'
                    END,
                    updated_at = NOW()
                WHERE store_id = %s
            """, (json.dumps(metrics), store_id))
        elif model_type == 'trending':
            cur.execute("""
                UPDATE store_ai_status 
                SET trending_model_status = 'ready',
                    trending_model_version = 'v1.0',
                    trending_model_trained_at = NOW(),
                    trending_model_metrics = %s,
                    current_strategy = CASE 
                        WHEN recommendation_model_status = 'ready' THEN 'full_ai'
                        ELSE 'hybrid'
                    END,
                    updated_at = NOW()
                WHERE store_id = %s
            """, (json.dumps(metrics), store_id))
        
        conn.commit()
        logger.info(f"Updated AI status for store {store_id}, model: {model_type}")
    except Exception as e:
        logger.error(f"Failed to update store AI status: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
