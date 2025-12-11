"""
Trending prediction model trainer
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import time
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data_extraction import extract_time_series_data

logger = logging.getLogger(__name__)


class TrendingTrainer:
    """Trains LightGBM trending prediction model"""
    
    def __init__(self, store_id):
        self.store_id = store_id
        
    def train(self):
        """Train trending prediction model for the store"""
        logger.info(f"=" * 60)
        logger.info(f"TRENDING TRAINING - Store {self.store_id}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Step 1: Extract and engineer time series features
        time_series_df = extract_time_series_data(self.store_id)
        
        # Step 2: Prepare features for LightGBM
        logger.info(f"\nStep 3: Preparing features for LightGBM...")
        
        feature_cols = [
            # Temporal features
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'is_weekend',
            # Lag features
            'sales_lag_1', 'sales_lag_3', 'sales_lag_7', 'sales_lag_14',
            # Rolling statistics
            'sales_rolling_mean_3', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30',
            'sales_rolling_std_7', 'sales_rolling_std_14',
            # Velocity and acceleration
            'sales_velocity', 'sales_acceleration', 'week_over_week_growth',
            # Other metrics
            'popularity_score', 'order_count', 'avg_price'
        ]
        
        missing_cols = [col for col in feature_cols if col not in time_series_df.columns]
        if missing_cols:
            logger.warning(f"  ⚠️  Missing columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in time_series_df.columns]
        
        logger.info(f"  ✅ Selected {len(feature_cols)} features:")
        for i, col in enumerate(feature_cols, 1):
            logger.info(f"    {i}. {col}")
        
        X = time_series_df[feature_cols].values
        y = time_series_df['daily_sales'].values
        
        logger.info(f"\n  Dataset shape:")
        logger.info(f"    X: {X.shape}")
        logger.info(f"    y: {y.shape}")
        
        # Step 3: Train/Val/Test split (70/15/15) - temporal
        logger.info(f"\nStep 4: Temporal train/val/test split (70/15/15)...")
        
        # Split indices
        split1_idx = int(len(X) * 0.7)
        split2_idx = int(len(X) * 0.85)
        
        X_train = X[:split1_idx]
        X_val = X[split1_idx:split2_idx]
        X_test = X[split2_idx:]
        
        y_train = y[:split1_idx]
        y_val = y[split1_idx:split2_idx]
        y_test = y[split2_idx:]
        
        logger.info(f"  ✅ Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  ✅ Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  ✅ Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Step 4: Train LightGBM model with validation set
        logger.info(f"\nStep 5: Training LightGBM model...")
        logger.info(f"  Configuration:")
        logger.info(f"    - Objective: regression")
        logger.info(f"    - Metric: MAE, RMSE")
        logger.info(f"    - Max depth: {config.TREND_MAX_DEPTH}")
        logger.info(f"    - Learning rate: {config.TREND_LEARNING_RATE}")
        logger.info(f"    - Early stopping: {config.TREND_EARLY_STOPPING} rounds")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': ['mae', 'rmse'],
            'boosting_type': 'gbdt',
            'num_leaves': config.TREND_NUM_LEAVES,
            'max_depth': config.TREND_MAX_DEPTH,
            'learning_rate': config.TREND_LEARNING_RATE,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=config.TREND_MAX_ROUNDS,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.TREND_EARLY_STOPPING), 
                lgb.log_evaluation(period=100)
            ]
        )
        
        logger.info(f"\n  ✅ Training completed")
        logger.info(f"  ✅ Best iteration: {model.best_iteration}")
        
        # Step 5: Evaluate on validation set
        logger.info(f"\nStep 6: Evaluating on validation set...")
        
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        val_mae = np.mean(np.abs(y_val - y_val_pred))
        val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
        val_mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100
        
        logger.info(f"  Validation Metrics:")
        logger.info(f"    MAE: {val_mae:.4f}")
        logger.info(f"    RMSE: {val_rmse:.4f}")
        logger.info(f"    MAPE: {val_mape:.2f}%")
        
        # Step 6: Final evaluation on test set
        logger.info(f"\nStep 7: Final evaluation on test set...")
        
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        # Top-20 accuracy on test set
        test_start_idx = split2_idx
        test_df_with_pred = time_series_df.iloc[test_start_idx:].copy()
        test_df_with_pred['predicted_sales'] = y_pred
        
        latest_predictions = test_df_with_pred.groupby('item_id').last().reset_index()
        top_20_actual = set(latest_predictions.nlargest(20, 'daily_sales')['item_id'])
        top_20_predicted = set(latest_predictions.nlargest(20, 'predicted_sales')['item_id'])
        top_20_accuracy = len(top_20_actual & top_20_predicted) / 20
        
        training_time = time.time() - start_time
        
        logger.info(f"  Test Metrics:")
        logger.info(f"    MAE: {mae:.4f}")
        logger.info(f"    RMSE: {rmse:.4f}")
        logger.info(f"    MAPE: {mape:.2f}%")
        logger.info(f"    Top-20 Accuracy: {top_20_accuracy:.2f}")
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'top_20_accuracy': float(top_20_accuracy),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_mape': float(val_mape),
            'training_time': int(training_time),
            'num_products': int(time_series_df['item_id'].nunique()),
            'num_days': int((time_series_df['date'].max() - time_series_df['date'].min()).days),
            'num_records': int(len(time_series_df)),
            'best_iteration': int(model.best_iteration),
            'num_features': len(feature_cols),
            'preprocessing_pipeline': 'validated'
        }
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importance().tolist()))
        top_5_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"\n  Top 5 Important Features:")
        for i, (feat, imp) in enumerate(top_5_features, 1):
            logger.info(f"    {i}. {feat}: {imp:,.0f}")
        
        metrics['top_features'] = dict(top_5_features)
        
        # Step 7: Save predictions to trending_accuracy table
        logger.info(f"\nStep 7: Saving predictions to trending_accuracy...")
        self._save_predictions_to_accuracy_table(
            time_series_df, 
            model, 
            feature_cols,
            y_test,
            y_pred
        )
        
        # Prepare metadata
        metadata = {
            'feature_cols': feature_cols,
            'metrics': metrics,
            'preprocessing_version': 'v2.0_validated'
        }
        
        return {
            'model': model,
            'metadata': metadata,
            'time_series_df': time_series_df,
            'metrics': metrics
        }
    
    def _save_predictions_to_accuracy_table(self, time_series_df, model, feature_cols, y_test, y_pred):
        """Save predictions to trending_accuracy for later comparison with actual sales"""
        from database import get_db_connection
        from datetime import datetime, timedelta
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Get latest data for each item to predict next 7 days
            latest_data = time_series_df.groupby('item_id').last().reset_index()
            X_latest = latest_data[feature_cols].values
            predicted_sales = model.predict(X_latest, num_iteration=model.best_iteration)
            
            prediction_date = datetime.now().date()
            
            # Insert predictions
            records_inserted = 0
            for idx, row in latest_data.iterrows():
                cur.execute("""
                    INSERT INTO trending_accuracy 
                    (store_id, prediction_date, item_id, variant_id, predicted_sales, actual_sales, absolute_error)
                    VALUES (%s, %s, %s, %s, %s, NULL, NULL)
                    ON CONFLICT DO NOTHING
                """, (
                    self.store_id,
                    prediction_date,
                    int(row['item_id']),
                    int(row.get('variant_id', row['item_id'])),
                    float(predicted_sales[idx])
                ))
                records_inserted += cur.rowcount
            
            conn.commit()
            logger.info(f"  ✅ Saved {records_inserted} predictions to trending_accuracy")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to save predictions: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

