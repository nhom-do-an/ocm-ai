"""
Python Training Service for AI Models
Exposes Flask API endpoints to train recommendation and trending models
Refactored version with separated concerns and environment configuration
"""

from flask import Flask, request, jsonify
import torch
import pickle
import traceback
import logging

# Import modules
from config import config
from database import update_training_log, update_store_ai_status
from trainers import RecommendationTrainer, TrendingTrainer
from trainers.next_item import NextItemTrainer
from cache_manager import cache_recommendations, cache_trending

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_recommendation_model(store_id, model_data):
    """Save recommendation model to disk"""
    logger.info(f"\nSaving recommendation model for store {store_id}...")
    
    model_dir = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'recommendation'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'neumf_model.pth'
    
    torch.save(model_data, model_path)
    
    logger.info(f"  ✅ Model saved to {model_path}")
    return model_path


def save_trending_model(store_id, model, metadata):
    """Save trending model to disk"""
    logger.info(f"\nSaving trending model for store {store_id}...")
    
    model_dir = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'trending'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'lightgbm_model.txt'
    metadata_path = model_dir / 'metadata.pkl'
    
    model.save_model(str(model_path))
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"  ✅ Model saved to {model_path}")
    logger.info(f"  ✅ Metadata saved to {metadata_path}")
    
    return model_path


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ai-training-service',
        'version': '2.0',
        'config_loaded': True
    }), 200


@app.route('/train/recommendation', methods=['POST'])
def train_recommendation():
    """Train recommendation model for a store using validated preprocessing pipeline"""
    data = request.json
    store_id = data.get('store_id')
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RECOMMENDATION TRAINING REQUEST - Store {store_id}")
    logger.info(f"{'='*80}")
    
    try:
        # Update training log to running
        update_training_log(store_id, 'recommendation', 'running')
        
        # Train model
        trainer = RecommendationTrainer(store_id)
        result = trainer.train()
        
        # Save model
        model_path = save_recommendation_model(store_id, result['model_data'])
        
        # Cache recommendations
        cached_count, model_version = cache_recommendations(
            store_id, 
            result['model'], 
            result['train_dataset'], 
            result['all_users'],
            trainer.device
        )
        
        result['metrics']['cached_users'] = cached_count
        result['metrics']['model_version'] = model_version
        
        # Update database
        update_store_ai_status(store_id, 'recommendation', result['metrics'])
        update_training_log(store_id, 'recommendation', 'success', result['metrics'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RECOMMENDATION TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}\n")
        
        return jsonify({
            'success': True,
            'store_id': store_id,
            'model_type': 'recommendation',
            'metrics': result['metrics'],
            'model_path': str(model_path),
            'preprocessing': 'validated_pipeline_v2.0'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Training failed for store {store_id}: {str(e)}")
        logger.error(traceback.format_exc())
        update_training_log(store_id, 'recommendation', 'failed', error=str(e))
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/train/trending', methods=['POST'])
def train_trending():
    """Train trending prediction model for a store using validated preprocessing pipeline"""
    data = request.json
    store_id = data.get('store_id')
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRENDING TRAINING REQUEST - Store {store_id}")
    logger.info(f"{'='*80}")
    
    try:
        # Update training log to running
        update_training_log(store_id, 'trending', 'running')
        
        # Train model
        trainer = TrendingTrainer(store_id)
        result = trainer.train()
        
        # Save model
        model_path = save_trending_model(store_id, result['model'], result['metadata'])
        
        # Cache trending predictions
        cached_count, model_version = cache_trending(
            store_id,
            result['model'],
            result['time_series_df'],
            result['metadata']['feature_cols']
        )
        
        result['metrics']['cached_products'] = cached_count
        result['metrics']['model_version'] = model_version
        
        # Update database
        update_store_ai_status(store_id, 'trending', result['metrics'])
        update_training_log(store_id, 'trending', 'success', result['metrics'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRENDING TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}\n")
        
        return jsonify({
            'success': True,
            'store_id': store_id,
            'model_type': 'trending',
            'metrics': result['metrics'],
            'model_path': str(model_path),
            'preprocessing': 'validated_pipeline_v2.0'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Training failed for store {store_id}: {str(e)}")
        logger.error(traceback.format_exc())
        update_training_log(store_id, 'trending', 'failed', error=str(e))
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/predict/recommendations', methods=['POST'])
def predict_recommendations():
    """Generate recommendations for a user (inference endpoint)"""
    data = request.json
    store_id = data.get('store_id')
    user_id = data.get('user_id')
    n = data.get('n', 10)
    
    if not store_id or not user_id:
        return jsonify({'error': 'store_id and user_id required'}), 400
    
    try:
        from src.models.recommendation.train_neumf import NeuMF
        
        # Load trained model
        model_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'recommendation' / 'neumf_model.pth'
        
        if not model_path.exists():
            return jsonify({'error': f'Model not found for store {store_id}. Please train first.'}), 404
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Reconstruct model
        device = torch.device('cpu')
        model = NeuMF(
            n_users=checkpoint['n_users'],
            n_items=checkpoint['n_items'],
            embed_dim=config.REC_EMBED_DIM,
            hidden_layers=config.REC_HIDDEN_LAYERS
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get user index
        user_to_idx = checkpoint['user_to_idx']
        item_to_idx = checkpoint['item_to_idx']
        
        if user_id not in user_to_idx:
            return jsonify({'error': f'User {user_id} not in training data'}), 404
        
        user_idx = user_to_idx[user_id]
        
        # Generate predictions for all items
        predictions = []
        with torch.no_grad():
            for item_id, item_idx in item_to_idx.items():
                user_tensor = torch.tensor([user_idx], dtype=torch.long)
                item_tensor = torch.tensor([item_idx], dtype=torch.long)
                score = model(user_tensor, item_tensor).item()
                predictions.append({
                    'item_id': int(item_id),
                    'variant_id': int(item_id),
                    'score': float(score),
                })
        
        # Sort by score and take top N
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)[:n]
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        return jsonify({
            'store_id': store_id,
            'user_id': user_id,
            'recommendations': predictions
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/trending', methods=['POST'])
def predict_trending():
    """Generate trending predictions (inference endpoint)"""
    data = request.json
    store_id = data.get('store_id')
    n = data.get('n', 20)
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    try:
        import lightgbm as lgb
        from data_extraction import extract_time_series_data
        
        # Load trained model
        model_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'trending' / 'lightgbm_model.txt'
        metadata_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'trending' / 'metadata.pkl'
        
        if not model_path.exists():
            return jsonify({'error': f'Model not found for store {store_id}. Please train first.'}), 404
        
        model = lgb.Booster(model_file=str(model_path))
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract recent data for each item
        time_series_df = extract_time_series_data(store_id, days_back=30)
        
        # Get latest record for each item
        latest_data = time_series_df.groupby('item_id').last().reset_index()
        latest_data = latest_data.dropna()
        
        # Predict
        feature_cols = metadata['feature_cols']
        X = latest_data[feature_cols].values
        predicted_sales = model.predict(X)
        
        # Create predictions list
        predictions = []
        for idx, row in latest_data.iterrows():
            predictions.append({
                'item_id': int(row['item_id']),
                'variant_id': int(row['variant_id']) if 'variant_id' in row else int(row['item_id']),
                'predicted_sales': float(predicted_sales[idx]),
                'trend_score': float(predicted_sales[idx] / (row.get('sales_rolling_mean_7', 1) + 1e-10))
            })
        
        # Sort by predicted sales and take top N
        predictions = sorted(predictions, key=lambda x: x['predicted_sales'], reverse=True)[:n]
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        return jsonify({
            'store_id': store_id,
            'trending': predictions
        }), 200
        
    except Exception as e:
        logger.error(f"Trending prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/train/next-item', methods=['POST'])
@app.route('/train/next_item', methods=['POST'])
def train_next_item():
    """Train next item prediction model (Sequential patterns)"""
    data = request.json
    store_id = data.get('store_id')
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    logger.info(f"\n{'='*80}")
    logger.info(f"NEXT ITEM PREDICTION TRAINING REQUEST - Store {store_id}")
    logger.info(f"{'='*80}")
    
    try:
        # Train model
        trainer = NextItemTrainer(store_id)
        result = trainer.train()
        
        # Save model
        model_dir = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'next_item'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'next_item_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"NEXT ITEM PREDICTION TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        
        return jsonify({
            'status': 'success',
            'store_id': store_id,
            'model_path': str(model_path),
            'metrics': result['metrics']
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Next item training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/update/trending-cache', methods=['POST'])
def update_trending_cache():
    """Update trending cache daily without retraining the model"""
    data = request.json
    store_id = data.get('store_id')
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    try:
        import lightgbm as lgb
        from data_extraction import extract_time_series_data
        from cache_manager import cache_trending
        
        # Load existing trained model
        model_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'trending' / 'lightgbm_model.txt'
        metadata_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'trending' / 'metadata.pkl'
        
        if not model_path.exists():
            return jsonify({'error': f'Model not found for store {store_id}. Please train first.'}), 404
        
        model = lgb.Booster(model_file=str(model_path))
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract latest time series data
        time_series_df = extract_time_series_data(store_id, days_back=30)
        
        # Update cache using existing model (use_existing_version=True)
        cached_count, model_version = cache_trending(
            store_id,
            model,
            time_series_df,
            metadata['feature_cols'],
            use_existing_version=True
        )
        
        logger.info(f"Updated trending cache for store {store_id}: {cached_count} products")
        
        return jsonify({
            'success': True,
            'store_id': store_id,
            'cached_products': cached_count,
            'model_version': model_version
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to update trending cache: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/predict/next-items', methods=['POST'])
def predict_next_items():
    """Predict next items for a customer based on purchase history"""
    data = request.json
    store_id = data.get('store_id')
    item_history = data.get('item_history', [])  # List of item IDs
    top_k = data.get('top_k', 10)
    
    if not store_id:
        return jsonify({'error': 'store_id required'}), 400
    
    try:
        # Load model
        model_path = config.MODEL_SAVE_DIR / f'store_{store_id}' / 'next_item' / 'next_item_model.pkl'
        
        if not model_path.exists():
            return jsonify({'error': f'Model not found for store {store_id}. Please train first.'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        transition_probs = model_data['transition_patterns']
        
        # Predict
        if not item_history:
            return jsonify({'predictions': []}), 200
        
        # History is ordered DESC (newest first), so first item is the most recent
        last_item = item_history[0]
        
        if last_item not in transition_probs:
            return jsonify({'predictions': []}), 200
        
        predictions = transition_probs[last_item]
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
        
        return jsonify({
            'store_id': store_id,
            'input_history': item_history,
            'predictions': results
        }), 200
        
    except Exception as e:
        logger.error(f"Next item prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("AI TRAINING SERVICE v2.0")
    logger.info("=" * 80)
    logger.info(f"Flask Host: {config.FLASK_HOST}")
    logger.info(f"Flask Port: {config.FLASK_PORT}")
    logger.info(f"Database: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
    logger.info(f"Models Directory: {config.MODEL_SAVE_DIR}")
    logger.info("=" * 80)
    
    app.run(
        host=config.FLASK_HOST, 
        port=config.FLASK_PORT, 
        debug=config.FLASK_DEBUG
    )
