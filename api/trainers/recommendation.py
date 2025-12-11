"""
Recommendation model trainer
"""

import torch
import logging
import time
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config
from data_extraction import extract_interactions
from src.models.recommendation.train_neumf import NeuMF, InteractionDataset

logger = logging.getLogger(__name__)


class RecommendationTrainer:
    """Trains NeuMF recommendation model"""
    
    def __init__(self, store_id):
        self.store_id = store_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self):
        """Train recommendation model for the store"""
        logger.info(f"=" * 60)
        logger.info(f"RECOMMENDATION TRAINING - Store {self.store_id}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Step 1: Extract and preprocess interactions
        interactions_df = extract_interactions(self.store_id)
        
        # Step 2: Prepare data for collaborative filtering
        logger.info(f"\nStep 2: Preparing data for NeuMF model...")
        
        all_users = interactions_df['user_id'].unique()
        all_items = interactions_df['item_id'].unique()
        
        logger.info(f"  ✅ Users: {len(all_users):,}")
        logger.info(f"  ✅ Items: {len(all_items):,}")
        logger.info(f"  ✅ Sparsity: {(1 - len(interactions_df) / (len(all_users) * len(all_items))) * 100:.2f}%")
        
        # Aggregate interactions
        agg_interactions = interactions_df.groupby(['user_id', 'item_id']).agg({
            'weight': 'sum',
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        agg_interactions['rating'] = agg_interactions['weight'] / agg_interactions['weight'].max()
        
        logger.info(f"  ✅ Aggregated to {len(agg_interactions):,} unique user-item pairs")
        
        # Step 3: Train/Val/Test split (temporal) - 70/15/15
        logger.info(f"\nStep 3: Temporal train/val/test split (70/15/15)...")
        
        interactions_with_time = interactions_df.merge(
            agg_interactions[['user_id', 'item_id', 'rating']], 
            on=['user_id', 'item_id']
        )
        interactions_with_time = interactions_with_time.sort_values('timestamp')
        
        # Split indices: 70% train, 15% val, 15% test
        split1_idx = int(len(interactions_with_time) * 0.7)
        split2_idx = int(len(interactions_with_time) * 0.85)
        
        train_data = interactions_with_time.iloc[:split1_idx]
        val_data = interactions_with_time.iloc[split1_idx:split2_idx]
        test_data = interactions_with_time.iloc[split2_idx:]
        
        train_df = train_data.groupby(['user_id', 'item_id'])['rating'].max().reset_index()
        val_df = val_data.groupby(['user_id', 'item_id'])['rating'].max().reset_index()
        test_df = test_data.groupby(['user_id', 'item_id'])['rating'].max().reset_index()
        
        # Rename 'rating' to 'weight' for dataset compatibility
        train_df = train_df.rename(columns={'rating': 'weight'})
        val_df = val_df.rename(columns={'rating': 'weight'})
        test_df = test_df.rename(columns={'rating': 'weight'})
        
        logger.info(f"  ✅ Train: {len(train_df):,} interactions ({len(train_df)/len(agg_interactions)*100:.1f}%)")
        logger.info(f"  ✅ Validation: {len(val_df):,} interactions ({len(val_df)/len(agg_interactions)*100:.1f}%)")
        logger.info(f"  ✅ Test: {len(test_df):,} interactions ({len(test_df)/len(agg_interactions)*100:.1f}%)")
        
        # Step 4: Create datasets
        logger.info(f"\nStep 4: Creating PyTorch datasets...")
        
        train_dataset = InteractionDataset(train_df, all_users, all_items)
        val_dataset = InteractionDataset(val_df, all_users, all_items)
        test_dataset = InteractionDataset(test_df, all_users, all_items)
        
        train_loader = DataLoader(train_dataset, batch_size=config.REC_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.REC_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.REC_BATCH_SIZE, shuffle=False)
        
        logger.info(f"  ✅ Train loader: {len(train_loader)} batches")
        logger.info(f"  ✅ Validation loader: {len(val_loader)} batches")
        logger.info(f"  ✅ Test loader: {len(test_loader)} batches")
        
        # Step 5: Initialize and train model
        logger.info(f"\nStep 5: Training NeuMF model...")
        logger.info(f"  Configuration:")
        logger.info(f"    - Embedding dimension: {config.REC_EMBED_DIM}")
        logger.info(f"    - MLP layers: {config.REC_HIDDEN_LAYERS}")
        logger.info(f"    - Optimizer: Adam (lr={config.REC_LEARNING_RATE})")
        logger.info(f"    - Loss: MSE")
        logger.info(f"    - Epochs: {config.REC_EPOCHS}")
        logger.info(f"    - Device: {self.device}")
        
        model = NeuMF(
            n_users=len(all_users),
            n_items=len(all_items),
            embed_dim=config.REC_EMBED_DIM,
            hidden_layers=config.REC_HIDDEN_LAYERS
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.REC_LEARNING_RATE)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        logger.info(f"\n  Training progress:")
        for epoch in range(config.REC_EPOCHS):
            # Train
            model.train()
            train_loss = 0
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(users, items).squeeze()
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items, ratings in val_loader:
                    users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                    predictions = model(users, items).squeeze()
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"    Epoch {epoch+1}/{config.REC_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        
        logger.info(f"\n  ✅ Training completed in {training_time:.2f}s")
        logger.info(f"  ✅ Best validation loss: {best_val_loss:.4f}")
        
        # Step 6: Final evaluation on test set
        logger.info(f"\nStep 6: Final evaluation on test set...")
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for users, items, ratings in test_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                predictions = model(users, items).squeeze()
                loss = criterion(predictions, ratings)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        logger.info(f"  ✅ Test loss: {test_loss:.4f}")
        
        # Step 7: Calculate ranking metrics (NDCG, Precision, Recall, Hit Rate)
        logger.info(f"\nStep 7: Calculating ranking metrics on test set...")
        
        model.eval()
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        hit_rate_scores = []
        auc_scores = []
        
        k = 10  # Top-K recommendations
        
        with torch.no_grad():
            # Group test data by user
            test_users = test_df['user_id'].unique()
            
            for user_id in test_users:
                if user_id not in train_dataset.user_to_idx:
                    continue
                
                user_idx = train_dataset.user_to_idx[user_id]
                
                # Get ground truth items for this user
                ground_truth = set(test_df[test_df['user_id'] == user_id]['item_id'].values)
                
                if len(ground_truth) == 0:
                    continue
                
                # Generate predictions for all items
                scores = []
                y_true = []  # For AUC
                y_score = []  # For AUC
                
                for item_id, item_idx in train_dataset.item_to_idx.items():
                    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                    score = model(user_tensor, item_tensor).cpu().item()
                    scores.append((item_id, score))
                    
                    # For AUC calculation
                    y_true.append(1 if item_id in ground_truth else 0)
                    y_score.append(score)
                
                # Calculate AUC (if we have both positive and negative samples)
                if len(set(y_true)) > 1:  # Need at least one positive and one negative
                    from sklearn.metrics import roc_auc_score
                    try:
                        auc = roc_auc_score(y_true, y_score)
                        auc_scores.append(auc)
                    except:
                        pass  # Skip if calculation fails
                
                # Sort by score and get top K
                scores.sort(key=lambda x: x[1], reverse=True)
                top_k_items = [item_id for item_id, _ in scores[:k]]
                
                # Calculate metrics
                hits = len(set(top_k_items) & ground_truth)
                
                # Precision@K
                precision = hits / k
                precision_scores.append(precision)
                
                # Recall@K
                recall = hits / len(ground_truth)
                recall_scores.append(recall)
                
                # Hit Rate@K (1 if at least one hit, 0 otherwise)
                hit_rate = 1.0 if hits > 0 else 0.0
                hit_rate_scores.append(hit_rate)
                
                # NDCG@K
                dcg = 0.0
                idcg = 0.0
                for i, item_id in enumerate(top_k_items):
                    if item_id in ground_truth:
                        dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
                
                # Ideal DCG (if all top-K were relevant)
                for i in range(min(k, len(ground_truth))):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_scores.append(ndcg)
        
        # Average metrics
        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        avg_hit_rate = float(np.mean(hit_rate_scores)) if hit_rate_scores else 0.0
        avg_auc = float(np.mean(auc_scores)) if auc_scores else 0.0
        
        logger.info(f"  ✅ Evaluated {len(ndcg_scores)} users")
        logger.info(f"  Metrics @{k}:")
        logger.info(f"    NDCG:      {avg_ndcg:.4f}")
        logger.info(f"    Precision: {avg_precision:.4f}")
        logger.info(f"    Recall:    {avg_recall:.4f}")
        logger.info(f"    Hit Rate:  {avg_hit_rate:.4f}")
        logger.info(f"    AUC:       {avg_auc:.4f}")
        
        # Step 7: Prepare results
        metrics = {
            'final_train_loss': float(train_loss),
            'final_test_loss': float(test_loss),
            'best_val_loss': float(best_val_loss),
            'ndcg@10': avg_ndcg,
            'precision@10': avg_precision,
            'recall@10': avg_recall,
            'hit_rate@10': avg_hit_rate,
            'auc': avg_auc,
            'training_time': int(training_time),
            'num_users': int(len(all_users)),
            'num_items': int(len(all_items)),
            'num_interactions': int(len(interactions_df)),
            'num_order_interactions': int(len(interactions_df[interactions_df['action_type'] == 'order'])),
            'num_cart_interactions': int(len(interactions_df[interactions_df['action_type'] == 'cart'])),
            'sparsity': float((1 - len(interactions_df) / (len(all_users) * len(all_items))) * 100),
            'epochs': config.REC_EPOCHS,
            'preprocessing_pipeline': 'validated'
        }
        
        model_data = {
            'model_state_dict': model.state_dict(),
            'user_to_idx': train_dataset.user_to_idx,
            'item_to_idx': train_dataset.item_to_idx,
            'n_users': len(all_users),
            'n_items': len(all_items),
            'metrics': metrics,
            'preprocessing_version': 'v2.0_validated'
        }
        
        return {
            'model': model,
            'model_data': model_data,
            'train_dataset': train_dataset,
            'all_users': all_users,
            'metrics': metrics
        }
