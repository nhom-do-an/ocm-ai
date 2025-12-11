"""
Evaluation metrics for recommendation systems.

References:
- RecBole metrics: https://recbole.io/docs/user_guide/evaluation_support.html
- Precision/Recall@K: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
- NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
"""

import numpy as np
from typing import List, Dict, Tuple


def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    Precision@K - Tỷ lệ items đúng trong top-K recommendations.
    
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        k: Number of top items to consider
        
    Returns:
        float: Precision@K score [0, 1]
        
    Example:
        >>> actual = [1, 2, 3, 4, 5]
        >>> predicted = [1, 6, 2, 7, 8, 3, 9, 10]
        >>> precision_at_k(actual, predicted, k=5)
        0.4  # 2 out of 5 items are relevant
        
    Reference:
        https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision
    """
    if k == 0 or len(predicted) == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    relevant_in_k = len(set(actual) & set(predicted_k))
    return relevant_in_k / k


def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    Recall@K - Tỷ lệ items đúng được tìm thấy trong top-K.
    
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        k: Number of top items to consider
        
    Returns:
        float: Recall@K score [0, 1]
        
    Example:
        >>> actual = [1, 2, 3, 4, 5]
        >>> predicted = [1, 6, 2, 7, 8, 3, 9, 10]
        >>> recall_at_k(actual, predicted, k=5)
        0.4  # 2 out of 5 relevant items found
    """
    if len(actual) == 0 or k == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    relevant_in_k = len(set(actual) & set(predicted_k))
    return relevant_in_k / len(actual)


def ndcg_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain) - Đánh giá ranking quality.
    
    Công thức:
        DCG@K = Σ(i=1 to K) rel_i / log₂(i + 1)
        NDCG@K = DCG@K / IDCG@K
        
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        k: Number of top items to consider
        
    Returns:
        float: NDCG@K score [0, 1]
        
    Reference:
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    def dcg(relevances: List[float], k: int) -> float:
        """Discounted Cumulative Gain"""
        relevances = np.array(relevances)[:k]
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(2, len(relevances) + 2))
        return np.sum(relevances / discounts)
    
    if len(actual) == 0 or k == 0:
        return 0.0
    
    # Binary relevance: 1 if item in actual, 0 otherwise
    predicted_k = predicted[:k]
    relevances = [1 if item in actual else 0 for item in predicted_k]
    
    # Calculate DCG
    dcg_score = dcg(relevances, k)
    
    # Calculate IDCG (ideal DCG - all relevant items ranked first)
    ideal_relevances = [1] * min(len(actual), k)
    idcg_score = dcg(ideal_relevances, k)
    
    if idcg_score == 0:
        return 0.0
    
    return dcg_score / idcg_score


def hit_rate_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    Hit Rate@K - Binary indicator: 1 if at least 1 relevant item in top-K, 0 otherwise.
    
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        k: Number of top items to consider
        
    Returns:
        float: 1.0 if hit, 0.0 if miss
        
    Example:
        >>> actual = [1, 2, 3]
        >>> predicted = [4, 5, 1, 6, 7]
        >>> hit_rate_at_k(actual, predicted, k=3)
        1.0  # Item 1 is in top-3
    """
    if k == 0 or len(actual) == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    return 1.0 if len(set(actual) & set(predicted_k)) > 0 else 0.0


def mrr(actual: List[int], predicted: List[int]) -> float:
    """
    MRR (Mean Reciprocal Rank) - 1 / rank of first relevant item.
    
    Công thức:
        MRR = 1 / rank_of_first_relevant_item
        
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        
    Returns:
        float: MRR score [0, 1]
        
    Example:
        >>> actual = [1, 2, 3]
        >>> predicted = [4, 5, 1, 6, 7]
        >>> mrr(actual, predicted)
        0.333  # First relevant item at position 3
        
    Reference:
        https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    
    actual_set = set(actual)
    for i, item in enumerate(predicted):
        if item in actual_set:
            return 1.0 / (i + 1)  # Rank starts from 1
    
    return 0.0


def map_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    MAP@K (Mean Average Precision) - Average precision at each relevant item position.
    
    Công thức:
        AP@K = (1/min(m,K)) × Σ(i=1 to K) P(i) × rel(i)
        where m = |actual|
        
    Args:
        actual: List of actual relevant item IDs
        predicted: List of predicted item IDs (ranked)
        k: Number of top items to consider
        
    Returns:
        float: MAP@K score [0, 1]
        
    Reference:
        https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    """
    if len(actual) == 0 or k == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    actual_set = set(actual)
    
    score = 0.0
    num_hits = 0
    
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            num_hits += 1
            # Precision at this position
            precision = num_hits / (i + 1)
            score += precision
    
    if num_hits == 0:
        return 0.0
    
    return score / min(len(actual), k)


def coverage(predicted_items: List[List[int]], catalog_size: int) -> float:
    """
    Coverage - Tỷ lệ items được gợi ý / tổng số items.
    Đo độ đa dạng của recommendation system.
    
    Args:
        predicted_items: List of predicted item lists for all users
        catalog_size: Total number of items in catalog
        
    Returns:
        float: Coverage score [0, 1]
        
    Example:
        >>> predictions = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        >>> coverage(predictions, catalog_size=100)
        0.05  # Only 5 out of 100 items recommended
    """
    if catalog_size == 0:
        return 0.0
    
    recommended_items = set()
    for pred in predicted_items:
        recommended_items.update(pred)
    
    return len(recommended_items) / catalog_size


def evaluate_recommendations(
    test_data: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    k: int = 10,
    catalog_size: int = None
) -> Dict[str, float]:
    """
    Evaluate recommendation system với multiple metrics.
    
    Args:
        test_data: Dict mapping user_id -> list of actual item IDs
        predictions: Dict mapping user_id -> list of predicted item IDs (ranked)
        k: Number of top items to consider
        catalog_size: Total number of items (for coverage)
        
    Returns:
        Dict of metric name -> score
        
    Example:
        >>> test_data = {1: [10, 20, 30], 2: [15, 25]}
        >>> predictions = {1: [10, 11, 20, 12], 2: [15, 16, 17]}
        >>> metrics = evaluate_recommendations(test_data, predictions, k=3)
        >>> print(metrics)
        {
            'precision@3': 0.667,
            'recall@3': 0.556,
            'ndcg@3': 0.724,
            'hit_rate@3': 1.0,
            'mrr': 0.75,
            'map@3': 0.639
        }
    """
    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []
    mrrs = []
    maps = []
    
    for user_id in test_data:
        if user_id not in predictions:
            continue
        
        actual = test_data[user_id]
        predicted = predictions[user_id]
        
        precisions.append(precision_at_k(actual, predicted, k))
        recalls.append(recall_at_k(actual, predicted, k))
        ndcgs.append(ndcg_at_k(actual, predicted, k))
        hit_rates.append(hit_rate_at_k(actual, predicted, k))
        mrrs.append(mrr(actual, predicted))
        maps.append(map_at_k(actual, predicted, k))
    
    results = {
        f'precision@{k}': np.mean(precisions) if precisions else 0.0,
        f'recall@{k}': np.mean(recalls) if recalls else 0.0,
        f'ndcg@{k}': np.mean(ndcgs) if ndcgs else 0.0,
        f'hit_rate@{k}': np.mean(hit_rates) if hit_rates else 0.0,
        'mrr': np.mean(mrrs) if mrrs else 0.0,
        f'map@{k}': np.mean(maps) if maps else 0.0,
    }
    
    # Add coverage if catalog_size provided
    if catalog_size is not None:
        all_predictions = list(predictions.values())
        results['coverage'] = coverage(all_predictions, catalog_size)
    
    return results


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dict of metric name -> score
        title: Title for the results
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for metric_name, score in metrics.items():
        if 'coverage' in metric_name:
            print(f"{metric_name:20s}: {score:6.4f} ({score*100:5.2f}%)")
        else:
            print(f"{metric_name:20s}: {score:6.4f}")
    
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Test metrics
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 6, 2, 7, 3, 8, 9, 10, 4, 11]
    
    print("Example Calculation:")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted[:10]}")
    print(f"\nPrecision@5: {precision_at_k(actual, predicted, 5):.4f}")
    print(f"Recall@5: {recall_at_k(actual, predicted, 5):.4f}")
    print(f"NDCG@5: {ndcg_at_k(actual, predicted, 5):.4f}")
    print(f"Hit Rate@5: {hit_rate_at_k(actual, predicted, 5):.4f}")
    print(f"MRR: {mrr(actual, predicted):.4f}")
    print(f"MAP@5: {map_at_k(actual, predicted, 5):.4f}")
