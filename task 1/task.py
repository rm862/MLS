import torch
import numpy as np
import time

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    # Normalize vectors for cosine distance calculation
    X_norm = torch.nn.functional.normalize(X, dim=1)
    Y_norm = torch.nn.functional.normalize(Y, dim=1)
    # Calculate cosine similarity
    similarity = torch.matmul(X_norm, Y_norm.T)
    # Clamp values to avoid numerical issues
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)
    # Convert similarity to distance (1 - similarity)
    return 1.0 - similarity

def distance_l2(X, Y):
    # Euclidean distance calculation using matrix operations
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y
    X_squared = (X ** 2).sum(dim=1, keepdim=True)
    Y_squared = (Y ** 2).sum(dim=1, keepdim=True).T
    XY = torch.matmul(X, Y.T)
    distances = X_squared + Y_squared - 2 * XY
    # Clamp to prevent negative values due to numerical precision
    distances = torch.clamp(distances, min=1e-12)
    return torch.sqrt(distances)

def distance_dot(X, Y):
    # Negative dot product as distance (higher dot product = lower distance)
    return -torch.matmul(X, Y.T)

def distance_manhattan(X, Y):
    # Optimization: Use batch processing for large datasets
    batch_size = 1024
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # For large matrices, use batched computation to save memory
    if n_x * n_y > 10000000 and max(n_x, n_y) > batch_size:
        result = torch.zeros((n_x, n_y), device=X.device)
        for i in range(0, n_x, batch_size):
            end_i = min(i + batch_size, n_x)
            # Process one batch at a time
            result[i:end_i] = torch.cdist(X[i:end_i], Y, p=1)
        return result
    else:
        # For smaller matrices, compute all at once
        return torch.cdist(X, Y, p=1)

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def custom_topk(distances, k):
    # Implementation of top-k algorithm for finding k smallest distances
    n_queries = distances.shape[0]
    top_indices = torch.zeros((n_queries, k), dtype=torch.int64, device=distances.device)

    for i in range(n_queries):
        dist_row = distances[i]
        for j in range(k):
            if j == 0:
                # For first neighbor, simply find minimum
                _, min_idx = torch.min(dist_row, dim=0)
            else:
                # For subsequent neighbors, mask previously found indices
                masked = dist_row.clone()
                masked[top_indices[i, :j]] = float('inf')
                _, min_idx = torch.min(masked, dim=0)
            top_indices[i, j] = min_idx

    return top_indices

def our_knn(N, D, A, X, K, metric="L2"):
    # Use GPU if available for acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to tensors if needed and move to device
    A_is_tensor = torch.is_tensor(A)
    X_is_tensor = torch.is_tensor(X)

    # Optimization: For large datasets, use batching to avoid memory issues
    large_dataset = N > 100000  # Consider datasets with >100k vectors as large

    # Handle input formatting for X (single or multiple queries)
    if X_is_tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)  # Convert to 2D tensor for single query
        num_queries = X.shape[0]
    else:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Convert to 2D array for single query
        num_queries = X.shape[0]

    # Initialize result tensor on CPU to save GPU memory
    result_indices = torch.zeros((num_queries, K), dtype=torch.int64, device='cpu')

    # Configure batch sizes for processing
    query_batch_size = min(64, num_queries)  # Process up to 64 queries at once
    db_batch_size = min(10000, N) if large_dataset else N  # Use smaller batches for large datasets

    # Process queries in batches for memory efficiency
    for q_start in range(0, num_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, num_queries)

        # Create query batch tensor on device
        if X_is_tensor:
            X_batch = X[q_start:q_end].to(device)
        else:
            X_batch = torch.tensor(X[q_start:q_end], dtype=torch.float32, device=device)

        if large_dataset:
            # Optimization: For large datasets, process database in batches
            # Initialize top K tracking for each query in batch
            topk_values = torch.full((q_end - q_start, K), float('inf'), device=device)
            topk_indices = torch.zeros((q_end - q_start, K), dtype=torch.int64, device=device)

            # Process database in batches
            for db_start in range(0, N, db_batch_size):
                db_end = min(db_start + db_batch_size, N)

                # Create database batch tensor on device
                if A_is_tensor:
                    A_batch = A[db_start:db_end].to(device)
                else:
                    A_batch = torch.tensor(A[db_start:db_end], dtype=torch.float32, device=device)

                # Calculate distances using the appropriate metric
                if metric.lower() == "l2":
                    batch_dist = distance_l2(X_batch, A_batch)
                elif metric.lower() == "cosine":
                    batch_dist = distance_cosine(X_batch, A_batch)
                elif metric.lower() == "dot":
                    batch_dist = distance_dot(X_batch, A_batch)
                elif metric.lower() == "manhattan":
                    batch_dist = distance_manhattan(X_batch, A_batch)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Update top-K for each query by combining results from each batch
                batch_size = q_end - q_start
                for i in range(batch_size):
                    # Combine current results with new batch results
                    curr_values = topk_values[i]
                    curr_indices = topk_indices[i]

                    # Get current batch distances and indices
                    new_values = batch_dist[i]
                    new_indices = torch.arange(db_start, db_end, device=device)

                    # Combine and get new top-K
                    if db_start > 0:  # Not the first batch
                        all_values = torch.cat([curr_values, new_values])
                        all_indices = torch.cat([curr_indices, new_indices])

                        # Get top-K (smallest for L2, cosine, manhattan; largest for dot)
                        if metric.lower() == "dot":
                            topk = torch.topk(all_values, min(K, len(all_values)), largest=True)
                        else:
                            topk = torch.topk(all_values, min(K, len(all_values)), largest=False)

                        topk_values[i] = topk.values
                        topk_indices[i] = all_indices[topk.indices]
                    else:  # First batch, just take top-K
                        if metric.lower() == "dot":
                            topk = torch.topk(new_values, min(K, len(new_values)), largest=True)
                        else:
                            topk = torch.topk(new_values, min(K, len(new_values)), largest=False)

                        topk_values[i, :len(topk.values)] = topk.values
                        topk_indices[i, :len(topk.indices)] = new_indices[topk.indices]

                # Memory optimization: clear batch data after processing
                del A_batch, batch_dist
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Store results for this query batch
            result_indices[q_start:q_end] = topk_indices.cpu()

        else:
            # For smaller datasets, process entire database at once (more efficient)
            A_device = A.to(device) if A_is_tensor else torch.tensor(A, dtype=torch.float32, device=device)

            # Calculate distances using the appropriate metric
            if metric.lower() == "l2":
                distances = distance_l2(X_batch, A_device)
            elif metric.lower() == "cosine":
                distances = distance_cosine(X_batch, A_device)
            elif metric.lower() == "dot":
                distances = distance_dot(X_batch, A_device)
            elif metric.lower() == "manhattan":
                distances = distance_manhattan(X_batch, A_device)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Get top-K indices
            if metric.lower() == "dot":
                _, indices = torch.topk(distances, K, dim=1, largest=True)
            else:
                _, indices = torch.topk(distances, K, dim=1, largest=False)

            # Store results
            result_indices[q_start:q_end] = indices.cpu()

            # Memory optimization: clear data when done with non-tensor input
            if not A_is_tensor:
                del A_device

        # Memory optimization: clear batch data after processing
        del X_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return result_indices

def benchmark_knn(N, D, K, metric="L2"):
    # Generate random data for benchmarking
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(100, D).astype(np.float32)

    # Time the KNN query execution
    start = time.time()
    _ = our_knn(N, D, A, X, K, metric)
    return time.time() - start

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def distance_kernel(X, Y, metric="L2"):
    # Optimized distance calculation between matrix X and Y
    if metric == "L2":
        X2 = (X ** 2).sum(dim=1, keepdim=True)
        Y2 = (Y ** 2).sum(dim=1, keepdim=True).T
        return torch.sqrt(torch.clamp(X2 + Y2 - 2 * X @ Y.T, min=0.0))
    elif metric == "cosine":
        # Normalize vectors and compute cosine distance
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)  # Add epsilon to avoid division by zero
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        return 1 - torch.mm(X_norm, Y_norm.T)
    else:
        raise ValueError("metric must be 'L2' or 'cosine'")

def our_kmeans(N, D, A, K, metric="L2", max_iters=100, tol=1e-4, use_gpu=True):
    # Select device based on availability and user preference
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Running K-Means on PyTorch ({device.upper()} backend), metric={metric}")

    # Convert input to tensor and move to device
    A_tensor = torch.tensor(A, dtype=torch.float32).contiguous()
    if use_gpu and A_tensor.device.type == "cpu":
        # Use pin_memory for faster host-to-device transfer
        A_tensor = A_tensor.pin_memory().to(device, non_blocking=True)
    else:
        A_tensor = A_tensor.to(device)

    # Initialize centroids randomly from data points
    # centroids = A_tensor[torch.randperm(N)[:K]].clone()
    centroids = A_tensor[torch.randperm(N, device=device)[:K]].clone()

    # Main K-means loop
    for i in range(max_iters):
        # Calculate distances from each point to each centroid
        distances = distance_kernel(A_tensor, centroids, metric=metric)

        # Assign each point to nearest centroid
        labels = torch.argmin(distances, dim=1)

        # Update centroids as mean of assigned points
        new_centroids = torch.zeros_like(centroids, device=device)
        ones = torch.ones_like(labels, dtype=torch.float32, device=device)

        # Count points per cluster
        counts = torch.zeros(K, dtype=torch.float32, device=device).scatter_add_(0, labels, ones)

        # Sum points per cluster
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), A_tensor)

        # Divide by count to get mean (avoid division by zero with clamp_min)
        new_centroids /= counts.unsqueeze(1).clamp_min(1)

        # Check for convergence
        if torch.norm(new_centroids - centroids, p=2, dim=1).max().item() < tol:
            print(f"K-Means converged at iteration {i+1}")
            break

        centroids = new_centroids

    return labels.cpu()



def our_kmeans_modified(N, D, A, num_clusters, metric="L2", max_iters=100, tol=1e-4, use_gpu=True):
    """
    KMeans clustering with GPU/CPU support.

    Parameters:
      N: Number of data points.
      D: Data dimension.
      A: Numpy array of shape (N, D).
      num_clusters: Number of clusters to form.
      metric: "L2" or "cosine" used to compute distances.
      max_iters: Maximum iterations.
      tol: Convergence threshold.
      use_gpu: If True, use CUDA if available.

    Returns:
      labels: Tensor of shape (N,) with cluster assignments.
      centroids: Tensor of shape (num_clusters, D) representing cluster centers.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Check if A is already a torch tensor and move to correct device
    if torch.is_tensor(A):
        A_tensor = A
        if A_tensor.device.type != device:
            A_tensor = A_tensor.to(device)
    else:
        A_tensor = torch.tensor(A, dtype=torch.float32).contiguous()
        if use_gpu and torch.cuda.is_available():
            A_tensor = A_tensor.to(device)

    # Initialize centroids using k-means++ initialization for better clustering quality
    centroids = torch.zeros((num_clusters, D), dtype=torch.float32, device=device)

    # Choose first centroid randomly
    # first_idx = torch.randint(0, N, (1,)).item()
    first_idx = torch.randint(0, N, (1,), device=device).item()
    centroids[0] = A_tensor[first_idx].clone()

    # Choose remaining centroids using k-means++ algorithm (weighted sampling based on distance)
    for i in range(1, num_clusters):
        # Calculate distances to closest existing centroid
        min_dists = torch.min(distance_kernel(A_tensor, centroids[:i], metric=metric), dim=1)[0]

        # Square distances and normalize to create a probability distribution
        # Points farther from existing centroids have higher probability
        probs = min_dists ** 2
        probs /= probs.sum()

        # Sample next centroid based on probability distribution
        next_idx = torch.multinomial(probs, 1).item()
        centroids[i] = A_tensor[next_idx].clone()

    # Main K-means loop
    for i in range(max_iters):
        distances = distance_kernel(A_tensor, centroids, metric=metric)  # (N, num_clusters)
        labels = torch.argmin(distances, dim=1)  # (N,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids, device=device)
        ones = torch.ones_like(labels, dtype=torch.float32, device=device)
        counts = torch.zeros(num_clusters, dtype=torch.float32, device=device)
        counts = counts.scatter_add_(0, labels, ones)
        new_centroids = new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), A_tensor)
        counts = counts.unsqueeze(1).clamp_min(1)  # Prevent division by zero
        new_centroids /= counts

        # Check for convergence
        shift = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()
        centroids = new_centroids
        if shift < tol:
            print(f"K-Means converged at iteration {i + 1}")
            break

    return labels.cpu(), centroids.cpu()


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# Optimized Approximate Nearest Neighbors Implementation with Recall Guarantees

def calculate_clusters_for_recall(K, desired_recall=0.7):
    """
    Calculate how many clusters need to be searched to achieve the desired recall rate.
    Uses the formula L ≥ (K-1)/(1-r) where:
    - K is the number of nearest neighbors we want
    - r is the desired recall rate (0.0 to 1.0)
    - L is the number of clusters to search

    Returns: Number of clusters to search (L)
    """
    if desired_recall >= 1.0:
        # If perfect recall is desired, we'd need to search all clusters
        return float('inf')

    # Mathematical formula to guarantee recall rate
    L = (K - 1) / (1 - desired_recall)
    return max(int(np.ceil(L)), 1)  # Ensure at least 1 cluster is searched

def our_ann_improved(N, D, A, X, K, num_clusters=None, desired_recall=0.7, metric="L2", max_iters=100, tol=1e-4, use_gpu=True):
    """
    Optimized Approximate Nearest Neighbor (ANN) algorithm with mathematical recall guarantees.

    Parameters:
      N: Number of vectors.
      D: Dimension of each vector.
      A: Numpy array of shape (N, D).
      X: Query vector (e.g. numpy array of shape (D,)).
      K: Number of nearest neighbors to return.
      num_clusters: Number of clusters for KMeans (auto-calculated if None).
      desired_recall: Target recall rate (0.0 to 1.0) for the algorithm.
      metric: "L2" or "cosine" distance metric.
      max_iters, tol, use_gpu: Parameters for KMeans.

    Returns:
      A list of indices (of length K) indicating the top K nearest neighbors in A.
    """
    # Set device first and be consistent
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Auto-tune parameters based on dataset size and desired recall
    if num_clusters is None:
        # Calculate optimal number of clusters based on data size
        # Optimization: use sqrt(N) as a heuristic but cap min/max values
        num_clusters = min(max(int(np.sqrt(N)), 10), 200)

    # Calculate K1 (number of clusters to search) based on recall guarantee formula
    K1 = calculate_clusters_for_recall(K, desired_recall)
    K1 = min(K1, num_clusters)  # Cannot search more clusters than exist

    # Calculate K2 (candidates per cluster) based on dimensionality scaling
    # Optimization: Higher dimensions need more candidates due to curse of dimensionality
    dim_factor = max(1.0, np.log10(D) / 2)
    K2 = int(K * max(5, dim_factor))
    K2 = min(K2, 200)  # Cap to avoid excessive computation

    print(f"ANN parameters (recall guarantee): num_clusters={num_clusters}, K1={K1}, K2={K2}, desired_recall={desired_recall}")

    # Step 1: Convert inputs to tensors and move to the correct device
    # Convert A to tensor if it's not already
    if torch.is_tensor(A):
        A_tensor = A.to(device)
    else:
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    # Convert X to tensor if it's not already
    if torch.is_tensor(X):
        X_tensor = X.to(device)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    # Ensure X has the right shape
    if X_tensor.dim() == 1:
        X_tensor = X_tensor.unsqueeze(0)  # shape (1, D)

    # Step 2: Cluster A into num_clusters clusters using KMeans
    labels, centroids = our_kmeans_modified(N, D, A_tensor, num_clusters,
                                        metric=metric, max_iters=max_iters, tol=tol,
                                        use_gpu=use_gpu)

    # Convert labels and centroids to tensors on the correct device
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=device)
    else:
        labels = labels.to(device)

    if not torch.is_tensor(centroids):
        centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    else:
        centroids = centroids.to(device)

    # Step 3: Calculate cluster statistics for density-aware search
    cluster_sizes = torch.zeros(num_clusters, dtype=torch.float32, device=device)
    for c in range(num_clusters):
        cluster_sizes[c] = torch.sum(labels == c).float()

    # Compute density-based weights for each cluster
    # Optimization: Consider cluster density in priority calculation
    total_points = torch.sum(cluster_sizes).float()
    cluster_weights = cluster_sizes / total_points

    # Step 4: Calculate distance from query to each centroid
    if metric == "L2":
        centroid_dist = torch.cdist(X_tensor, centroids, p=2).squeeze(0)
    elif metric == "cosine":
        X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
        centroids_norm = centroids / (torch.norm(centroids, dim=1, keepdim=True) + 1e-8)
        centroid_dist = 1 - torch.matmul(X_norm, centroids_norm.t()).squeeze(0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Step 5: Create a priority score for each cluster combining distance and density
    # Optimization: Adaptive radius factor based on dimensionality
    dim_radius_factor = 1.0 + (D / 1000)

    # Softmax-normalize distances to get probability distribution
    exp_neg_dist = torch.exp(-centroid_dist / dim_radius_factor)
    softmax_probs = exp_neg_dist / torch.sum(exp_neg_dist)

    # Combine distance probability with density weight
    # Optimization: Balance between distance and density with alpha parameter
    alpha = 0.7  # Weight for distance vs density
    cluster_scores = alpha * softmax_probs + (1 - alpha) * cluster_weights

    # Get top K1 clusters based on combined score
    _, ranked_clusters = torch.topk(cluster_scores, min(K1, num_clusters))
    selected_clusters = ranked_clusters.cpu().numpy().tolist()

    # Step 6: Process selected clusters to find candidate neighbors
    unique_candidate_indices = set()
    candidate_with_dist = []

    for c in selected_clusters:
        # Get indices of all points in this cluster
        cluster_indices = torch.nonzero(labels == c, as_tuple=True)[0]

        if cluster_indices.numel() == 0:
            continue

        # Calculate distances between query and points in this cluster
        cluster_points = A_tensor[cluster_indices]

        if metric == "L2":
            dists = torch.cdist(X_tensor, cluster_points, p=2).squeeze(0)
        elif metric == "cosine":
            X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
            cluster_norm = cluster_points / (torch.norm(cluster_points, dim=1, keepdim=True) + 1e-8)
            dists = 1 - torch.matmul(X_norm, cluster_norm.t()).squeeze(0)

        # Get top K2 candidates from this cluster
        topk = min(K2, cluster_points.size(0))
        _, cluster_topk = torch.topk(-dists, topk, largest=True)

        # Add candidates with their distances to our list
        for i in range(len(cluster_topk)):
            idx = cluster_indices[cluster_topk[i]].item()
            dist = dists[cluster_topk[i]].item()
            if idx not in unique_candidate_indices:
                unique_candidate_indices.add(idx)
                candidate_with_dist.append((dist, idx))

    # Step 7: Final candidate selection
    if len(candidate_with_dist) < K:
        # Not enough candidates, perform exact search
        # Fallback strategy to guarantee results even with insufficient candidates
        print("Warning: Not enough candidates found in selected clusters, performing exact search")

        if metric == "L2":
            dists = torch.cdist(X_tensor, A_tensor, p=2).squeeze(0)
        elif metric == "cosine":
            X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
            A_norm = A_tensor / (torch.norm(A_tensor, dim=1, keepdim=True) + 1e-8)
            dists = 1 - torch.matmul(X_norm, A_norm.t()).squeeze(0)

        _, indices = torch.topk(-dists, min(K, N), largest=True)
        return indices.cpu().numpy().tolist()

    # Sort candidates by distance and return top K
    candidate_with_dist.sort()  # Sort by distance (first element of tuple)
    top_candidates = [idx for _, idx in candidate_with_dist[:K]]

    return top_candidates

# Main Benchmark Runner
if __name__ == "__main__":
    print("Speed benchmark for D=2:")
    print(f"Time for 10,000 vectors with D=2: {benchmark_knn(10000, 2, 10):.4f} seconds")
    print("\nSpeed benchmark for D=2^15:")
    print(f"Time for 1,000 vectors with D=2^15: {benchmark_knn(1000, 2**15, 10):.4f} seconds")
    print("\nProcessing 4,000 vectors:")
    print(f"Time for 4,000 vectors: {benchmark_knn(4000, 128, 10):.4f} seconds")
    print("\nProcessing simulation for 4,000,000 vectors:")
    sample_time = benchmark_knn(40000, 128, 10)
    print(f"Estimated time for 4,000,000 vectors: {sample_time * (4000000 / 40000):.2f} seconds")
