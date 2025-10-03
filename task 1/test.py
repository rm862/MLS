# command to use for Google Colab %%writefile test.py 
import sys
import time
import numpy as np
import torch
from task import our_knn, our_kmeans, our_ann_improved

def generate_data(size, for_knn=False):
    """
    Generate synthetic data for testing KNN or K-means algorithms
    Parameters:
        size: String indicating dataset size ("small", "medium", "large", etc.)
        for_knn: Boolean indicating if data is for KNN (includes query vectors)
    Returns:
        N, D, A, [X], K: Dataset parameters, data matrix, [query vectors], number of neighbors/clusters
    """
    if size == "small":
        N, D, K = 100, 2, 5      # Small dataset: 100 vectors, 2 dimensions, 5 neighbors/clusters
        M = 10                   # 10 query vectors
    elif size == "medium":
        N, D, K = 4000, 128, 10  # Medium: 4,000 vectors, 128 dimensions
        M = 100
    elif size == "large":
        N, D, K = 10000, 1024, 20  # Large: 10,000 vectors, 1024 dimensions
        M = 100
    elif size == "large_dim2":
        N, D, K = 10000, 2, 10   # Testing with very low dimension but large number of vectors
        M = 100
    elif size == "large_dim32k":
        N, D, K = 1000, 2**15, 10  # Testing with very high dimension (32,768)
        M = 10
    elif size == "huge":
        # Simulate large dataset performance without using full 4M vectors
        N, D, K = 40000, 128, 10  # Use 40,000 as representative for 4,000,000
        M = 100
        print("Note: Using 40,000 vectors to extrapolate performance for 4,000,000 vectors")

    # Generate random data with normal distribution (float32 for GPU compatibility)
    A = np.random.randn(N, D).astype(np.float32)

    if for_knn:
        # For KNN we need query vectors
        X = np.random.randn(M, D).astype(np.float32)
        return N, D, A, X, K
    else:
        # For K-means we only need the data matrix
        return N, D, A, K


def run_knn(N, D, A, X, K, metric, use_gpu, label):
    """
    Run and benchmark the KNN implementation
    """
    print(f"\n--- Running KNN with {metric.upper()} on {label.upper()} ({N} db, {X.shape[0]} queries, {D} dimensions) ---")

    # Ensure GPU environment is properly set before running KNN
    if use_gpu:
        if torch.cuda.is_available():
            # Synchronize and clear cache for consistent benchmarking
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            print("Warning: GPU requested but not available. Falling back to CPU.")

    # Measure execution time
    start = time.time()
    indices = our_knn(N, D, A, X, K, metric=metric)

    # Ensure all GPU operations complete before stopping timer
    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    print(f"First 3 query results:\n{indices[:3]}")
    print(f"Time: {end - start:.4f} sec")

def compare_cpu_gpu(size, metric="L2"):
    """
    Directly compare CPU and GPU performance on the same dataset
    """
    print(f"\n======= Comparing CPU vs GPU on {size.upper()} dataset with {metric} =======")

    # Generate data for KNN test
    N, D, A, X, K = generate_data(size, for_knn=True)

    # Run on CPU first
    torch.set_default_tensor_type('torch.FloatTensor')
    cpu_start = time.time()
    our_knn(N, D, A, X, K, metric=metric)
    cpu_end = time.time()
    cpu_time = cpu_end - cpu_start

    # Run on GPU if available
    if torch.cuda.is_available():
        # Set default tensor type to CUDA for GPU operations
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.synchronize()  # Ensure GPU is synchronized before timing
        gpu_start = time.time()
        our_knn(N, D, A, X, K, metric=metric)
        torch.cuda.synchronize()  # Ensure all GPU operations complete
        gpu_end = time.time()
        gpu_time = gpu_end - gpu_start

        # Calculate and display speedup
        speedup = cpu_time / gpu_time
        print(f"CPU time: {cpu_time:.4f} sec")
        print(f"GPU time: {gpu_time:.4f} sec")
        print(f"GPU speedup: {speedup:.2f}x")
    else:
        print("GPU not available, skipping comparison")

def extrapolate_large_dataset():
    """
    Estimate performance for very large datasets by extrapolating from smaller samples
    """
    print("\n======= Extrapolating performance for 4,000,000 vectors =======")

    # Test with 40,000 vectors to estimate performance for 4M
    N, D, A, X, K = generate_data("huge", for_knn=True)

    if torch.cuda.is_available():
        # Use GPU for extrapolation
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.synchronize()
        start = time.time()
        our_knn(N, D, A, X, K, metric="L2")
        torch.cuda.synchronize()
        end = time.time()
        measured_time = end - start

        # Linear extrapolation - assuming O(n) scaling which is optimistic
        # Actual scaling may be worse depending on algorithm implementation
        estimated_time = measured_time * (4000000 / N)
        print(f"Time for {N} vectors: {measured_time:.4f} sec")
        print(f"Estimated time for 4,000,000 vectors: {estimated_time:.2f} sec")
        print(f"Optimizations needed for 4M vectors: batching, memory management, results streaming")
    else:
        print("GPU not available, skipping extrapolation")

def compare_all():
    import traceback

    print("\n======= [KNN] Comparing CPU vs GPU =======")
    knn_metrics = ["L2", "cosine", "dot", "manhattan"]
    knn_datasets = ["small", "medium", "large", "large_dim2", "large_dim32k", "huge"]
    for size in knn_datasets:
        for metric in knn_metrics:
            try:
                print(f"\n--- [KNN] Comparing on {size.upper()} with {metric.upper()} ---")
                compare_cpu_gpu(size, metric)
            except Exception as e:
                print(f"[KNN COMPARISON] Skipped {size.upper()} with {metric.upper()} due to error: {e}")
                traceback.print_exc()

    print("\n======= [KMeans] Comparing CPU vs GPU =======")
    kmeans_metrics = ["L2", "cosine"]
    kmeans_datasets = ["small", "large"]
    for size in kmeans_datasets:
        for metric in kmeans_metrics:
            try:
                print(f"\n--- [KMEANS] Running on {size.upper()} with {metric.upper()} ---")
                N, D, A, K = generate_data(size)

                # CPU run
                cpu_start = time.time()
                our_kmeans(N, D, A, K, metric=metric, use_gpu=False)
                cpu_time = time.time() - cpu_start

                # GPU run
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_start = time.time()
                    our_kmeans(N, D, A, K, metric=metric, use_gpu=True)
                    torch.cuda.synchronize()
                    gpu_time = time.time() - gpu_start
                    print(f"CPU time: {cpu_time:.4f} sec")
                    print(f"GPU time: {gpu_time:.4f} sec")
                    print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
                else:
                    print("GPU not available, skipped GPU comparison")

            except Exception as e:
                print(f"[KMEANS COMPARISON] Skipped {size.upper()} with {metric.upper()} due to error: {e}")
                traceback.print_exc()

    print("\n======= [ANN] Comparing CPU vs GPU =======")
    ann_metrics = ["L2", "cosine"]
    ann_datasets = ["small", "large"]
    for size in ann_datasets:
        for metric in ann_metrics:
            try:
                print(f"\n--- [ANN] Running on {size.upper()} with {metric.upper()} ---")
                N, D, A, X, K = generate_data(size, for_knn=True)

                # CPU run
                cpu_start = time.time()
                our_ann_improved(N, D, A, X[0], K, metric=metric, use_gpu=False)
                cpu_time = time.time() - cpu_start

                # GPU run
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_start = time.time()
                    our_ann_improved(N, D, A, X[0], K, metric=metric, use_gpu=True)
                    torch.cuda.synchronize()
                    gpu_time = time.time() - gpu_start
                    print(f"CPU time: {cpu_time:.4f} sec")
                    print(f"GPU time: {gpu_time:.4f} sec")
                    print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
                else:
                    print("GPU not available, skipped GPU comparison")

            except Exception as e:
                print(f"[ANN COMPARISON] Skipped {size.upper()} with {metric.upper()} due to error: {e}")
                traceback.print_exc()

    print("\n======= Extrapolating for HUGE dataset =======")
    extrapolate_large_dataset()


def run_kmeans(N, D, A, K, metric, use_gpu, label):
    """
    Run and benchmark the K-means implementation
    """
    print(f"\n--- Running metric: {metric} on {label.upper()} ({N} samples) ---")
    start = time.time()
    labels = our_kmeans(N, D, A, K, metric=metric, use_gpu=use_gpu)
    end = time.time()
    print(f"First 10 labels: {labels[:10].tolist()}")
    print(f"Time on {label.upper()} ({metric}): {end - start:.4f} sec")

def run_ann_with_guarantees(N, D, A, X, K, metric, use_gpu, label):
    """
    Run and benchmark the improved ANN implementation with recall guarantees

    This is an optimization over traditional ANN by providing mathematical
    guarantees on recall rates, ensuring a minimum quality of results.
    """
    from task import our_ann_improved, our_knn
    import numpy as np
    import time
    import torch

    print(f"\n--- Running ANN (with recall guarantees) using {metric.upper()} on {label.upper()} ---")
    print(f"Dataset: {N} vectors, {D} dimensions, {X.shape[0]} queries, K={K}")

    # Set the appropriate device (CPU/GPU)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Optimization: Convert data to tensors and move to device once
    # This avoids repeated data transfers during processing
    if torch.is_tensor(A):
        A_tensor = A.to(device)
    else:
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    # Test different target recall rates (percentage of true nearest neighbors found)
    recall_targets = [0.7]  # Can be expanded to test multiple target recalls
    num_test_queries = min(10, X.shape[0])

    for target_recall in recall_targets:
        print(f"\n=== Testing with target recall: {target_recall:.2f} ===")

        total_actual_recall = 0.0
        total_ann_time = 0.0
        total_knn_time = 0.0

        # Optimization: Determine optimal number of clusters based on dataset size
        num_clusters = min(max(int(np.sqrt(N)), 10), 200)

        # Test multiple queries to get reliable statistics
        for i in range(num_test_queries):
            # Extract single query and ensure it's properly formatted
            if X.ndim > 1:
                query = X[i].reshape(1, -1)
            else:
                query = X.reshape(1, -1)

            # Optimization: Move query to the correct device
            if torch.is_tensor(query):
                query_tensor = query.to(device)
            else:
                query_tensor = torch.tensor(query, dtype=torch.float32, device=device)

            # Run improved ANN with recall guarantee
            start = time.time()
            ann_indices = our_ann_improved(N, D, A_tensor, query_tensor[0], K,
                                         num_clusters=num_clusters,
                                         desired_recall=target_recall,
                                         metric=metric, use_gpu=use_gpu)

            # Ensure timing captures all GPU operations
            if device == "cuda":
                torch.cuda.synchronize()

            ann_time = time.time() - start
            total_ann_time += ann_time

            # Run exact KNN for comparison (ground truth)
            start = time.time()
            knn_indices = our_knn(N, D, A_tensor, query_tensor, K, metric=metric)

            if device == "cuda":
                torch.cuda.synchronize()

            knn_time = time.time() - start
            total_knn_time += knn_time

            # Calculate actual recall by comparing ANN results with exact KNN results
            if torch.is_tensor(knn_indices):
                knn_set = set(knn_indices[0].cpu().tolist())
            else:
                knn_set = set(knn_indices[0].tolist())

            ann_set = set(ann_indices)
            correct = len(ann_set.intersection(knn_set))
            actual_recall = correct / K if K > 0 else 0
            total_actual_recall += actual_recall

            print(f"Query {i+1}: Recall={actual_recall:.2f} ({correct}/{K}), "
                  f"ANN: {ann_time:.4f}s, KNN: {knn_time:.4f}s")

        # Calculate and report average statistics
        avg_recall = total_actual_recall / num_test_queries
        avg_ann_time = total_ann_time / num_test_queries
        avg_knn_time = total_knn_time / num_test_queries
        avg_speedup = avg_knn_time / avg_ann_time if avg_ann_time > 0 else 0

        print(f"\nTarget recall: {target_recall:.2f}")
        print(f"Actual average recall: {avg_recall:.2f}")
        print(f"Average times - ANN: {avg_ann_time:.4f}s, KNN: {avg_knn_time:.4f}s")
        print(f"Average speedup: {avg_speedup:.2f}x")

        # Verify if recall guarantee was met
        if avg_recall >= target_recall:
            print(f"✓ SUCCESS: Average recall ({avg_recall:.2f}) >= Target recall ({target_recall:.2f})")
        else:
            print(f"✗ FAILED: Average recall ({avg_recall:.2f}) < Target recall ({target_recall:.2f})")

def test_ann_with_guarantees(use_gpu):
    """
    Run comprehensive tests for the improved ANN implementation across different datasets
    """
    import torch

    print("\n======= Testing ANN with Mathematical Recall Guarantees =======")

    label = "gpu" if use_gpu else "cpu"

    # Set up device environment
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()  # Clear GPU memory before tests

    # Test on multiple dataset sizes
    for size in ["small", "large"]:
        print(f"\n--- ANN Test on {size.upper()} dataset ---")
        N, D, A, X, K = generate_data(size, for_knn=True)

        # Optimization: Pre-convert data to tensors and move to device once
        # This avoids repeated data transfers during benchmarking
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        # Test different distance metrics
        for metric in ["L2", "cosine"]:
            try:
                run_ann_with_guarantees(N, D, A_tensor, X_tensor, K, metric, use_gpu, label)
            except Exception as e:
                print(f"[ANN] Error with {metric.upper()} on {size} dataset: {e}")
                import traceback
                traceback.print_exc()

def main():
    """
    Main function to run tests based on command line argument
    """
    if len(sys.argv) != 2 or sys.argv[1] not in ["gpu", "cpu", "comparison", "all"]:
        print("Usage: python test.py [gpu|cpu|comparison|all]")
        return

    mode = sys.argv[1]

    if mode == "gpu" or mode == "all":
        # Run tests on GPU
        use_gpu = True
        label = "gpu"

        # Set up GPU environment if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Default to CUDA tensors
        else:
            device = torch.device("cpu")
            print("Warning: GPU requested but not available. Using CPU.")
            torch.set_default_tensor_type('torch.FloatTensor')

        metrics = ["L2", "cosine", "dot", "manhattan"]

        # Test KNN on different dataset sizes
        for size in ["medium", "large_dim2", "large_dim32k"]:
            print(f"\n======= Testing on {size.upper()} dataset =======")

            # Generate data for KNN
            N, D, A, X, K = generate_data(size, for_knn=True)
            for metric in metrics:
                try:
                    run_knn(N, D, A, X, K, metric, use_gpu, label)
                except Exception as e:
                    print(f"[KNN] Skipped {metric.upper()} due to error: {e}")

        # Test K-Means on GPU
        for size in ["small", "large"]:
            print(f"\n======= [KMeans] Testing on {size.upper()} dataset =======")
            N, D, A, K = generate_data(size)
            for metric in ["L2", "cosine"]:
                run_kmeans(N, D, A, K, metric, use_gpu, label)

        # Test ANN with recall guarantees on GPU
        test_ann_with_guarantees(use_gpu=True)

    if mode == "cpu" or mode == "all":
        # Run tests on CPU
        use_gpu = False
        label = "cpu"

        device = torch.device("cpu")
        print("Using CPU")
        torch.set_default_tensor_type('torch.FloatTensor')

        metrics = ["L2", "cosine", "dot", "manhattan"]

        # Test KNN on different dataset sizes
        for size in ["medium", "large_dim2", "large_dim32k"]:
            print(f"\n======= Testing on {size.upper()} dataset =======")

            # Generate data for KNN
            N, D, A, X, K = generate_data(size, for_knn=True)
            for metric in metrics:
                try:
                    run_knn(N, D, A, X, K, metric, use_gpu, label)
                except Exception as e:
                    print(f"[KNN] Skipped {metric.upper()} due to error: {e}")

        # Test K-Means on CPU
        for size in ["small", "large"]:
            print(f"\n======= [KMeans] Testing on {size.upper()} dataset =======")
            N, D, A, K = generate_data(size)
            for metric in ["L2", "cosine"]:
                run_kmeans(N, D, A, K, metric, use_gpu, label)

        # Test ANN with recall guarantees on CPU
        test_ann_with_guarantees(use_gpu=False)

    if mode == "comparison" or mode == "all":
      compare_all()
      extrapolate_large_dataset()

if __name__ == "__main__":
    main()
