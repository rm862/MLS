## Features

- Efficient KNN implementation with multiple distance metrics (L2, cosine, dot product, Manhattan)
- Memory-optimized K-Means clustering algorithm 
- Approximate Nearest Neighbors (ANN) with recall guarantees
- GPU acceleration via PyTorch for significantly faster processing
- Automatic batching for large datasets
- Comprehensive benchmarking and testing utilities

## Requirements

- Python 3.6+
- PyTorch 1.8+ (with CUDA support for GPU acceleration)
- NumPy

## Installation

### Google Colab

1. Upload the `task.py` and `test.py` files to your Colab session
2. Execute the following to install dependencies:
   ```python
   !pip install torch numpy
   ```

### Local Environment

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install torch numpy
   ```

## Usage

The library provides three main algorithms:

1. `our_knn`: Optimized K-Nearest Neighbors
2. `our_kmeans`: Memory-efficient K-Means clustering
3. `our_ann_improved`: Approximate Nearest Neighbors with recall guarantees

### Running Tests

The `test.py` script provides comprehensive testing capabilities with three modes:

#### GPU Mode

Tests all algorithms with GPU acceleration (if available):

```bash
# In your terminal:
python test.py gpu

# In Google Colab:
!python test.py gpu
```

This will:
- Run KNN benchmarks on various dataset sizes using all distance metrics
- Test K-Means clustering with L2 and cosine distance
- Evaluate ANN with mathematical recall guarantees
- Use GPU acceleration for all computations (falls back to CPU if GPU is unavailable)

#### CPU Mode

Tests all algorithms using CPU only:

```bash
# In your terminal:
python test.py cpu

# In Google Colab:
!python test.py cpu
```

This will run the same suite of tests as GPU mode but explicitly using CPU computation.

#### Comparison Mode

Provides direct CPU vs GPU performance comparisons:

```bash
# In your terminal:
python test.py comparison

# In Google Colab:
!python test.py comparison
```

This will:
- Run comparative benchmarks of CPU vs GPU performance
- Display speedup metrics across different algorithms and dataset sizes
- Extrapolate performance for extremely large datasets (4M vectors)

### Colab Quick Start

Here's a complete example to get started in Google Colab:

```python
# Upload files to Colab
# (drag and drop the task.py and test.py files to the file browser)

# Install dependencies
!pip install torch numpy

# Run GPU tests
!python test.py gpu

# Run CPU tests
!python test.py cpu

# Run comparison tests
!python test.py comparison
```

## Algorithm Details

### KNN Implementation

The KNN algorithm (`our_knn`) supports multiple distance metrics:
- L2 (Euclidean distance)
- Cosine distance
- Dot product distance
- Manhattan distance

Performance optimizations include:
- Batch processing for memory efficiency
- GPU acceleration for distance computations
- Efficient top-k selection algorithm

### K-Means Implementation

The K-Means algorithm (`our_kmeans`) features:
- Optimized centroid updates using scatter operations
- Support for both L2 and cosine metrics
- GPU acceleration for cluster assignments
- Early stopping based on convergence criteria

### ANN Implementation

The Approximate Nearest Neighbors algorithm (`our_ann_improved`) implements:
- Mathematical recall guarantees through analyzed cluster search
- Adaptive cluster selection based on data density
- Optimized candidate selection
- Fallback strategy for edge cases

## Performance Considerations

- For extremely large datasets (>100k vectors), batching is automatically employed
- High-dimensional data (>10k dimensions) benefits significantly from GPU acceleration
- The extrapolation test provides estimates for scaling to millions of vectors
