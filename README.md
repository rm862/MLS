# Machine Learning Systems - Course Project

This repository contains implementations for two major tasks in machine learning systems: efficient vector operations with GPU acceleration and a production-ready RAG (Retrieval-Augmented Generation) system.

## Repository Structure

```
.
├── task 1/
│   ├── task.py          # Core implementations for KNN, K-Means, and ANN
│   ├── test.py          # Comprehensive test suit
    ├── MLS.ipynb        # Jupyter Source File
│   └── README.md        # Detailed usage instructions for Task 1
├── task 2/
│   ├── serving_rag.py   # FastAPI server with batched inference
│   ├── load_test.py     # Performance testing and benchmarking
│   └── README.md        # Detailed usage instructions for Task 2
├── MLS Report.pdf       # Comprehensive Report Submitted
├── .gitignore
├── LICENSE
└── README.md
```

## Task 1: GPU-Accelerated Vector Operations

Implementation of efficient nearest neighbor search, clustering, and approximate search algorithms with PyTorch GPU acceleration.

### Key Features

- **Distance Metrics**: L2, Cosine, Dot Product, Manhattan
- **K-Nearest Neighbors**: Exact search with GPU acceleration and memory-efficient batching
- **K-Means Clustering**: K-Means++ initialization with L2 and cosine metrics
- **Approximate NN**: Mathematical recall guarantees using cluster-based search (formula: L ≥ (K-1)/(1-r))

### Performance Highlights

- 10,000 vectors (D=2): < 0.1s
- 1,000 vectors (D=32,768): ~5s
- Automatic batching for datasets >100k vectors
- ANN speedup: 5-20x over exact KNN with 70% recall guarantee

**See `task 1/README.md` for detailed usage instructions and examples.**

## Task 2: RAG System with Dynamic Batching

Production-ready RAG (Retrieval-Augmented Generation) system with efficient request batching for improved throughput.

### Key Features

- **FastAPI Server**: REST API with automatic documentation
- **Dynamic Request Batching**: Queue-based aggregation (max batch size: 4, max wait: 1s)
- **Dual-Mode Processing**: Batched (`/rag`) and non-batched (`/rag_no_batch`) endpoints
- **E5 Embeddings + Qwen LLM**: Semantic search with context-aware generation
- **Load Testing Framework**: Async testing with performance visualization

### Performance Characteristics

- Batched mode: Lower latency at high RPS, better throughput
- Non-batched mode: Faster single-request response at low RPS
- Crossover point: ~2-5 RPS where batching becomes beneficial

**See `task 2/README.md` for setup instructions, configuration details, and usage examples.**

## Quick Start

### Task 1
```bash
cd "task 1"
pip install torch numpy
python test.py gpu  # or cpu, comparison, all
```

### Task 2
```bash
cd "task 2"
pip install torch transformers fastapi uvicorn pydantic aiohttp matplotlib numpy

# Update model paths in serving_rag.py, then:
python serving_rag.py  # Terminal 1
python load_test.py    # Terminal 2 (after server starts)
```

## Requirements Summary

**Task 1:**
- Python 3.6+
- PyTorch 1.8+ (with CUDA for GPU)
- NumPy

**Task 2:**
- Python 3.8+
- PyTorch, Transformers
- FastAPI, Uvicorn, Pydantic
- aiohttp, matplotlib, NumPy

## Key Optimizations

### Task 1
- Automatic GPU/CPU device selection
- Memory-efficient batching for large matrices
- Numerical stability (clamping, epsilon values)
- Smart parameter auto-tuning for ANN

### Task 2
- Time-bounded request batching with background worker thread
- Non-blocking async request handling
- Pre-computed document embeddings
- Per-request error isolation

## Documentation

Each task folder contains a detailed README with:
- Complete usage instructions
- Code examples
- Configuration options
- Performance tuning guides
- Troubleshooting tips

## License

See LICENSE file for details.

## Notes

- GPU acceleration requires CUDA-capable device
- Task 2 requires pre-downloaded models (E5 and Qwen)
- Refer to task-specific READMEs for detailed setup and usage
