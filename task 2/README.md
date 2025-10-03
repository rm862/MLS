# RAG System with Batched Processing

This code contains a Retrieval-Augmented Generation (RAG) system with batched processing capabilities, along with a load testing tool to evaluate its performance.

## Overview

The system consists of two main components:
1. **RAG Server (`serving_rag.py`)**: A FastAPI application that provides endpoints for question answering using a RAG approach.
2. **Load Testing Tool (`load_test.py`)**: A script to evaluate the performance of the RAG server under various loads.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- aiohttp
- matplotlib
- numpy
- pydantic

## Installation

```bash

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers fastapi uvicorn pydantic aiohttp matplotlib numpy
```

## Setting Up the Models

Before running the server, you need to download or specify the paths to the models:

1. Embedding model (E5)
2. Language model (Qwen)

Update the model paths in `serving_rag.py`:

```python
EMBED_MODEL_NAME = "/path/to/your/e5_model"
chat_pipeline = pipeline("text-generation", model="/path/to/your/qwen_model", truncation=True)
```

## Running the System

### Step 1: Start the RAG Server

First, start the RAG server by running:

```bash
python serving_rag.py
```

This will start the FastAPI application on http://0.0.0.0:8000. The server provides two endpoints:
- `/rag`: Uses batched processing for improved throughput
- `/rag_no_batch`: Processes each request individually (for comparison)

### Step 2: Run the Load Test

While the server is running, open a new terminal and run the load testing script:

```bash
python load_test.py
```

The load test will:
1. Send requests to both the batched and non-batched endpoints
2. Test different request rates (RPS)
3. Generate performance metrics
4. Create comparison plots
5. Save the results to JSON and PNG files

## Configuration

You can modify the following parameters in `load_test.py`:

```python
base_url = "http://localhost:8000"  # Change if your server is running on a different host/port
duration = 10  # Test duration in seconds
rps_list = [0.5, 1, 2, 5, 10]  # Request rates to test
query = "Which animals can hover in the air?"  # Example query
```

## Understanding the Results

After running the load test, you'll get:
- Console output showing metrics for each test run
- A comparison plot (`rag_compare_[timestamp].png`) showing:
  - Average latency vs. RPS
  - Throughput vs. RPS
- A JSON file (`rag_compare_[timestamp].json`) with detailed metrics

## System Architecture

### RAG Server Features

- **Embedding-based Retrieval**: Uses the E5 model to create document and query embeddings
- **In-memory Document Store**: Comes with example documents that can be expanded
- **Batched Processing**: Aggregates requests to process multiple queries at once
- **Configurable Parameters**: Adjustable batch size and waiting time

### Load Testing Tool Features

- **Async Processing**: Uses `asyncio` and `aiohttp` for efficient request handling
- **Multiple RPS Testing**: Tests various request rates
- **Performance Metrics**: Measures latency, throughput, and percentiles
- **Visualization**: Generates comparison plots

## Important Notes

1. **Sequential Execution**: The RAG server (`serving_rag.py`) must be started and running before executing the load test.
2. **Resource Requirements**: Running the models requires sufficient RAM and potentially GPU resources, depending on model size.
3. **Timeouts**: Adjust timeout settings in `load_test.py` if you encounter timeout errors with larger models or slower hardware.

## Advanced Configuration

- **Batch Processing Parameters**: In `serving_rag.py`, you can adjust:
  ```python
  MAX_BATCH_SIZE = 4      # Maximum requests per batch
  MAX_WAITING_TIME = 1    # Maximum waiting time in seconds
  ```

- **Model Generation Parameters**: Modify the parameters in `chat_pipeline` call:
  ```python
  chat_pipeline(prompt, max_length=50, do_sample=True, truncation=True)
  ```