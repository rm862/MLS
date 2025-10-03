import asyncio
import aiohttp
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

async def send_request(session, url, payload):
    """
    Send an HTTP POST request and record the latency from sending the request
    to receiving the response. Return a tuple (latency, response_json).
    """
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=120) as response:
            resp_json = await response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        resp_json = None
    end = time.perf_counter()
    return end - start, resp_json

async def run_load_test(url, rps, duration, payload):
    """
    Send requests sequentially based on the target request rate (rps) and the test duration.
    Record performance metrics including total requests, total test duration,
    average latency, 90th percentile latency, 99th percentile latency and achieved throughput.
    Return a dictionary containing these metrics.
    """
    total_requests = int(rps * duration)
    tasks = []
    latencies = []

    async with aiohttp.ClientSession() as session:
        test_start = time.perf_counter()
        for i in range(total_requests):
            task = asyncio.create_task(send_request(session, url, payload))
            tasks.append(task)
            await asyncio.sleep(1.0 / rps)
        results = await asyncio.gather(*tasks)
        test_end = time.perf_counter()

    for latency, _ in results:
        latencies.append(latency)

    total_time = test_end - test_start
    request_count = len(latencies)
    avg_latency = sum(latencies) / request_count if request_count > 0 else 0.0
    sorted_latencies = sorted(latencies)
    p90 = sorted_latencies[int(0.9 * request_count) - 1] if request_count >= 10 else avg_latency
    p99 = sorted_latencies[int(0.99 * request_count) - 1] if request_count >= 100 else avg_latency
    throughput = request_count / total_time if total_time > 0 else 0.0

    return {
        "rps": rps,
        "duration": duration,
        "total_requests": request_count,
        "total_time": total_time,
        "avg_latency_ms": avg_latency * 1000,
        "p90_latency_ms": p90 * 1000,
        "p99_latency_ms": p99 * 1000,
        "throughput": throughput
    }

async def multi_run_tests(base_url, endpoint, rps_list, duration, payload):
    """
    Run load tests for each rps value in rps_list to the specified endpoint (e.g. /rag or /rag_no_batch).
    Return a list of result dictionaries.
    """
    results = []
    full_url = base_url.rstrip("/") + endpoint
    for rps in rps_list:
        print(f"Testing endpoint {endpoint} with rps={rps}, duration={duration}")
        res = await run_load_test(full_url, rps, duration, payload)
        results.append(res)
        # Small delay between tests
        await asyncio.sleep(2)
    return results

def plot_compare_results(batched_results, non_batched_results, output_png):
    """
    Plot comparison of batched and non-batched results for different rps values.
    Produce two subplots: one for average latency and one for throughput.
    Save the plot to output_png.
    """
    rps_batched = [r["rps"] for r in batched_results]
    latency_batched = [r["avg_latency_ms"] for r in batched_results]
    throughput_batched = [r["throughput"] for r in batched_results]

    rps_non_batched = [r["rps"] for r in non_batched_results]
    latency_non_batched = [r["avg_latency_ms"] for r in non_batched_results]
    throughput_non_batched = [r["throughput"] for r in non_batched_results]

    plt.figure(figsize=(12, 6))

    # Subplot for average latency
    plt.subplot(1, 2, 1)
    plt.plot(rps_batched, latency_batched, marker='o', label='Batched')
    plt.plot(rps_non_batched, latency_non_batched, marker='o', label='Non-Batched')
    plt.xlabel('RPS')
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Latency vs RPS')
    plt.grid(True)
    plt.legend()

    # Subplot for throughput
    plt.subplot(1, 2, 2)
    plt.plot(rps_batched, throughput_batched, marker='o', label='Batched')
    plt.plot(rps_non_batched, throughput_non_batched, marker='o', label='Non-Batched')
    plt.xlabel('RPS')
    plt.ylabel('Throughput (req/s)')
    plt.title('Throughput vs RPS')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Comparison plot saved to {output_png}")

def main():
    # Set base URL and fixed duration/query/k
    base_url = "http://localhost:8000"
    # Endpoints: /rag is batched, /rag_no_batch is non-batched
    batched_endpoint = "/rag"
    non_batched_endpoint = "/rag_no_batch"
    
    duration = 10  # seconds
    # Define rps values to test
    rps_list = [0.5, 1, 2, 5, 10]
    query = "Which animals can hover in the air?"
    k = 2
    payload = {"query": query, "k": k}
    
    # Run tests for both batched and non-batched modes
    loop = asyncio.get_event_loop()
    print("Running batched tests...")
    batched_results = loop.run_until_complete(multi_run_tests(base_url, batched_endpoint, rps_list, duration, payload))
    print("Running non-batched tests...")
    non_batched_results = loop.run_until_complete(multi_run_tests(base_url, non_batched_endpoint, rps_list, duration, payload))
    
    # Print results
    print("\nBatched Results:")
    for res in batched_results:
        print(res)
    print("\nNon-Batched Results:")
    for res in non_batched_results:
        print(res)
    
    # Generate comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_png = f"rag_compare_{timestamp}.png"
    plot_compare_results(batched_results, non_batched_results, output_png)
    
    # Save results to a JSON file
    output_json = f"rag_compare_{timestamp}.json"
    results_output = {
        "batched": batched_results,
        "non_batched": non_batched_results
    }
    with open(output_json, 'w') as f:
        json.dump(results_output, f, indent=2, default=str)
    print(f"Detailed comparison results saved to {output_json}")

if __name__ == "__main__":
    main()
