import time
import json
import requests
from typing import Dict, Any
import statistics

class PerformanceTestRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize performance test runner
        
        :param base_url: Base URL for the API endpoints
        """
        self.base_url = base_url
        self.results: Dict[str, Any] = {
            "timestamp": time.time(),
            "tests": {}
        }
    
    def _measure_endpoint_performance(self, endpoint: str, method: str = "GET", payload: Dict = None) -> Dict[str, float]:
        """
        Measure performance of a specific endpoint
        
        :param endpoint: API endpoint to test
        :param method: HTTP method (GET, POST, etc.)
        :param payload: Optional payload for POST/PUT requests
        :return: Performance metrics dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        # Warm-up request
        requests.request(method, url, json=payload)
        
        # Measure performance
        response_times = []
        for _ in range(10):  # 10 iterations
            start_time = time.time()
            response = requests.request(method, url, json=payload)
            end_time = time.time()
            
            response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            "mean_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "response_time_std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def run_performance_tests(self):
        """
        Run performance tests for different API endpoints
        """
        # Test image upload endpoint
        self.results["tests"]["image_upload"] = self._measure_endpoint_performance(
            "/upload", 
            method="POST", 
            payload={"test_image": "base64_encoded_test_image"}
        )
        
        # Test text extraction endpoint
        self.results["tests"]["text_extraction"] = self._measure_endpoint_performance(
            "/extract-text", 
            method="POST", 
            payload={"image": "base64_encoded_test_image"}
        )
        
        # Test performance metrics endpoint
        self.results["tests"]["performance_metrics"] = self._measure_endpoint_performance(
            "/performance-metrics"
        )
    
    def save_results(self, filename: str = "performance_results.json"):
        """
        Save performance test results to a JSON file
        
        :param filename: Output filename for results
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Performance test results saved to {filename}")
    
    def analyze_results(self):
        """
        Analyze and print performance test results
        """
        print("\n--- Performance Test Results ---")
        for endpoint, metrics in self.results["tests"].items():
            print(f"\nEndpoint: {endpoint}")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.2f} ms")

def main():
    runner = PerformanceTestRunner()
    runner.run_performance_tests()
    runner.save_results()
    runner.analyze_results()

if __name__ == "__main__":
    main()
