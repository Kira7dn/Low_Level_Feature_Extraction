import os
import sys
import time
import logging
import numpy as np
from PIL import Image
import cv2

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import our modules
from app.services.image_processor import ImageProcessor
from app.monitoring.performance import PerformanceMonitor

def generate_test_image(width=1000, height=1000, as_bytes=False):
    """Generate a test image for performance monitoring"""
    # Create a random numpy array image
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    if as_bytes:
        # Encode the image to bytes
        _, encoded_image = cv2.imencode('.png', test_image)
        return encoded_image.tobytes()
    
    return test_image

def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring for image processing methods"""
    print("\n--- Performance Monitoring Demonstration ---")
    
    # Generate test images
    cv2_image = generate_test_image()
    cv2_image_bytes = generate_test_image(as_bytes=True)
    pil_image = Image.fromarray(cv2_image)
    
    # 1. Image Loading Performance
    print("\n1. Image Loading Performance:")
    loaded_cv2_image = ImageProcessor.load_cv2_image(cv2_image_bytes)
    loaded_pil_image = ImageProcessor.load_image(cv2_image_bytes)
    
    # 2. Image Resizing Performance
    print("\n2. Image Resizing Performance:")
    resized_cv2_image = ImageProcessor.resize_image(cv2_image, max_width=500, max_height=500)
    resized_pil_image = ImageProcessor.resize_image(pil_image, max_width=500, max_height=500)
    
    # 3. Image Compression Performance
    print("\n3. Image Compression Performance:")
    compressed_cv2_image = ImageProcessor.compress_image(cv2_image, quality=75)
    compressed_pil_image = ImageProcessor.compress_image(pil_image, quality=75)
    
    # 4. Analyze Performance Metrics
    print("\n4. Performance Metrics Analysis:")
    metrics = {
        'load_cv2_image': PerformanceMonitor.analyze_performance_metrics('load_cv2_image'),
        'load_image': PerformanceMonitor.analyze_performance_metrics('load_image'),
        'resize_image': PerformanceMonitor.analyze_performance_metrics('resize_image'),
        'compress_image': PerformanceMonitor.analyze_performance_metrics('compress_image')
    }
    
    # Print performance metrics
    for method, analysis in metrics.items():
        print(f"\nPerformance Metrics for {method}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    
    # 5. View Raw Performance Log
    print("\n5. Raw Performance Log:")
    log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'performance_metrics.json')
    
    if os.path.exists(log_file):
        import json
        with open(log_file, 'r') as f:
            performance_log = json.load(f)
        
        print(f"Total log entries: {len(performance_log)}")
        print("Last 3 log entries:")
        for entry in performance_log[-3:]:
            print(json.dumps(entry, indent=2))

if __name__ == '__main__':
    demonstrate_performance_monitoring()
