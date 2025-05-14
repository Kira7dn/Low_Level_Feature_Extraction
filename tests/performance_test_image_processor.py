import io
import time
import pytest
import numpy as np
from PIL import Image
import cv2

from app.services.image_processor import ImageProcessor
from app.frontend_utils import ImageCdnManager

def generate_test_images():
    """Generate test images of different sizes and types"""
    # PIL Image
    pil_small = Image.new('RGB', (500, 500), color='red')
    pil_large = Image.new('RGB', (3000, 3000), color='blue')
    
    # OpenCV Image
    cv2_small = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2_small[:] = [0, 255, 0]  # Green image
    cv2_large = np.zeros((3000, 3000, 3), dtype=np.uint8)
    cv2_large[:] = [255, 0, 0]  # Red image
    
    return {
        'pil_small': pil_small,
        'pil_large': pil_large,
        'cv2_small': cv2_small,
        'cv2_large': cv2_large
    }

def test_image_processing_performance():
    """Performance benchmark for image processing methods"""
    test_images = generate_test_images()
    
    # Performance tracking
    performance_results = {}
    
    # Test auto_process_image for different image types and sizes
    for name, image in test_images.items():
        # Convert PIL images to bytes
        if isinstance(image, Image.Image):
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')
            image_bytes = byte_arr.getvalue()
        else:
            # For OpenCV images
            image_bytes = cv2.imencode('.png', image)[1].tobytes()
        
        # Measure processing time
        start_time = time.time()
        processed_image = ImageProcessor.auto_process_image(
            image_bytes, 
            max_width=1920, 
            max_height=1080, 
            target_format='webp'
        )
        end_time = time.time()
        
        # Store performance results
        performance_results[name] = {
            'processing_time': end_time - start_time,
            'input_size': len(image_bytes),
            'output_size': len(processed_image)
        }
    
    # Performance assertions
    for name, result in performance_results.items():
        print(f"\nPerformance for {name}:")
        print(f"Processing Time: {result['processing_time']:.4f} seconds")
        print(f"Input Size: {result['input_size']} bytes")
        print(f"Output Size: {result['output_size']} bytes")
        print(f"Size Reduction: {(1 - result['output_size'] / result['input_size']) * 100:.2f}%")
        
        # Performance expectations
        assert result['processing_time'] < 2.0, f"Processing {name} took too long"
        assert result['output_size'] < result['input_size'], f"Output size should be smaller for {name}"

def test_cdn_url_generation_performance():
    """Performance benchmark for CDN URL generation"""
    cdn_manager = ImageCdnManager('https://cdn.example.com/images')
    
    # Measure URL generation time
    start_time = time.time()
    
    # Generate multiple URLs
    urls = [
        cdn_manager.get_optimized_image_url(f'/path/to/image{i}.jpg') 
        for i in range(100)
    ]
    
    end_time = time.time()
    
    # Performance assertions
    processing_time = end_time - start_time
    print(f"\nCDN URL Generation Performance:")
    print(f"Total URLs Generated: {len(urls)}")
    print(f"Total Processing Time: {processing_time:.4f} seconds")
    print(f"Average Time per URL: {processing_time / len(urls):.6f} seconds")
    
    assert processing_time < 1.0, "URL generation took too long"
    assert len(urls) == 100, "Failed to generate expected number of URLs"
