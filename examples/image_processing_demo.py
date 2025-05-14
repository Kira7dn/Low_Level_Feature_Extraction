import os
import io
import cv2
import numpy as np
from PIL import Image

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.services.image_processor import ImageProcessor
from app.frontend_utils import ImageCdnManager

def generate_sample_images():
    """Generate sample images for demonstration"""
    # Create directory for sample images
    os.makedirs('sample_images', exist_ok=True)
    
    # PIL Image (PNG)
    pil_img = Image.new('RGB', (2000, 2000), color='red')
    pil_img.save('sample_images/large_red_image.png')
    
    # OpenCV Image (JPEG)
    cv2_img = np.zeros((3000, 3000, 3), dtype=np.uint8)
    cv2_img[:] = [0, 255, 0]  # Green image
    cv2.imwrite('sample_images/large_green_image.jpg', cv2_img)

def demonstrate_image_processing():
    """Demonstrate various image processing techniques"""
    # Load sample images
    with open('sample_images/large_red_image.png', 'rb') as f:
        pil_image_bytes = f.read()
    
    with open('sample_images/large_green_image.jpg', 'rb') as f:
        cv2_image_bytes = f.read()
    
    # 1. Basic Image Processing
    print("\n1. Basic Image Processing")
    basic_processed = ImageProcessor.auto_process_image(pil_image_bytes)
    processed_img = Image.open(io.BytesIO(basic_processed))
    print(f"Basic Processed Image - Format: {processed_img.format}")
    print(f"Basic Processed Image - Size: {processed_img.size}")
    processed_img.save('sample_images/basic_processed.webp')
    
    # 2. Custom Image Processing
    print("\n2. Custom Image Processing")
    custom_processed = ImageProcessor.auto_process_image(
        cv2_image_bytes, 
        max_width=800,      # Resize to max 800px width
        max_height=600,     # Resize to max 600px height
        target_format='webp',  # Convert to WebP format
        quality=90          # High-quality compression
    )
    custom_processed_img = Image.open(io.BytesIO(custom_processed))
    print(f"Custom Processed Image - Format: {custom_processed_img.format}")
    print(f"Custom Processed Image - Size: {custom_processed_img.size}")
    custom_processed_img.save('sample_images/custom_processed.webp')
    
    # 3. CDN URL Generation
    print("\n3. CDN URL Generation")
    cdn_manager = ImageCdnManager('https://cdn.example.com')
    
    # Generate optimized image URL
    cdn_url = cdn_manager.get_optimized_image_url(
        '/path/to/image.jpg', 
        {'resize': {'width': 800, 'height': 600}}
    )
    print(f"Generated CDN URL: {cdn_url}")
    
    # 4. Responsive Image URLs
    print("\n4. Responsive Image URLs")
    responsive_urls = cdn_manager.get_responsive_image_urls('/path/to/image.jpg')
    for size, url in responsive_urls.items():
        print(f"{size.capitalize()} Image URL: {url}")

def main():
    generate_sample_images()
    demonstrate_image_processing()

if __name__ == '__main__':
    main()
