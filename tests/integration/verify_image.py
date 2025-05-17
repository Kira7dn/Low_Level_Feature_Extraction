"""Script to verify image files."""
import os
import sys
import cv2

def verify_image(file_path):
    """Verify if the file is a valid image."""
    print(f"\nVerifying image: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    # Try to read the file as bytes
    with open(file_path, 'rb') as f:
        header = f.read(8)  # PNG header is 8 bytes
        print(f"File header (hex): {header.hex()}")
    
    # Check if it's a PNG file
    is_png = header.startswith(b'\x89PNG\r\n\x1a\x0a')
    print(f"Is PNG file: {is_png}")
    
    # Try to load with OpenCV
    img = cv2.imread(file_path)
    print(f"OpenCV loaded: {img is not None}")
    if img is not None:
        print(f"Image dimensions: {img.shape}")
    
    return img is not None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_image.py <image_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    is_valid = verify_image(file_path)
    print(f"\nImage is valid: {is_valid}")
    sys.exit(0 if is_valid else 1)
