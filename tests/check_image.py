from PIL import Image
import os

def check_image(image_path):
    try:
        with Image.open(image_path) as img:
            print(f"\nImage: {os.path.basename(image_path)}")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            print(f"Info: {img.info}")
    except Exception as e:
        print(f"Error with {image_path}: {str(e)}")

# Check specific test images
test_images = [
    "sample_design.png",
    "sample_design.webp",
    "image.png"
]

for image_name in test_images:
    image_path = os.path.join(os.path.dirname(__file__), "test_images", image_name)
    check_image(image_path)
