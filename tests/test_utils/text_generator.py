"""Generate test images for text extraction testing."""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Create test images directory
test_images_dir = Path(__file__).parent / 'test_images'
test_images_dir.mkdir(exist_ok=True)

# Test cases with their text content
test_cases = [
    {
        'name': 'simple_text',
        'text': 'Simple Test Text',
        'font_size': 40,
        'bg_color': (255, 255, 255),
        'text_color': (0, 0, 0)
    },
    {
        'name': 'numbers',
        'text': '1234567890',
        'font_size': 50,
        'bg_color': (255, 255, 255),
        'text_color': (0, 0, 0)
    },
    {
        'name': 'mixed_case',
        'text': 'Mixed Case TeXt',
        'font_size': 40,
        'bg_color': (255, 255, 255),
        'text_color': (0, 0, 0)
    },
    {
        'name': 'special_chars',
        'text': 'Special: !@#$%^&*()_+',
        'font_size': 40,
        'bg_color': (255, 255, 255),
        'text_color': (0, 0, 0)
    }
]

def generate_image(text, font_size, bg_color, text_color, output_path):
    """Generate an image with the given text and save it to the output path."""
    # Create a temporary image to calculate text size
    temp_img = Image.new('RGB', (1, 1), bg_color)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create the actual image with padding
    padding = 20
    width = text_width + 2 * padding
    height = text_height + 2 * padding
    
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw the text centered
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Save the image
    img.save(output_path, dpi=(300, 300))
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    # Generate all test images
    for case in test_cases:
        output_path = test_images_dir / f"{case['name']}.png"
        generate_image(
            text=case['text'],
            font_size=case['font_size'],
            bg_color=case['bg_color'],
            text_color=case['text_color'],
            output_path=output_path
        )
    
    print("\nTest images generated successfully!")
