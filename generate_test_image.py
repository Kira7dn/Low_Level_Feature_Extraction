from PIL import Image, ImageDraw, ImageFont
import os

def create_test_images():
    # Ensure test_images directory exists
    os.makedirs('test_images', exist_ok=True)
    
    # Color and design test image
    color_image = Image.new('RGB', (300, 200), color='white')
    draw = ImageDraw.Draw(color_image)
    draw.rectangle([50, 50, 150, 150], fill='blue')
    draw.rectangle([0, 0, 300, 200], fill='lightgray', outline='lightgray')
    draw.ellipse([200, 50, 250, 100], fill='red')
    draw.ellipse([200, 100, 250, 150], fill='green')
    color_image.save('test_images/sample_design.png')
    
    # Text and font test image
    text_image = Image.new('RGB', (400, 200), color='white')
    text_draw = ImageDraw.Draw(text_image)
    try:
        font = ImageFont.truetype('arial.ttf', 36)
    except IOError:
        font = ImageFont.load_default()
    text_draw.text((50, 50), 'Design AI', font=font, fill='black')
    text_draw.text((50, 100), 'Low-Level Features', font=font, fill='blue')
    text_image.save('test_images/text_sample.png')
    
    # Shapes test image
    shapes_image = Image.new('RGB', (300, 200), color='white')
    shapes_draw = ImageDraw.Draw(shapes_image)
    shapes_draw.rectangle([50, 50, 150, 150], outline='black', width=3)
    shapes_draw.ellipse([200, 50, 250, 100], outline='red', width=3)
    shapes_draw.polygon([(50, 150), (100, 100), (150, 150)], outline='green', width=3)
    shapes_image.save('test_images/shapes_sample.png')
    
    # Shadows test image
    shadows_image = Image.new('RGB', (300, 200), color='white')
    shadows_draw = ImageDraw.Draw(shadows_image)
    shadows_draw.rectangle([50, 50, 150, 150], fill='gray', outline='darkgray')
    shadows_draw.rectangle([60, 60, 140, 140], fill='lightgray', outline='gray')
    shadows_image.save('test_images/shadows_sample.png')
    
    # Fonts test image
    fonts_image = Image.new('RGB', (400, 200), color='white')
    fonts_draw = ImageDraw.Draw(fonts_image)
    try:
        font_bold = ImageFont.truetype('arial.ttf', 36, index=1)  # Bold variant
        font_italic = ImageFont.truetype('arial.ttf', 36, index=2)  # Italic variant
    except IOError:
        font_bold = font_italic = ImageFont.load_default()
    fonts_draw.text((50, 50), 'Bold Text', font=font_bold, fill='black')
    fonts_draw.text((50, 100), 'Italic Text', font=font_italic, fill='blue')
    fonts_image.save('test_images/fonts_sample.png')
    
    print("Test images created successfully!")

if __name__ == '__main__':
    create_test_images()
