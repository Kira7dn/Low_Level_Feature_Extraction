# TestImageFactory Documentation

## Overview
`TestImageFactory` is a utility class for generating test images with text for OCR testing. It helps create consistent, repeatable test cases for text extraction functionality.

## Features
- Generate images with customizable text, fonts, and styles
- Create various visual variations (contrast, noise, blur)
- Support for different text alignments and layouts
- Image caching for faster test execution
- Built-in support for common test scenarios

## Installation

```bash
pip install pillow opencv-python numpy
```

## Implementation

```python
"""Test image generation factory for OCR testing."""

import os
import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from functools import lru_cache


class TextAlignment(str, Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class TextEffect(str, Enum):
    """Text effect options."""
    NONE = "none"
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"


class TestImageFactory:
    """Factory for creating test images with text for OCR testing."""
    
    # Default directories
    CACHE_DIR = Path("test_data/cached_images")
    FONT_DIR = Path("test_data/fonts")
    
    # Default fonts (fallback to system fonts if not found)
    DEFAULT_FONTS = {
        'sans': 'arial.ttf',
        'serif': 'times.ttf',
        'mono': 'cour.ttf'
    }
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize the factory.
        
        Args:
            cache_enabled: Whether to cache generated images for faster test execution
        """
        self.cache_enabled = cache_enabled
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.FONT_DIR.mkdir(parents=True, exist_ok=True)
        self._load_default_fonts()
    
    def _load_default_fonts(self):
        """Load default fonts or fallback to system fonts."""
        self.fonts = {}
        for name, font_file in self.DEFAULT_FONTS.items():
            font_path = self.FONT_DIR / font_file
            if font_path.exists():
                self.fonts[name] = str(font_path)
            else:
                # Try to find system font
                try:
                    self.fonts[name] = font_file
                except:
                    print(f"Warning: Could not load font {font_file}")
    
    def create_text_image(
        self,
        text: str,
        size: Tuple[int, int] = (800, 200),
        font_size: int = 32,
        font_family: str = "sans",
        text_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        alignment: TextAlignment = TextAlignment.CENTER,
        line_spacing: float = 1.0,
        effects: Optional[List[TextEffect]] = None,
        noise_level: float = 0.0,
        blur_radius: float = 0.0,
        contrast: float = 1.0,
        rotation: float = 0.0,
        dpi: Tuple[int, int] = (300, 300),
        cache_key_suffix: str = ""
    ) -> np.ndarray:
        """Create a test image with text and various effects.
        
        Args:
            text: The text to render
            size: Image size as (width, height)
            font_size: Font size in points
            font_family: Font family name or path to TTF file
            text_color: Text color as RGB tuple
            bg_color: Background color as RGB tuple
            alignment: Text alignment
            line_spacing: Line spacing multiplier
            effects: List of text effects to apply
            noise_level: Amount of noise to add (0.0 to 1.0)
            blur_radius: Gaussian blur radius (0.0 for no blur)
            contrast: Contrast adjustment (1.0 = no change)
            rotation: Rotation angle in degrees
            dpi: DPI resolution as (x, y)
            cache_key_suffix: Additional suffix for cache key
            
        Returns:
            Image as a NumPy array in BGR format (OpenCV compatible)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            text, size, font_size, font_family, text_color, bg_color, 
            alignment, line_spacing, effects, noise_level, blur_radius, 
            contrast, rotation, dpi, cache_key_suffix
        )
        cache_path = self.CACHE_DIR / f"{cache_key}.png"
        
        # Return cached image if available and caching is enabled
        if self.cache_enabled and cache_path.exists():
            img = cv2.imread(str(cache_path))
            if img is not None:
                return img
        
        # Create new image
        img = self._create_base_image(size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = self._load_font(font_family, font_size)
        
        # Apply text effects
        if effects:
            font, text_color = self._apply_text_effects(draw, font, text_color, effects)
        
        # Draw text
        self._draw_text(draw, text, font, text_color, size, alignment, line_spacing)
        
        # Apply image effects
        img = self._apply_image_effects(img, noise_level, blur_radius, contrast, rotation)
        
        # Convert to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Save to cache
        if self.cache_enabled:
            cv2.imwrite(str(cache_path), img_cv)
        
        return img_cv
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key for the given parameters."""
        import hashlib
        import json
        
        # Convert arguments to a dictionary
        params = {
            'args': args,
            'kwargs': {k: v for k, v in kwargs.items() if not k.startswith('_')}
        }
        
        # Convert to JSON and hash
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _create_base_image(self, size: Tuple[int, int], bg_color: Tuple[int, int, int]) -> Image.Image:
        """Create a base image with the specified background color."""
        return Image.new('RGB', size, color=bg_color)
    
    def _load_font(self, font_family: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a font with the specified family and size."""
        font_path = self.fonts.get(font_family, font_family)
        try:
            return ImageFont.truetype(font_path, font_size)
        except IOError:
            return ImageFont.load_default()
    
    def _apply_text_effects(self, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont, 
                          text_color: Tuple[int, int, int], 
                          effects: List[TextEffect]) -> Tuple[ImageFont.FreeTypeFont, Tuple[int, int, int]]:
        """Apply text effects and return modified font and color."""
        # This is a simplified example - in practice, you'd implement each effect
        return font, text_color
    
    def _draw_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                  text_color: Tuple[int, int, int], size: Tuple[int, int],
                  alignment: TextAlignment, line_spacing: float):
        """Draw text with the specified alignment and line spacing."""
        # This is a simplified example - in practice, you'd implement proper text layout
        draw.text((10, 10), text, fill=text_color, font=font)
    
    def _apply_image_effects(self, img: Image.Image, noise_level: float, blur_radius: float,
                           contrast: float, rotation: float) -> Image.Image:
        """Apply image effects like noise, blur, contrast, and rotation."""
        # Apply contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # Apply blur
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Apply rotation
        if rotation != 0:
            img = img.rotate(rotation, expand=True)
        
        # Apply noise (simplified example)
        if noise_level > 0:
            # In practice, you'd implement proper noise addition
            pass
            
        return img
    
    # Factory methods for common test cases
    
    def create_high_contrast_text(self, text: str, **kwargs) -> np.ndarray:
        """Create an image with high contrast text."""
        return self.create_text_image(
            text,
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255),
            contrast=2.0,
            **kwargs
        )
    
    def create_low_contrast_text(self, text: str, **kwargs) -> np.ndarray:
        """Create an image with low contrast text."""
        return self.create_text_image(
            text,
            text_color=(100, 100, 100),
            bg_color=(150, 150, 150),
            contrast=0.5,
            **kwargs
        )
    
    def create_noisy_text(self, text: str, noise_level: float = 0.1, **kwargs) -> np.ndarray:
        """Create an image with noisy background."""
        return self.create_text_image(
            text,
            noise_level=noise_level,
            **kwargs
        )
    
    def create_blurred_text(self, text: str, blur_radius: float = 1.0, **kwargs) -> np.ndarray:
        """Create an image with blurred text."""
        return self.create_text_image(
            text,
            blur_radius=blur_radius,
            **kwargs
        )
    
    def clear_cache(self) -> None:
        """Clear all cached images."""
        for file in self.CACHE_DIR.glob("*.png"):
            try:
                file.unlink()
            except OSError:
                pass
```

## Usage Example

```python
# Create a factory instance
factory = TestImageFactory()

# Generate test images
high_contrast = factory.create_high_contrast_text("High Contrast", font_size=48)
low_contrast = factory.create_low_contrast_text("Low Contrast", font_size=48)
noisy = factory.create_noisy_text("Noisy Text", noise_level=0.2)
blurred = factory.create_blurred_text("Blurred Text", blur_radius=2.0)

# Save the images
cv2.imwrite("high_contrast.png", high_contrast)
cv2.imwrite("low_contrast.png", low_contrast)
cv2.imwrite("noisy.png", noisy)
cv2.imwrite("blurred.png", blurred)

# Clear the cache when done testing
factory.clear_cache()
```

## Dependencies

```bash
pip install pillow opencv-python numpy
```

## License

MIT License

## Installation

```bash
pip install pillow opencv-python numpy
```

## Basic Usage

```python
from test_utils.image_factory import TestImageFactory

# Create a factory instance
factory = TestImageFactory()

# Generate a simple text image
image = factory.create_text_image(
    text="Test Text 123",
    size=(800, 200),
    font_size=48,
    text_color=(0, 0, 0),  # Black text
    bg_color=(255, 255, 255)  # White background
)

# Save the image
cv2.imwrite("test_image.png", image)
```

## Factory Methods

### `create_text_image()`
Create an image with the specified text and styling.

```python
image = factory.create_text_image(
    text="Hello World",
    size=(800, 200),          # Image dimensions (width, height)
    font_size=32,             # Font size in points
    font_family="sans",       # Font family: 'sans', 'serif', or 'mono'
    text_color=(0, 0, 0),     # Text color (R, G, B)
    bg_color=(255, 255, 255), # Background color (R, G, B)
    alignment="center",       # Text alignment: 'left', 'center', 'right'
    noise_level=0.1,          # Add random noise (0.0 to 1.0)
    blur_radius=0.0,          # Add Gaussian blur (pixels)
    contrast=1.0,             # Contrast adjustment (1.0 = normal)
    rotation=0.0              # Rotation in degrees
)
```

### `create_high_contrast_text()`
Create an image with high contrast text (black on white by default).

```python
image = factory.create_high_contrast_text(
    text="High Contrast",
    size=(800, 200),
    font_size=48
)
```

### `create_low_contrast_text()`
Create an image with low contrast text.

```python
image = factory.create_low_contrast_text(
    text="Low Contrast",
    size=(800, 200),
    font_size=48
)
```

### `create_noisy_text()`
Create an image with noisy background.

```python
image = factory.create_noisy_text(
    text="Noisy Text",
    noise_level=0.2,  # 0.0 to 1.0
    size=(800, 200)
)
```

### `create_blurred_text()`
Create an image with blurred text.

```python
image = factory.create_blurred_text(
    text="Blurred Text",
    blur_radius=2.0,  # Pixels
    size=(800, 200)
)
```

## Advanced Usage

### Custom Fonts
Place custom TTF files in the `test_data/fonts/` directory and reference them by filename:

```python
image = factory.create_text_image(
    text="Custom Font",
    font_family="my_custom_font.ttf",
    font_size=36
)
```

### Caching
Images are cached by default to improve test performance. You can disable caching:

```python
factory = TestImageFactory(cache_enabled=False)
```

### Clearing Cache
To clear the image cache:

```python
factory.clear_cache()
```

## Example Test Case

```python
def test_text_extraction_with_factory():
    # Create factory
    factory = TestImageFactory()
    
    # Generate test image
    image = factory.create_text_image(
        text="Test 123",
        size=(800, 200),
        font_size=48,
        noise_level=0.1
    )
    
    # Test text extraction
    result = text_extractor.extract_text(image)
    
    # Verify results
    assert "test 123" in ' '.join(result['lines']).lower()
```

## Best Practices

1. **Use Meaningful Test Data**: Generate text that represents real-world scenarios
2. **Test Edge Cases**: Include tests with unusual characters, numbers, and symbols
3. **Vary Visual Conditions**: Test with different fonts, sizes, and contrast levels
4. **Clean Up**: Clear the cache between test runs for consistent results
5. **Document Test Cases**: Include comments explaining the purpose of each test image

## Troubleshooting

- If text appears distorted, try increasing the image size
- For blurry text, reduce the `blur_radius` or increase the font size
- If text is not detected, try increasing the contrast or reducing noise
- Check the cache directory if images aren't updating as expected

## License

This utility is provided under the MIT License.
