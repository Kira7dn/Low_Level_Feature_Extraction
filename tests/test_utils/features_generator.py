"""
Generate test images with various text and style features for testing.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from PIL import Image, ImageDraw, ImageFont
import random

# Create test directory
test_dir = Path(__file__).parent.parent / 'test_data' / 'test_suite'
test_dir.mkdir(parents=True, exist_ok=True)

def _color_to_hex(color_spec) -> str:
    """Convert color specification to hex format."""
    if isinstance(color_spec, str):
        if color_spec.startswith('#'):
            return color_spec
        try:
            # Convert color name to RGB
            from PIL import ImageColor
            return ImageColor.getrgb(color_spec).hex()
        except (ValueError, AttributeError):
            return '#000000'  # Default to black on error
    elif isinstance(color_spec, tuple):
        mode, base_color = color_spec
        try:
            from PIL import ImageColor
            r, g, b = ImageColor.getrgb(base_color)
            if mode == 'darker':
                return f'#{max(0, r-50):02x}{max(0, g-50):02x}{max(0, b-50):02x}'
            elif mode == 'lighter':
                return f'#{min(255, r+50):02x}{min(255, g+50):02x}{min(255, b+50):02x}'
            return f'#{r:02x}{g:02x}{b:02x}'
        except (ValueError, AttributeError):
            return '#000000'  # Default to black on error
    return '#000000'  # Default to black

# Type aliases
Color = Union[str, Tuple[int, int, int]]
Size = Tuple[int, int]
Position = Tuple[int, int]

@dataclass
class ColorConfig:
    """Configuration for color features."""
    primary: Color = "#000000"
    background: Color = "#FFFFFF"
    accents: List[Color] = None
    
    def __post_init__(self):
        if self.accents is None:
            # Generate some default accent colors if none provided
            self.accents = ["#666666", "#999999"]

@dataclass
class TextConfig:
    """Configuration for text features."""
    lines: List[str] = None
    position: Position = (50, 50)
    
    def __post_init__(self):
        if self.lines is None:
            self.lines = ["Sample Text"]

@dataclass
class FontConfig:
    """Configuration for font features."""
    family: str = "Arial"
    size: int = 24
    style: str = "normal"  # normal, bold, italic

@dataclass
class ImageConfig:
    """Configuration for image generation."""
    size: Size = (800, 600)
    output_dir: Path = Path("tests/test_data")
    filename: Optional[str] = None

class SimpleFeatureGenerator:
    """A simple image generator for test data."""
    
    @staticmethod
    def _get_font(font_size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
        """Get a font with the specified size."""
        try:
            if font_path:
                return ImageFont.truetype(font_path, font_size)
            return ImageFont.truetype("arial.ttf", font_size)
        except (IOError, AttributeError):
            return ImageFont.load_default()
    
    @classmethod
    def create_image(
        cls,
        color_config: Optional[ColorConfig] = None,
        text_config: Optional[TextConfig] = None,
        font_config: Optional[FontConfig] = None,
        image_config: Optional[ImageConfig] = None
    ) -> Dict[str, Any]:
        """
        Create an image with specified features.
        
        Args:
            color_config: Color configuration
            text_config: Text configuration
            font_config: Font configuration
            image_config: Image configuration
            
        Returns:
            {
                'image_path': '/path/to/image.png',
                'features': {
                    'colors': {...},
                    'text': {...},
                    'fonts': {...}
                }
            }
        """
        # Apply defaults
        color_config = color_config or ColorConfig()
        text_config = text_config or TextConfig()
        font_config = font_config or FontConfig()
        image_config = image_config or ImageConfig()
        
        # Create output directory
        image_config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create image with background color
        image = Image.new('RGB', image_config.size, color_config.background)
        draw = ImageDraw.Draw(image)
        
        # Draw text if provided
        if text_config.lines:
            font = cls._get_font(font_config.size, font_config.family)
            text = "\n".join(text_config.lines)
            draw.text(text_config.position, text, fill=color_config.primary, font=font)
        
        # Generate filename if not provided
        if not image_config.filename:
            image_config.filename = f"img_{random.randint(1000, 9999)}.png"
        
        # Save image
        image_path = image_config.output_dir / image_config.filename
        image.save(image_path)
        
        # Return the features and image path
        return {
            'image_path': str(image_path.absolute()),
            'features': {
                'colors': {
                    'primary': _color_to_hex(color_config.primary),
                    'background': _color_to_hex(color_config.background),
                    'accent': [_color_to_hex(accent) for accent in color_config.accents]
                },
                'text': {
                    'lines': text_config.lines
                },
                'fonts': {
                    'font_family': font_config.family,
                    'font_size': font_config.size,
                    'font_style': font_config.style
                }
            }
        }

# Test cases with their configurations
test_cases = [
    {
        'name': 'simple_text',
        'features': {
            'colors': {
                'primary': '#000000',
                'background': '#FFFFFF',
                'accent': ['#333333', '#666666']
            },
            'text': {
                'lines': ['Simple Test Text']
            },
            'fonts': {
                'font_family': 'Arial',
                'font_size': 40,
                'font_style': 'normal'
            }
        }
    },
    {
        'name': 'colored_text',
        'features': {
            'colors': {
                'primary': '#1a73e8',
                'background': '#f8f9fa',
                'accent': ['#0d47a1', '#64b5f6']
            },
            'text': {
                'lines': ['Colored Text', 'With Accent Colors']
            },
            'fonts': {
                'font_family': 'Arial',
                'font_size': 32,
                'font_style': 'normal'
            }
        }
    },
    {
        'name': 'dark_theme',
        'features': {
            'colors': {
                'primary': '#ffffff',
                'background': '#202124',
                'accent': ['#e8eaed', '#9aa0a6']
            },
            'text': {
                'lines': ['Dark Theme', 'White on Dark']
            },
            'fonts': {
                'font_family': 'Arial',
                'font_size': 36,
                'font_style': 'normal'
            }
        }
    },
    {
        'name': 'product_card',
        'features': {
            'colors': {
                'primary': '#1e88e5',
                'background': '#f5f5f5',
                'accent': ['#0d47a1', '#64b5f6']
            },
            'text': {
                'lines': [
                    'Product Name: Premium Widget',
                    'Price: $99.99',
                    'In Stock: Yes'
                ]
            },
            'fonts': {
                'font_family': 'Arial',
                'font_size': 28,
                'font_style': 'bold'
            }
        }
    },
    {
        'name': 'documentation',
        'features': {
            'colors': {
                'primary': '#2e7d32',
                'background': '#e8f5e9',
                'accent': ['#1b5e20', '#4caf50']
            },
            'text': {
                'lines': [
                    'Up-to-date documentation for LLMs and AI code editors.',
                    'Copy the latest docs and code for any library —',
                    'paste into Cursor, Claude, or other LLMs.',
                    'Search a library (e.g. Next, React)'
                ]
            },
            'fonts': {
                'font_family': 'Arial',
                'font_size': 20,
                'font_style': 'normal'
            }
        }
    }
]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_image(text_lines, font_config, colors, output_path):
    """Generate an image with the given text and save it to the output path."""
    # Join text lines with newlines
    text = '\n'.join(text_lines)
    
    # Convert hex colors to RGB
    bg_color = hex_to_rgb(colors['background'].lstrip('#'))
    text_color = hex_to_rgb(colors['primary'].lstrip('#'))
    
    # Create a temporary image to calculate text size
    temp_img = Image.new('RGB', (1, 1), bg_color)
    try:
        font = ImageFont.truetype("arial.ttf", font_config['font_size'])
    except IOError:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create the actual image with padding
    padding = 40
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
    # Prepare test suite data
    test_suite = {
        'test_cases': []
    }
    
    # Generate all test images and collect test cases
    for case in test_cases:
        output_path = test_dir / f"{case['name']}.png"
        
        # Generate the image
        generate_image(
            text_lines=case['features']['text']['lines'],
            font_config=case['features']['fonts'],
            colors=case['features']['colors'],
            output_path=output_path
        )
        
        # Add to test suite with full feature structure
        test_case = {
            'name': case['name'],
            'image_path': str(output_path.relative_to(test_dir.parent.parent)),
            'description': case.get('description', case['name'].replace('_', ' ').title()),
            'expected_features': case['features'],
            'status': 'success',
            'errors': {},
            'metadata': {
                'total_features_requested': 3,  # colors, text, fonts
                'features_processed': 3,
                'features_failed': 0,
                'processing_time_ms': 0.0
            }
        }
        test_suite['test_cases'].append(test_case)
    
    # Save test suite to JSON
    json_path = test_dir / 'test_suite.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_suite, f, indent=2, ensure_ascii=False)
    
    print(f"\nTest images generated successfully!")
    print(f"Test suite saved to: {json_path}")
    print(f"Total test cases: {len(test_suite['test_cases'])}")
