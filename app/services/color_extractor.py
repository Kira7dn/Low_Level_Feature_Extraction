import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional, Any
import warnings
from pydantic import BaseModel, Field
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans

# Suppress specific warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

# Import the ColorFeatures model
from .models import ColorFeatures

# Keep ColorPalette for backward compatibility
class ColorPalette(ColorFeatures):
    """Legacy color palette class. Use ColorFeatures instead.
    
    Deprecated: This class is maintained for backward compatibility.
    New code should use the ColorFeatures class from app.services.models.
    """
    class Config:
        json_schema_extra = {
            "example": {
                "primary": "#1a73e8",
                "background": "#f8f9fa",
                "accent": ["#0d47a1", "#64b5f6"],
                "metadata": {
                    "success": True,
                    "timestamp": 0.0,
                    "processing_time": 0.0
                }
            }
        }

class ColorExtractor:
    @staticmethod
    def rgb_to_hex(rgb: tuple) -> str:
        """Convert RGB tuple to HEX string."""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert HEX color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def get_contrast_ratio(color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors (WCAG 2.0)."""
        def get_luminance(c: str) -> float:
            # Convert hex to RGB
            r, g, b = (int(c[i:i+2], 16) / 255.0 for i in (1, 3, 5) if len(c) >= 6)
            # Convert to linear RGB
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1 = get_luminance(color1)
        l2 = get_luminance(color2)
        lighter, darker = (l1, l2) if l1 > l2 else (l2, l1)
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def is_light_color(rgb: tuple) -> bool:
        """Check if a color is light using relative luminance."""
        r, g, b = [x / 255.0 for x in rgb]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return luminance > 0.6
    
    @staticmethod
    def _process_image(image: Union[np.ndarray, Image.Image, None]) -> np.ndarray:
        """Process input image and convert to RGB numpy array.
        
        Args:
            image: Input image as numpy array, PIL Image, or None
            
        Returns:
            np.ndarray: Processed RGB image as numpy array with shape (height, width, 3)
            
        Raises:
            ValueError: If image format is not supported or image is invalid
        """
        DEFAULT_IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Handle None input
        if image is None:
            return DEFAULT_IMAGE
            
        # Handle PIL Image
        if isinstance(image, Image.Image):
            try:
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                if img_array.size == 0:
                    return DEFAULT_IMAGE
                    
                # Handle different color modes
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    img_array = np.array(background)
                elif image.mode != 'RGB':
                    img_array = np.array(image.convert('RGB'))
                    
                return img_array.astype(np.uint8)
                
            except Exception as e:
                print(f"Error processing PIL Image: {e}")
                return DEFAULT_IMAGE
                
        # Handle numpy array
        if isinstance(image, np.ndarray):
            try:
                # Handle empty array
                if image.size == 0:
                    return DEFAULT_IMAGE
                    
                # Make a copy to avoid modifying the original
                img = image.copy()
                
                # Ensure we have at least 2D array
                if len(img.shape) == 0:
                    return DEFAULT_IMAGE
                    
                # Handle 1D array (flattened image)
                if len(img.shape) == 1:
                    # Try to reshape to 2D if possible
                    side = int(np.sqrt(len(img) / 3))
                    if side * side * 3 == len(img):
                        img = img.reshape((side, side, 3))
                    else:
                        return DEFAULT_IMAGE
                
                # Handle 2D grayscale
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # Handle 3D images
                elif len(img.shape) == 3:
                    # Check channel position (H,W,C) or (C,H,W)
                    if img.shape[0] <= 4:  # Assume (C,H,W)
                        img = np.transpose(img, (1, 2, 0))  # Convert to (H,W,C)
                    
                    # Handle different channel counts
                    if img.shape[2] == 1:  # Single channel
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 3:  # RGB or BGR
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    elif img.shape[2] == 4:  # RGBA or BGRA
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:  # More than 4 channels, take first 3
                        img = img[..., :3]
                
                # Ensure correct data type and range
                if img.dtype != np.uint8:
                    if np.issubdtype(img.dtype, np.floating):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                return img
                
            except Exception as e:
                print(f"Error processing numpy array: {e}")
                return DEFAULT_IMAGE
        
        # If we get here, the input type is not supported
        return DEFAULT_IMAGE
    
    @staticmethod
    def _get_dominant_colors(pixels: np.ndarray, n_colors: int) -> tuple:
        """Extract dominant colors using K-means clustering."""
        # Get unique colors to avoid duplicate points
        unique_colors = np.unique(pixels, axis=0)
        
        # Adjust number of colors if needed
        actual_n_colors = min(n_colors, len(unique_colors))
        if actual_n_colors < n_colors:
            print(f"Warning: Only {len(unique_colors)} unique colors found, "
                  f"reducing number of clusters from {n_colors} to {actual_n_colors}")
        
        if actual_n_colors <= 1:
            return unique_colors, np.array([0] * len(unique_colors))
        
        # Convert to float32 for K-means
        pixels_float = np.float32(unique_colors)
        
        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
        try:
            _, labels, centers = cv2.kmeans(
                pixels_float, actual_n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            return centers.astype(np.uint8), labels.flatten()
        except cv2.error as e:
            print(f"Error in K-means: {str(e)}")
            # Fallback to using unique colors directly
            return unique_colors[:actual_n_colors], np.arange(actual_n_colors)
    
    @staticmethod
    def extract_colors(image: Union[np.ndarray, Image.Image], n_colors: int = 5) -> ColorFeatures:
        """
        Extract dominant colors from an image.
        
        Args:
            image: Input image as numpy array or PIL Image
            n_colors: Number of colors to extract (default: 5)
        
        Returns:
            ColorFeatures: Object containing primary, background, and accent colors
        """
        try:
            # Process the input image
            img_array = ColorExtractor._process_image(image)
            
            # Reshape the image to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Add a small amount of noise to prevent duplicate points
            if len(pixels) > 0:
                noise = np.random.normal(0, 0.5, pixels.shape).astype(np.int8)
                pixels = np.clip(pixels.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            
            # Get dominant colors
            centers, labels = ColorExtractor._get_dominant_colors(pixels, n_colors)
            
            # Count occurrences of each label
            if len(centers) > 1:
                counts = np.bincount(labels, minlength=len(centers))
                # Sort colors by frequency (most common first)
                sorted_indices = np.argsort(-counts)
                centers = centers[sorted_indices]
                counts = counts[sorted_indices]
            
            # Convert colors to hex
            hex_colors = [ColorExtractor.rgb_to_hex(tuple(color)) for color in centers]
            
            # Remove white and black from palette if present
            hex_colors = [c for c in hex_colors if c.lower() not in ['#ffffff', '#000000']]
            
            # If no colors left after filtering, add a contrasting color
            if not hex_colors:
                bg_color = '#000000' if ColorExtractor.is_light_color((255, 255, 255)) else '#FFFFFF'
                return ColorFeatures(
                    primary=bg_color,
                    background=bg_color,
                    accent=[bg_color] * 3,
                    metadata={
                        'success': True,
                        'timestamp': 0.0,  # Will be set by the router
                        'processing_time': 0.0  # Will be set by the router
                    }
                )
            
            # Find the primary color (most frequent non-background color)
            primary = hex_colors[0] if hex_colors else '#000000'
            
            # Get accent colors (all other colors except primary)
            accent_colors = [c for c in hex_colors if c != primary][:3]  # Max 3 accent colors
            
            # Ensure we have at least 3 accent colors by duplicating if necessary
            while len(accent_colors) < 3:
                if accent_colors:
                    accent_colors.append(accent_colors[-1])
                else:
                    accent_colors.append(primary)
            
            # Determine background color (light or dark based on primary color)
            bg_color = '#FFFFFF' if not ColorExtractor.is_light_color(
                ColorExtractor.hex_to_rgb(primary)) else '#000000'
            
            return ColorFeatures(
                primary=primary,
                background=bg_color,
                accent=accent_colors[:3],  # Return max 3 accent colors
                metadata={
                    'success': True,
                    'timestamp': 0.0,  # Will be set by the router
                    'processing_time': 0.0  # Will be set by the router
                }
            )
            
        except Exception as e:
            import traceback
            print(f"Error in extract_colors: {str(e)}\n{traceback.format_exc()}")
            # Return default colors in case of error
            return ColorFeatures(
                primary='#000000',
                background='#FFFFFF',
                accent=['#666666', '#999999', '#CCCCCC'],
                metadata={
                    'success': False,
                    'error': str(e),
                    'timestamp': 0.0,
                    'processing_time': 0.0
                }
            )
