import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Union, Optional, Tuple
from app.services.color_namer import ColorNamer

class ColorExtractor:
    @staticmethod
    def rgb_to_hex(rgb: tuple) -> str:
        """Convert RGB tuple to HEX string"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    @staticmethod
    def extract_colors(image: Union[np.ndarray, Image.Image], n_colors: int = 5) -> Dict[str, Union[str, List[str]]]:
        """
        Extract dominant colors using K-means clustering
        
        Args:
            image: Input image as numpy array or PIL Image
            n_colors: Number of colors to extract (default: 5)
        
        Returns:
            Dictionary with primary, background, and accent colors
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image to speed up processing
        img = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
        
        # Reshape the image to be a list of pixels
        pixels = img.reshape(-1, 3)
        
        # Convert to float for better precision
        pixels = np.float32(pixels)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(
            pixels, 
            n_colors, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        # Count occurrences of each label
        counts = np.bincount(labels.flatten())
        
        # Sort colors by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_centers = centers[sorted_indices]
        
        # Convert to hex
        hex_colors = [ColorExtractor.rgb_to_hex(color) for color in sorted_centers]
        
        # Return dictionary with primary, background, and accent colors
        return {
            "primary": hex_colors[0] if hex_colors else "#000000",
            "background": hex_colors[-1] if hex_colors else "#000000",
            "accent": hex_colors[1:-1] if len(hex_colors) > 2 else []
        }
    
    @staticmethod
    def analyze_palette(image: Union[np.ndarray, Image.Image], n_colors: int = 5) -> Dict[str, Union[Dict[str, Union[str, List[int]]], List[Dict[str, Union[str, List[int]]]]]]:
        """
        Analyze color palette with detailed information
        
        Args:
            image: Input image as numpy array or PIL Image
        
        Returns:
            Dictionary with color palette details
        """
        # Extract colors
        hex_colors = ColorExtractor.extract_colors(image, n_colors)
        
        # Convert hex back to RGB for detailed response
        palette = []
        for hex_color in hex_colors:
            # Convert hex to RGB
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Get color details including name
            color_details = ColorNamer.get_color_details(rgb)
            palette.append(color_details)
        
        return {
            'colors': palette,
            'primary_color': palette[0] if palette else None,
            'background_color': palette[-1] if palette else None
        }
