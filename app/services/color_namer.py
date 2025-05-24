import numpy as np
from typing import Dict, List, Tuple, Union

class ColorNamer:
    """
    Service for identifying color names based on RGB values
    """
    # Predefined color dictionary with common color names and their RGB values
    COLOR_DICT: Dict[str, Tuple[int, int, int]] = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'navy': (0, 0, 128),
        'olive': (128, 128, 0),
        'maroon': (128, 0, 0),
        'teal': (0, 128, 128),
        'silver': (192, 192, 192),
        'lime': (0, 255, 0),
        'aqua': (0, 255, 255)
    }

    @staticmethod
    def rgb_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """
        Calculate Euclidean distance between two RGB colors
        
        Args:
            rgb1: First RGB color tuple
            rgb2: Second RGB color tuple
        
        Returns:
            Euclidean distance between the two colors
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

    @classmethod
    def get_color_name(cls, rgb: Tuple[int, int, int]) -> str:
        """
        Find the closest named color for a given RGB value
        
        Args:
            rgb: RGB color tuple to name
        
        Returns:
            Name of the closest color
        """
        # Ensure input is a tuple of integers
        rgb = tuple(map(int, rgb))
        
        # Find the color with the minimum distance
        closest_color = min(
            cls.COLOR_DICT.items(), 
            key=lambda x: cls.rgb_distance(rgb, x[1])
        )
        
        return closest_color[0]

    @classmethod
    def get_color_details(cls, rgb: Tuple[int, int, int]) -> Dict[str, Union[str, List[int]]]:
        """
        Get comprehensive color details including name
        
        Args:
            rgb: RGB color tuple
        
        Returns:
            Dictionary with color details
        """
        return {
            'rgb': list(rgb),
            'hex': '#{:02x}{:02x}{:02x}'.format(*rgb),
            'name': cls.get_color_name(rgb)
        }
