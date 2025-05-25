import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional

from app.api.v1.models.analyze import FontFeatures  # Import the FontFeatures model

class FontDetector:
    # List of common fonts to compare against
    COMMON_FONTS = [
        "Arial", "Helvetica", "Roboto", "Open Sans", "Lato", 
        "Montserrat", "Times New Roman", "Georgia", "Courier New",
        "Verdana", "Tahoma", "Trebuchet MS", "Impact"
    ]

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for text detection
        
        Args:
            image (np.ndarray): Input image in BGR format
        
        Returns:
            np.ndarray: Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary

    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions containing text
        
        Args:
            image (np.ndarray): Preprocessed binary image
        
        Returns:
            List of text region bounding boxes (x, y, width, height)
        """
        # Find contours
        contours, _ = cv2.findContours(
            image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours to find potential text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio and size
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 15 and h > 8:
                text_regions.append((x, y, w, h))
        
        return text_regions

    @staticmethod
    def estimate_font_size(region_height: int) -> int:
        """
        Estimate font size based on region height
        
        Args:
            region_height (int): Height of text region
        
        Returns:
            int: Estimated font size
        """
        # Simple heuristic: font size is roughly 70-80% of region height
        return int(region_height * 0.75)

    @staticmethod
    def estimate_font_weight(region: np.ndarray) -> str:
        """
        Estimate font weight based on pixel intensity
        
        Args:
            region (np.ndarray): Text region image
        
        Returns:
            str: Estimated font weight ('Light', 'Regular', 'Bold')
        """
        # Convert to grayscale if not already
        if len(region.shape) > 2:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate average pixel intensity
        avg_intensity = np.mean(region)
        
        # Precise heuristic thresholds for font weight
        if avg_intensity >= 250:  # Near white
            return "Light"
        elif avg_intensity > 190:  # Light to medium gray
            return "Regular"
        else:  # Dark gray or black
            return "Bold"

    @classmethod
    def detect_font(cls, image: np.ndarray) -> Optional[FontFeatures]:
        """
        Main method to detect font properties
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            Optional[FontFeatures]: Font detection results or None if no text detected
        """
        try:
            # Preprocess image
            binary = cls.preprocess_image(image)
            
            # Detect text regions
            text_regions = cls.detect_text_regions(binary)
            
            # If no text regions found, return None
            if not text_regions:
                return None
            
            # Use the largest text region for analysis
            largest_region = max(text_regions, key=lambda r: r[2] * r[3])
            x, y, w, h = largest_region
            
            # Extract text region
            text_region = image[y:y+h, x:x+w]
            
            # Get font family and estimate size/weight
            font_family = cls.identify_font_family(text_region)
            font_size = cls.estimate_font_size(h)
            font_weight = cls.estimate_font_weight(text_region)
            
            # Return FontFeatures object
            return FontFeatures(
                font_family=font_family,
                font_size=float(font_size),
                font_style=font_weight,
                confidence=0.8  # Default confidence, can be refined
            )
            
        except Exception as e:
            # Log error and return None
            import logging
            logging.error(f"Error in font detection: {str(e)}")
            return None

    @staticmethod
    def identify_font_family(region: np.ndarray) -> str:
        """
        Attempt to identify font family
        
        Args:
            region (np.ndarray): Text region image
        
        Returns:
            str: Estimated font family name
        """
        # This is a placeholder. In a real-world scenario, 
        # this would use more advanced font matching techniques
        return "Arial"  # Default fallback
