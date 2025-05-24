import cv2
import numpy as np

class ShadowAnalyzer:
    @staticmethod
    def preprocess_image(image):
        """Preprocess image for shadow detection: grayscale and blur"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    @staticmethod
    def analyze_shadow_level(image):
        """Analyze shadow intensity and map to shadow_level (High/Moderate/Low)"""
        processed = ShadowAnalyzer.preprocess_image(image)
        # Use adaptive thresholding to identify dark regions (potential shadows)
        thresh = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        # Only consider pixels in detected shadow regions
        shadow_pixels = processed[thresh == 255]
        if shadow_pixels.size == 0:
            return "Low"
        avg_darkness = 255 - np.mean(shadow_pixels)  # Higher means darker
        # Thresholds can be tuned based on dataset
        if avg_darkness < 30:
            return "Low"
        elif avg_darkness < 60:
            return "Moderate"
        else:
            return "High"
