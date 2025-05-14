import cv2
import numpy as np
from typing import Dict, List, Any
from scipy.spatial import distance

class ShapeAnalyzer:
    @staticmethod
    def preprocess_image(image):
        """
        Preprocess image for shape detection
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image ready for shape analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated

    @staticmethod
    def detect_border_radius(contour, epsilon_factor=0.02):
        """
        Detect border radius of a contour
        
        Args:
            contour: OpenCV contour
            epsilon_factor: Factor for contour approximation
        
        Returns:
            Estimated border radius as a float
        """
        # Approximate the contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If it's a rectangle with rounded corners
        if len(approx) > 4:
            # Calculate convex hull to get the difference
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            # Estimate border radius based on the difference between hull and contour
            if hull_area > 0:
                area_ratio = 1 - (contour_area / hull_area)
                border_radius = area_ratio * 50.0  # Scaling factor can be adjusted
                return max(0.0, border_radius)
        
        return 0.0

    @staticmethod
    def analyze_shapes(image) -> Dict[str, Any]:
        """
        Analyze shapes in the image
        
        Args:
            image: Input image
        
        Returns:
            Dictionary containing shape analysis results
        """
        # Preprocess the image
        preprocessed = ShapeAnalyzer.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze shapes
        shape_results = []
        for contour in contours:
            # Filter out very small contours
            if cv2.contourArea(contour) < 100:  # Minimum area threshold
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Detect border radius
            border_radius = ShapeAnalyzer.detect_border_radius(contour)
            
            # Estimate shape type
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            shape_type = "unknown"
            if len(approx) == 3:
                shape_type = "triangle"
            elif len(approx) == 4:
                # Check if it's a rectangle or square
                aspect_ratio = w / float(h)
                shape_type = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
            elif len(approx) > 4:
                # Check if it's close to a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    shape_type = "circle" if circularity > 0.8 else "polygon"
            
            shape_results.append({
                "type": shape_type,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "border_radius": border_radius,
                "area": cv2.contourArea(contour)
            })
        
        return {
            "shapes": shape_results,
            "total_shapes": len(shape_results),
            "metadata": {
                "image_width": image.shape[1],
                "image_height": image.shape[0]
            }
        }
