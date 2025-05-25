from typing import Union, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

from app.services.image_processor import ImageProcessor

class ImageTransformer:
    @staticmethod
    def resize(
        image: Union[Image.Image, np.ndarray], 
        width: Optional[int] = None, 
        height: Optional[int] = None, 
        maintain_aspect_ratio: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Resize image with optional aspect ratio preservation
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            width (Optional[int]): Target width
            height (Optional[int]): Target height
            maintain_aspect_ratio (bool): Preserve image aspect ratio
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Resized image
        """
        # PIL Image handling
        if isinstance(image, Image.Image):
            if maintain_aspect_ratio:
                image.thumbnail((width or height, height or width))
                return image
            
            return image.resize((width, height))
        
        # OpenCV image handling
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
            
            if maintain_aspect_ratio:
                # Calculate scaling factor
                scale = min(
                    (width or orig_width) / orig_width, 
                    (height or orig_height) / orig_height
                )
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
            else:
                new_width = width or orig_width
                new_height = height or orig_height
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
        
        raise TypeError("Unsupported image type")
    
    @staticmethod
    def apply_filter(
        image: Union[Image.Image, np.ndarray], 
        filter_type: str = 'blur'
    ) -> Union[Image.Image, np.ndarray]:
        """
        Apply various image filters
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            filter_type (str): Type of filter to apply
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Filtered image
        """
        # PIL Image handling
        if isinstance(image, Image.Image):
            filters = {
                'blur': ImageFilter.BLUR,
                'contour': ImageFilter.CONTOUR,
                'detail': ImageFilter.DETAIL,
                'edge_enhance': ImageFilter.EDGE_ENHANCE,
                'emboss': ImageFilter.EMBOSS,
                'sharpen': ImageFilter.SHARPEN,
            }
            
            if filter_type not in filters:
                raise ValueError(f"Unsupported filter type: {filter_type}")
            
            return image.filter(filters[filter_type])
        
        # OpenCV image handling
        if isinstance(image, np.ndarray):
            filters = {
                'blur': (3, 3),
                'gaussian_blur': (5, 5),
                'median_blur': 3,
            }
            
            if filter_type == 'blur':
                return cv2.blur(image, filters['blur'])
            elif filter_type == 'gaussian_blur':
                return cv2.GaussianBlur(image, filters['gaussian_blur'], 0)
            elif filter_type == 'median_blur':
                return cv2.medianBlur(image, filters['median_blur'])
            
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        raise TypeError("Unsupported image type")
    
    @staticmethod
    def adjust_brightness_contrast(
        image: Union[Image.Image, np.ndarray], 
        brightness: float = 1.0, 
        contrast: float = 1.0
    ) -> Union[Image.Image, np.ndarray]:
        """
        Adjust image brightness and contrast
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            brightness (float): Brightness factor (1.0 = original)
            contrast (float): Contrast factor (1.0 = original)
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Adjusted image
        """
        # PIL Image handling
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(contrast)
        
        # OpenCV image handling
        if isinstance(image, np.ndarray):
            # Adjust brightness
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            
            # Adjust contrast
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
            
            return image
        
        raise TypeError("Unsupported image type")
    
    @staticmethod
    def generate_thumbnail(
        image: Union[Image.Image, np.ndarray], 
        size: Tuple[int, int] = (128, 128)
    ) -> Union[Image.Image, np.ndarray]:
        """
        Generate a thumbnail from the input image
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            size (Tuple[int, int]): Thumbnail size
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Thumbnail image
        """
        # PIL Image handling
        if isinstance(image, Image.Image):
            image.thumbnail(size)
            return image
        
        # OpenCV image handling
        if isinstance(image, np.ndarray):
            return cv2.resize(
                image, 
                size, 
                interpolation=cv2.INTER_AREA
            )
        
        raise TypeError("Unsupported image type")
