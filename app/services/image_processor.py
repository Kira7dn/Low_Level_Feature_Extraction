from PIL import Image
import io
import numpy as np
import cv2
from typing import Union, Tuple

class ImageProcessor:
    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes into PIL Image
        
        Args:
            image_bytes (bytes): Image content as bytes
        
        Returns:
            PIL.Image.Image: Loaded image
        """
        return Image.open(io.BytesIO(image_bytes))
    
    @staticmethod
    def load_cv2_image(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes into OpenCV format
        
        Args:
            image_bytes (bytes): Image content as bytes
        
        Returns:
            numpy.ndarray: Image in OpenCV format
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def resize_image(image: Union[Image.Image, np.ndarray], 
                     max_width: int = 1024, 
                     max_height: int = 1024) -> Union[Image.Image, np.ndarray]:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            max_width (int, optional): Maximum width. Defaults to 1024.
            max_height (int, optional): Maximum height. Defaults to 1024.
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Resized image
        """
        # Handle PIL Image
        if isinstance(image, Image.Image):
            image.thumbnail((max_width, max_height))
            return image
        
        # Handle OpenCV image
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            scaling_factor = min(max_width/width, max_height/height, 1)
            
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            
            return cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
        
        raise TypeError("Unsupported image type")
    
    @staticmethod
    def convert_format(image: Union[Image.Image, np.ndarray], 
                       target_format: str = 'png') -> bytes:
        """
        Convert image to specified format
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            target_format (str, optional): Target image format. Defaults to 'png'.
        
        Returns:
            bytes: Image in target format
        """
        # Handle PIL Image
        if isinstance(image, Image.Image):
            byte_arr = io.BytesIO()
            image.save(byte_arr, format=target_format.upper())
            return byte_arr.getvalue()
        
        # Handle OpenCV image
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode(f'.{target_format}', image)
            return buffer.tobytes()
        
        raise TypeError("Unsupported image type")
