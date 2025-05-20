from PIL import Image
import io
import numpy as np
import os
import cv2
from typing import Union, Tuple, Optional
import time
import logging
from app.monitoring.performance import PerformanceMonitor

# Configure logging
log_dir = '/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'image_processing_performance.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utility class with performance tracking"""
    
    # Configure logging
    logger = logging.getLogger(__name__)

    @staticmethod
    @PerformanceMonitor.track_performance()
    def load_image(image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes into PIL Image
        
        Args:
            image_bytes (bytes): Image content as bytes
        
        Returns:
            PIL.Image.Image: Loaded image
        """
        start_time = time.time()
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            # Log detailed error information
            ImageProcessor.logger.error(f"Image loading error: {str(e)}")
            ImageProcessor.logger.error(f"Image bytes length: {len(image_bytes) if image_bytes else 0}")
            raise
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            ImageProcessor.logger.info(f"Image loading took {processing_time:.4f} seconds")

    @staticmethod
    @PerformanceMonitor.track_performance()
    def load_cv2_image(image_bytes: bytes) -> np.ndarray:
        """Load image from bytes into OpenCV format
        
        Args:
            image_bytes (bytes): Image bytes to load
        
        Returns:
            np.ndarray: Image in OpenCV format
        """
        start_time = time.time()
        try:
            # Validate input
            if image_bytes is None or len(image_bytes) == 0:
                raise ValueError("No image bytes provided")
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            processing_time = time.time() - start_time
            ImageProcessor.logger.info(f"Image loading took {processing_time:.4f} seconds")
            ImageProcessor.logger.info(f"Image bytes length: {len(image_bytes)}")
            
            return image
        except Exception as e:
            ImageProcessor.logger.error(f"Image loading error: {e}")
            ImageProcessor.logger.error(f"Image bytes length: {len(image_bytes) if isinstance(image_bytes, bytes) else 0}")
            raise ValueError("Decoded image is empty")
            ImageProcessor.logger.error(f"Image bytes length: {len(image_bytes) if image_bytes else 0}")
            raise
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            ImageProcessor.logger.info(f"Image loading took {processing_time:.4f} seconds")

    @staticmethod
    @PerformanceMonitor.track_performance()
    def resize_image(image: Union[Image.Image, np.ndarray], max_width: int = 1920, max_height: int = 1080, fit: str = 'max') -> Union[Image.Image, np.ndarray]:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image (Union[PIL.Image.Image, numpy.ndarray]): Input image
            max_width (int, optional): Maximum width. Defaults to 1920.
            max_height (int, optional): Maximum height. Defaults to 1080.
            fit (str, optional): Fit mode. Defaults to 'max'.
        
        Returns:
            Union[PIL.Image.Image, numpy.ndarray]: Resized image
        """
        start_time = time.time()
        try:
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
                
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            raise TypeError("Unsupported image type")
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            ImageProcessor.logger.info(f"Image resize to {max_width}x{max_height} took {processing_time:.4f} seconds")

    @staticmethod
    @PerformanceMonitor.track_performance()
    def compress_image(image: Union[Image.Image, np.ndarray], quality: int = 85) -> bytes:
        """Compress image to reduce file size
        
        Args:
            image (Union[PIL.Image, numpy.ndarray]): Image to compress
            quality (int): Compression quality (0-100)
        
        Returns:
            Compressed image bytes
        """
        # Convert numpy array to PIL Image if needed
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Compress to WebP
        output = io.BytesIO()
        image.save(output, format='WebP', quality=quality)
        return output.getvalue()
    
    @staticmethod
    def convert_to_webp(image_bytes: bytes, quality: int = 85) -> bytes:
        """Convert image to WebP format
        
        Args:
            image_bytes (bytes): Original image bytes
            quality (int): Compression quality (0-100)
        
        Returns:
            WebP image bytes
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to WebP
        output = io.BytesIO()
        image.save(output, format='WebP', quality=quality)
        return output.getvalue()
    
    @staticmethod
    def lazy_load_image(image_path: str, max_width: int = 800, max_height: int = 600) -> Optional[bytes]:
        """Lazily load and optimize an image
        
        Args:
            image_path (str): Path to the image file
            max_width (int): Maximum width for lazy loading
            max_height (int): Maximum height for lazy loading
        
        Returns:
            Optimized image bytes or None if file not found
        """
        if not os.path.exists(image_path):
            return None
        
        # Load image
        image = Image.open(image_path)
        
        # Resize
        resized_image = ImageProcessor.resize_image(image, max_width, max_height)
        
        # Compress to WebP
        return ImageProcessor.compress_image(resized_image)
    
    @staticmethod
    @PerformanceMonitor.track_performance()
    def auto_process_image(
        image_bytes: bytes,
        max_width: int = 1920,
        max_height: int = 1080,
        quality: int = 85
    ) -> np.ndarray:
        """
        Process image from bytes to a standardized numpy array in BGR format.
        
        Args:
            image_bytes: Raw image data as bytes
            max_width: Maximum width for resizing
            max_height: Maximum height for resizing
            quality: Quality for compression (if needed)
            
        Returns:
            np.ndarray: Processed image in BGR format
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            # 1. Load image from bytes
            image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            # 2. Convert to RGB (PIL works with RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 3. Resize maintaining aspect ratio
            pil_image.thumbnail(
                (max_width, max_height),
                Image.Resampling.LANCZOS
            )
            
            # 4. Convert back to numpy array in BGR format
            resized_image = np.array(pil_image)
            return cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Image processing error: {str(e)}")

    @staticmethod
    def convert_format(image, target_format: str = 'png') -> bytes:
        """
        Convert image to specified format
        
        Args:
            image: Input image (PIL Image, numpy array, or bytes)
            target_format (str, optional): Target image format. Defaults to 'png'.
        
        Returns:
            bytes: Image in target format
        """
        start_time = time.time()
        try:
            # If image is bytes, first convert to appropriate image type
            if isinstance(image, bytes):
                try:
                    # Try loading as PIL Image first
                    image = ImageProcessor.load_image(image)
                except Exception:
                    # If not PIL, try loading as CV2 image
                    image = ImageProcessor.load_cv2_image(image)
            
            # Handle PIL Image
            if isinstance(image, Image.Image):
                byte_arr = io.BytesIO()
                image.save(byte_arr, format=target_format.upper())
                return byte_arr.getvalue()
            
            # Handle OpenCV image
            if isinstance(image, np.ndarray):
                _, buffer = cv2.imencode(f'.{target_format}', image)
                return buffer.tobytes()
            
            # Provide more informative error if type is truly unsupported
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL Image, numpy array, or bytes.")
        
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Image format conversion to {target_format} took {processing_time:.4f} seconds")
