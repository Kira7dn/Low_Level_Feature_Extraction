from PIL import Image
import io
import numpy as np
import os
import cv2
from typing import Union, Tuple, Optional
import time
import logging
from app.monitoring.performance import PerformanceMonitor
from app.config.settings import get_config, is_debug_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utility class with performance tracking"""
    
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(get_config('LOG_LEVEL')))

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
    def generate_cdn_url(base_url: str, image_path: str, transformations: Optional[dict] = None) -> str:
        """Generate a CDN-friendly URL with optional image transformations
        
        Args:
            base_url (str): Base CDN URL
            image_path (str): Path to the original image
            transformations (Optional[dict]): Image transformation parameters
        
        Returns:
            str: CDN URL with optional transformations
        """
        # Validate inputs
        if not base_url or not image_path:
            raise ValueError("Base URL and image path must be provided")
        
        # Extract filename
        filename = os.path.basename(image_path)
        
        # Default transformations
        default_transforms = {
            'format': 'webp',  # Convert to WebP
            'quality': 85,     # Compression quality
            'resize': {        # Optional resize
                'width': 1920,
                'height': 1080,
                'fit': 'max'   # Maintain aspect ratio
            }
        }
        
        # Merge default and provided transformations
        if transformations:
            default_transforms.update(transformations)
        
        # Construct CDN URL with transformations
        transform_params = []
        
        # Format transformation
        if default_transforms.get('format'):
            transform_params.append(f"f:{default_transforms['format']}")
        
        # Quality transformation
        if default_transforms.get('quality'):
            transform_params.append(f"q:{default_transforms['quality']}")
        
        # Resize transformation
        resize = default_transforms.get('resize', {})
        if resize:
            width = resize.get('width', 'auto')
            height = resize.get('height', 'auto')
            fit = resize.get('fit', 'max')
            transform_params.append(f"w:{width},h:{height},fit:{fit}")
        
        # Combine transformations
        transform_string = ','.join(transform_params)
        
        # Construct final CDN URL
        cdn_url = f"{base_url.rstrip('/')}/{transform_string}/{filename}"
        
        return cdn_url
    
    @staticmethod
    def auto_process_image(image, 
                            max_width: int = 1920, 
                            max_height: int = 1080, 
                            target_format: str = 'webp', 
                            quality: int = 85) -> bytes:
        """
        Automatically process an image with resizing, compression, and format conversion
        
        Args:
            image: Input image (PIL Image, numpy array, or bytes)
            max_width (int, optional): Maximum width. Defaults to 1920.
            max_height (int, optional): Maximum height. Defaults to 1080.
            target_format (str, optional): Target image format. Defaults to 'webp'.
            quality (int, optional): Compression quality. Defaults to 85.
        
        Returns:
            bytes: Optimized image
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
            
            # Resize image
            resized_image = ImageProcessor.resize_image(
                image, 
                max_width=max_width, 
                max_height=max_height
            )
            
            # Convert and compress image
            return ImageProcessor.convert_format(
                resized_image, 
                target_format=target_format
            )
        
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Auto image processing took {processing_time:.4f} seconds")

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
