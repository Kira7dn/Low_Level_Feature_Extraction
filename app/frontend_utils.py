from typing import Dict, Optional
from .services.image_processor import ImageProcessor

class ImageCdnManager:
    """
    Manages CDN URL generation for images with intelligent defaults and transformations
    """
    def __init__(self, base_cdn_url: str):
        """
        Initialize CDN URL manager with base CDN URL
        
        Args:
            base_cdn_url (str): Base URL for the CDN service
        """
        self.base_cdn_url = base_cdn_url

    def get_optimized_image_url(
        self, 
        image_path: str, 
        transformations: Optional[Dict] = None
    ) -> str:
        """
        Generate an optimized CDN URL for an image
        
        Args:
            image_path (str): Path to the original image
            transformations (dict, optional): Custom image transformations
        
        Returns:
            str: Optimized CDN URL for the image
        """
        # Default transformations for web optimization
        default_transforms = {
            'format': 'webp',
            'quality': 85,
            'resize': {
                'width': 1920,
                'height': 1080,
                'fit': 'max'
            }
        }

        # Merge default and custom transformations
        if transformations:
            default_transforms.update(transformations)

        # Generate CDN URL
        return ImageProcessor.generate_cdn_url(
            self.base_cdn_url, 
            image_path, 
            default_transforms
        )

    def get_responsive_image_urls(
        self, 
        image_path: str, 
        sizes: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, str]:
        """
        Generate multiple image URLs for responsive design
        
        Args:
            image_path (str): Path to the original image
            sizes (dict, optional): Dictionary of size configurations
        
        Returns:
            dict: Dictionary of image URLs for different screen sizes
        """
        # Default responsive sizes
        default_sizes = {
            'mobile': {'width': 480, 'height': 640},
            'tablet': {'width': 768, 'height': 1024},
            'desktop': {'width': 1920, 'height': 1080}
        }

        # Merge default and custom sizes
        if sizes:
            default_sizes.update(sizes)

        # Generate URLs for each size
        return {
            size: self.get_optimized_image_url(
                image_path, 
                {'resize': config}
            )
            for size, config in default_sizes.items()
        }
