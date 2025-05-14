from PIL import Image, ImageFilter
import io

class ImageProcessor:
    @staticmethod
    def load_image(file_bytes):
        """Load image from bytes."""
        return Image.open(io.BytesIO(file_bytes))

class ImageTransformer:
    @staticmethod
    def resize(image, width=None, height=None, maintain_aspect_ratio=True):
        """Resize image while maintaining aspect ratio if specified."""
        if width and height:
            return image.resize((width, height))
        elif width:
            aspect_ratio = image.height / image.width
            new_height = int(width * aspect_ratio)
            return image.resize((width, new_height))
        elif height:
            aspect_ratio = image.width / image.height
            new_width = int(height * aspect_ratio)
            return image.resize((new_width, height))
        return image

    @staticmethod
    def apply_filter(image, filter_name='BLUR'):
        """Apply a filter to the image."""
        filter_map = {
            'BLUR': ImageFilter.BLUR,
            'CONTOUR': ImageFilter.CONTOUR,
            'DETAIL': ImageFilter.DETAIL,
            'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
            'SHARPEN': ImageFilter.SHARPEN,
        }
        filter_obj = filter_map.get(filter_name.upper(), ImageFilter.BLUR)
        return image.filter(filter_obj)

    @staticmethod
    def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
        """Adjust image brightness and contrast."""
        from PIL import ImageEnhance
        
        # Adjust brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(brightness)
        
        # Adjust contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        return contrast_enhancer.enhance(contrast)

    @staticmethod
    def generate_thumbnail(image, size=(128, 128)):
        """Generate a thumbnail of the image."""
        image.thumbnail(size)
        return image
