import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Union, Optional, Tuple
from app.services.color_namer import ColorNamer

class ColorExtractor:
    @staticmethod
    def rgb_to_hex(rgb: tuple) -> str:
        """Convert RGB tuple to HEX string"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    @staticmethod
    def extract_colors(image: Union[np.ndarray, Image.Image], n_colors: int = 5) -> Dict[str, Union[str, List[str]]]:
        """
        Extract dominant colors using K-means clustering with improved color detection.
        
        Args:
            image: Input image as numpy array or PIL Image
            n_colors: Number of colors to extract (default: 5)
        
        Returns:
            Dictionary with the following structure:
            {
                'primary': str,         # Primary color in hex (e.g., '#007BFF')
                'background': str,      # Background color in hex
                'accent': List[str]      # List of accent colors in hex
            }
        """
        def get_contrast_ratio(color1: str, color2: str) -> float:
            """Calculate contrast ratio between two colors (WCAG 2.0)."""
            def get_luminance(c: str) -> float:
                # Convert hex to RGB
                r, g, b = (int(c[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                # Convert to linear RGB
                r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
                g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
                b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
                return 0.2126 * r + 0.7152 * g + 0.0722 * b
            
            l1 = get_luminance(color1)
            l2 = get_luminance(color2)
            lighter, darker = (l1, l2) if l1 > l2 else (l2, l1)
            return (lighter + 0.05) / (darker + 0.05)
        
        def is_light_color(rgb: tuple) -> bool:
            """Check if a color is light using relative luminance."""
            r, g, b = [x / 255.0 for x in rgb]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return luminance > 0.6
        
        def get_dominant_colors(pixels: np.ndarray, n_colors: int) -> tuple:
            """Extract dominant colors using K-means clustering."""
            if len(pixels) < n_colors * 5:  # Not enough pixels for clustering
                n_colors = max(1, len(pixels) // 5)
            
            if n_colors <= 0:
                return np.array([[0, 0, 0]]), np.array([len(pixels)])
                
            # Use mini-batch k-means for better performance with large images
            if len(pixels) > 1000:
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(
                    n_clusters=n_colors,
                    random_state=42,
                    batch_size=1000,
                    n_init=3
                )
                kmeans.fit(pixels)
                centers = kmeans.cluster_centers_.astype(np.uint8)
                counts = np.bincount(kmeans.labels_, minlength=len(centers))
            else:
                # For small images, use regular k-means with k-means++ initialization
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(
                    pixels.astype(np.float32),
                    n_colors,
                    None,
                    criteria,
                    10,
                    cv2.KMEANS_PP_CENTERS
                )
                counts = np.bincount(labels.flatten(), minlength=len(centers))
            
            # Sort colors by frequency (most common first)
            sorted_indices = np.argsort(-counts)
            return centers[sorted_indices], counts[sorted_indices]
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                # Convert to RGB mode if needed
                if image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    # Paste using alpha channel
                    background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' and len(image.split()) > 3 else None)
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                # Convert to numpy array
                image = np.array(image)
            
            # Ensure we have a valid image
            if image is None or image.size == 0:
                raise ValueError("Invalid or empty image")
                
            # Convert image to RGB format if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    # Convert RGBA to RGB with white background
                    alpha = image[:, :, 3] / 255.0
                    image = image[:, :, :3]
                    image = (255 * (1 - alpha)[:, :, None] + image * alpha[:, :, None]).astype(np.uint8)
                elif image.shape[2] == 1:  # Single channel
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] != 3:  # Not RGB
                    raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert from BGR to LAB color space - better for color perception
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Sample border pixels to determine background color (more efficient sampling)
            border_ratio = 0.15  # Sample 15% of the border for better accuracy
            border_width = max(1, int(min(width, height) * border_ratio))
            
            # Sample border pixels with adaptive stride
            stride = max(1, border_width // 5)  # More samples for better accuracy
            top_border = image_lab[:border_width:stride, :, :].reshape(-1, 3)
            bottom_border = image_lab[-border_width::stride, :, :].reshape(-1, 3)
            left_border = image_lab[:, :border_width:stride, :].reshape(-1, 3)
            right_border = image_lab[:, -border_width::stride, :].reshape(-1, 3)
            
            # Combine all border pixels
            border_pixels = np.vstack([top_border, bottom_border, left_border, right_border])
            
            # Find the most common color in LAB space with tolerance for similar colors
            from sklearn.cluster import DBSCAN
            
            # Use DBSCAN to group similar colors in LAB space
            clustering = DBSCAN(eps=5, min_samples=5, n_jobs=-1).fit(border_pixels)
            
            # Find the largest cluster (background)
            if len(np.unique(clustering.labels_)) > 1:
                bg_cluster = np.argmax(np.bincount(1 + clustering.labels_)) - 1
                bg_mask = clustering.labels_ == bg_cluster
                background_color_lab = np.median(border_pixels[bg_mask], axis=0).astype(np.uint8)
            else:
                # Fallback to most common color if clustering fails
                unique_colors, counts = np.unique(border_pixels, axis=0, return_counts=True)
                background_color_lab = unique_colors[np.argmax(counts)]
            
            # Convert background color back to RGB for consistency
            background_color_bgr = cv2.cvtColor(
                np.array([[background_color_lab]], dtype=np.uint8), 
                cv2.COLOR_LAB2BGR
            )[0][0]
            background_color_rgb = tuple(background_color_bgr[::-1])  # Convert to RGB
            bg_hex = ColorExtractor.rgb_to_hex(background_color_rgb)
            
            # Reshape the image to be a list of pixels for clustering
            pixels = image_lab.reshape(-1, 3).astype(np.float32)
            
            # Filter out background colors to focus on foreground content
            bg_distance = np.linalg.norm(pixels - background_color_lab, axis=1)
            fg_mask = bg_distance > 10  # Threshold to separate foreground
            
            if np.any(fg_mask):
                fg_pixels = pixels[fg_mask]
            else:
                fg_pixels = pixels  # Fallback to all pixels if no foreground detected
            
            # Get dominant colors from foreground
            if len(fg_pixels) > 0:
                centers, _ = get_dominant_colors(fg_pixels, n_colors)
                
                # Convert LAB colors back to RGB hex
                hex_colors = []
                for color in centers:
                    color_bgr = cv2.cvtColor(
                        np.array([[color.astype(np.uint8)]], dtype=np.uint8), 
                        cv2.COLOR_LAB2BGR
                    )[0][0]
                    hex_colors.append(ColorExtractor.rgb_to_hex(tuple(color_bgr[::-1])))
            else:
                hex_colors = ['#000000']  # Fallback color
            
            # Remove background color from palette if present
            hex_colors = [c for c in hex_colors if c.lower() != bg_hex.lower()]
            
            # If no colors left after filtering, add a contrasting color
            if not hex_colors:
                hex_colors = ['#000000' if is_light_color(background_color_rgb) else '#FFFFFF']
            
            # Find the most contrasting color to the background as primary
            max_contrast = 0
            primary = hex_colors[0]
            
            for color in hex_colors:
                if color.lower() == bg_hex.lower():
                    continue
                contrast = get_contrast_ratio(color, bg_hex)
                if contrast > max_contrast:
                    max_contrast = contrast
                    primary = color
            
            # Ensure good contrast for primary color
            if max_contrast < 3:  # WCAG AA minimum is 4.5:1 for normal text
                primary = '#000000' if is_light_color(background_color_rgb) else '#FFFFFF'
            
            # Get accent colors (excluding primary and background)
            accent = [c for c in hex_colors if c.lower() not in (primary.lower(), bg_hex.lower())]
            
            # Limit number of accent colors
            max_accent = min(4, len(accent))  # Maximum 4 accent colors
            accent = accent[:max_accent]
            
            # If no accent colors, generate some based on primary color
            if not accent and primary.lower() != bg_hex.lower():
                # Generate a complementary color
                r, g, b = [int(primary[i:i+2], 16) for i in (1, 3, 5)]
                comp_color = ColorExtractor.rgb_to_hex((255 - r, 255 - g, 255 - b))
                if comp_color.lower() not in (primary.lower(), bg_hex.lower()):
                    accent.append(comp_color)
            
            # Ensure we have at least one accent color
            if not accent and primary.lower() != bg_hex.lower():
                # Generate a lighter/darker variant of primary
                r, g, b = [int(primary[i:i+2], 16) for i in (1, 3, 5)]
                if is_light_color((r, g, b)):
                    # Darken if primary is light
                    variant = tuple(max(0, c - 40) for c in (r, g, b))
                else:
                    # Lighten if primary is dark
                    variant = tuple(min(255, c + 40) for c in (r, g, b))
                accent.append(ColorExtractor.rgb_to_hex(variant))
            
            # Return the final color palette
            return {
                'primary': primary,
                'background': bg_hex,
                'accent': accent
            }
        
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"Error in extract_colors: {str(e)}\n{traceback.format_exc()}")
            return {
                'primary': '#000000',
                'background': '#FFFFFF',
                'accent': []
            }
    
    @staticmethod
    def analyze_palette(image: Union[np.ndarray, Image.Image], n_colors: int = 5) -> Dict[str, Union[List[Dict[str, Union[str, List[int]]]], Dict[str, Union[str, List[int]]]]]:
        """
        Analyze color palette with detailed information
        
        Args:
            image: Input image as numpy array or PIL Image
        
        Returns:
            Dictionary with color palette details
        """
        # Extract colors
        hex_colors = ColorExtractor.extract_colors(image, n_colors)
        
        # Convert hex back to RGB for detailed response
        palette = []
        for hex_color in [hex_colors['primary'], hex_colors['background']] + hex_colors['accent']:
            # Convert hex to RGB
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Get color details including name
            color_details = ColorNamer.get_color_details(rgb)
            palette.append(color_details)
        
        return {
            'colors': palette,
            'primary': palette[0]['hex'] if palette else '#000000',
            'background': palette[1]['hex'] if len(palette) > 1 else '#000000',
            'accent': [color['hex'] for color in palette[2:]] if len(palette) > 2 else []
        }
