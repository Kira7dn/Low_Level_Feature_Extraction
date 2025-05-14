import pytest
from app.frontend_utils import ImageCdnManager

def test_image_cdn_manager_default():
    """Test default CDN URL generation"""
    cdn_manager = ImageCdnManager('https://cdn.example.com/images')
    
    image_url = cdn_manager.get_optimized_image_url('/path/to/sample.jpg')
    
    # Verify base URL structure
    assert image_url.startswith('https://cdn.example.com/images')
    assert 'sample.jpg' in image_url
    
    # Verify default transformations
    assert 'f:webp' in image_url
    assert 'q:85' in image_url
    assert 'w:1920,h:1080,fit:max' in image_url

def test_image_cdn_manager_custom_transformations():
    """Test CDN URL generation with custom transformations"""
    cdn_manager = ImageCdnManager('https://cdn.example.com/images')
    
    image_url = cdn_manager.get_optimized_image_url(
        '/path/to/custom.png', 
        {
            'format': 'avif',
            'quality': 75,
            'resize': {
                'width': 800,
                'height': 600,
                'fit': 'crop'
            }
        }
    )
    
    # Verify base URL structure
    assert image_url.startswith('https://cdn.example.com/images')
    assert 'custom.png' in image_url
    
    # Verify custom transformations
    assert 'f:avif' in image_url
    assert 'q:75' in image_url
    assert 'w:800,h:600,fit:crop' in image_url

def test_image_cdn_manager_responsive_urls():
    """Test responsive image URL generation"""
    cdn_manager = ImageCdnManager('https://cdn.example.com/images')
    
    responsive_urls = cdn_manager.get_responsive_image_urls('/path/to/responsive.jpg')
    
    # Verify responsive URLs
    assert len(responsive_urls) == 3
    assert 'mobile' in responsive_urls
    assert 'tablet' in responsive_urls
    assert 'desktop' in responsive_urls
    
    # Verify each URL contains expected transformations
    for size, url in responsive_urls.items():
        assert url.startswith('https://cdn.example.com/images')
        assert 'responsive.jpg' in url
        assert 'f:webp' in url
        assert 'q:85' in url
