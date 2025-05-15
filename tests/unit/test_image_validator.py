import pytest
import time
from fastapi import UploadFile, HTTPException
from app.utils.error_handler import ValidationException
from io import BytesIO
from app.utils.image_validator import validate_image, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from PIL import Image, ImageDraw
from typing import Tuple, Dict, Any, List
from tests.constants import validate_response_structure, validate_processing_time

class TestImageValidator:
    @pytest.fixture
    def image_size(self) -> Tuple[int, int]:
        """Standard test image size"""
        return (100, 100)
    
    @pytest.fixture
    def test_image(self, image_size: Tuple[int, int]) -> Image.Image:
        """Create a test image with a simple pattern"""
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, image_size[0]-1, image_size[1]-1], outline='black')
        return img
    
    @pytest.fixture
    def image_bytes(self, test_image: Image.Image) -> BytesIO:
        """Convert test image to bytes"""
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize('format,filename', [
        ('PNG', 'test.png'),
        ('JPEG', 'test.jpg'),
        ('JPEG', 'test.jpeg'),
        ('WebP', 'test.webp')
    ])
    async def test_valid_image(self, test_image: Image.Image, format: str, filename: str):
        """Test validation of valid images in different formats"""
        # Convert image to specified format
        img_bytes = BytesIO()
        test_image.save(img_bytes, format=format)
        img_bytes.seek(0)
        
        # Create upload file
        file = UploadFile(filename=filename, file=img_bytes)
        
        # Validate the image
        start_time = time.time()
        contents = await validate_image(file)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"contents": contents}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["contents"],
            value_types={"contents": bytes},
            context="image_validation"
        )
        assert is_valid, error_msg
        assert len(contents) > 0, "Image contents should not be empty"
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_validation"
        )
        assert is_valid, error_msg
        assert contents == img_bytes.getvalue(), "Returned contents should match original"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize('filename,expected_msg', [
        ('test.gif', 'File format not supported. Allowed formats'),
        ('test.bmp', 'File format not supported. Allowed formats'),
        ('test.tiff', 'File format not supported. Allowed formats'),
        ('test.txt', 'File format not supported. Allowed formats'),
        ('test', 'File format not supported. Allowed formats')
    ])
    async def test_invalid_extension(self, filename: str, expected_msg: str):
        """Test validation of files with invalid extensions"""
        test_bytes = BytesIO(b'fake content')
        file = UploadFile(filename=filename, file=test_bytes)
        
        with pytest.raises(ValidationException) as exc_info:
            await validate_image(file)
            
        assert exc_info.value.error_code == "invalid_file_type"
        assert expected_msg in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_file(self):
        """Test validation of empty files"""
        empty_bytes = BytesIO()
        file = UploadFile(filename="empty.png", file=empty_bytes)
        
        with pytest.raises(ValidationException) as exc_info:
            await validate_image(file)
            
        assert exc_info.value.error_code == "invalid_image_type"
        assert "Unable to identify image" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_oversized_file(self, test_image: Image.Image):
        """Test validation of files exceeding size limit"""
        # Create a large file that exceeds MAX_FILE_SIZE
        test_bytes = BytesIO(b'0' * (MAX_FILE_SIZE + 1))
        file = UploadFile(filename="test.png", file=test_bytes)
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_image(file)
            
        assert exc_info.value.status_code == 400
        assert "File size exceeds" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_corrupt_image(self):
        """Test validation of corrupt image files"""
        # Create a file that looks like PNG but is corrupt
        corrupt_bytes = BytesIO(b'\x89PNG\r\n\x1a\n' + b'corrupt data')
        file = UploadFile(filename="corrupt.png", file=corrupt_bytes)
        
        with pytest.raises(ValidationException) as exc_info:
            await validate_image(file)
            
        assert exc_info.value.error_code == "invalid_image_type"
        assert "Unable to identify image" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize('extension', [
        '.txt',
        '.pdf',
        '.json',
        '.html'
    ])
    async def test_non_image_file(self, extension: str):
        """Test validation of files that are not images"""
        test_bytes = BytesIO(b'Not an image content')
        file = UploadFile(filename=f"test{extension}", file=test_bytes)
        
        with pytest.raises(ValidationException) as exc_info:
            await validate_image(file)
            
        assert exc_info.value.error_code == "invalid_file_type"
        assert "File format not supported" in str(exc_info.value)
