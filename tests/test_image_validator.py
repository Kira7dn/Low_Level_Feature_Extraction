import pytest
from fastapi import UploadFile, HTTPException
from app.utils.error_handler import ValidationException
from io import BytesIO
from app.utils.image_validator import validate_image, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from PIL import Image, ImageDraw

@pytest.mark.asyncio
async def test_valid_image():
    # Create a test image file
    # Create a minimal valid PNG file
    test_img = Image.new('RGB', (10, 10), color='white')
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([0, 0, 9, 9], outline='black')

    test_image = BytesIO()
    test_img.save(test_image, format='PNG')
    test_image.seek(0)
    file = UploadFile(filename="test.png", file=test_image)
    
    # Validate the image
    contents = await validate_image(file)
    assert contents is not None
    assert len(contents) > 0
    assert contents == test_image.getvalue()

@pytest.mark.asyncio
async def test_invalid_extension():
    # Test unsupported file extension
    test_image = BytesIO(b'fake image content')
    file = UploadFile(filename="test.gif", file=test_image)
    
    with pytest.raises(ValidationException) as exc_info:
        await validate_image(file)
    
    assert exc_info.value.error_code == "invalid_file_type"
    assert exc_info.value.detail == "File format not supported. Allowed formats: jpeg, jpg, png, webp, bmp"

@pytest.mark.asyncio
async def test_file_size_limit():
    # Create a file larger than MAX_FILE_SIZE
    large_file = BytesIO(b'0' * (MAX_FILE_SIZE + 1))
    file = UploadFile(filename="large.png", file=large_file)
    
    with pytest.raises(HTTPException) as exc_info:
        await validate_image(file)
    
    assert exc_info.value.status_code == 400
    assert "File size exceeds the limit" in exc_info.value.detail
