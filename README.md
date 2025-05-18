# Low-Level Feature Extraction API

## Features

- **Image Analysis**: Extract colors, text, and fonts from images
- **Multiple Input Sources**: 
  - File uploads
  - Public URLs
- **Advanced Processing**:
  - Automatic image preprocessing
  - Multiple feature extraction
  - Configurable processing modes

## Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR (for text extraction)
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
5. Update the `.env` file with your configuration

## API Endpoints

### Analyze Image

#### Using File Upload
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "accept: application/json" \
     -F "file=@image.jpg" \
     -F "preprocessing=auto" \
     -F "features=colors" \
     -F "features=text"
```

#### Using Image URL
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com/image.jpg",
       "preprocessing": "auto",
       "features": ["colors", "text"]
     }'
```

#### Request Parameters

- `file` (File, optional): Image file to upload (max 10MB)
- `url` (string, optional): URL of the image to process
- `preprocessing` (string): Image preprocessing mode
  - `none`: No preprocessing (fastest, least accurate)
  - `auto`: Automatic preprocessing (recommended)
  - `high_quality`: Enhanced preprocessing (slower)
  - `performance`: Optimized for speed (may reduce accuracy)
- `features` (array of strings, optional): List of features to extract
  - `colors`: Extract dominant colors and color scheme
  - `text`: Extract text content and metadata using OCR
  - `fonts`: Identify font properties and styles

## Image Processing Pipeline

### Overview
The image processing pipeline provides advanced image optimization capabilities, supporting multiple input types and intelligent transformations.

### Quick Start

#### Basic Image Optimization
```python
from app.services.image_processor import ImageProcessor

# Basic usage - auto-resize, compress, and convert to WebP
optimized_image = ImageProcessor.auto_process_image(input_image)
```

#### Advanced Image Processing
```python
# Full control over image transformation
optimized_image = ImageProcessor.auto_process_image(
    input_image,
    max_width=800,      # Resize to max 800px width
    max_height=600,     # Resize to max 600px height
    target_format='avif',  # Convert to AVIF format
    quality=90          # High-quality compression
)
```

### Supported Input Types
- PIL Images
- OpenCV Images
- Image Bytes

### CDN URL Generation
```python
from app.frontend_utils import ImageCdnManager

# Create CDN manager
cdn_manager = ImageCdnManager('https://cdn.example.com')

# Generate optimized image URL
cdn_url = cdn_manager.get_optimized_image_url(
    '/path/to/image.jpg', 
    {'resize': {'width': 800, 'height': 600}}
)
```

### Responsive Image URLs
```python
# Generate multiple image URLs for different screen sizes
responsive_urls = cdn_manager.get_responsive_image_urls('/path/to/image.jpg')
```

### Performance Characteristics
- Supports multiple input types
- Optimizes image size and quality
- Consistent processing time (<2 seconds)
- Reduces image file size while maintaining visual quality

### Dependencies
- Pillow (PIL)
- OpenCV
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Example Usage

Check out the comprehensive example in `examples/image_processing_demo.py`:

```python
# Basic Image Processing
basic_processed = ImageProcessor.auto_process_image(input_image)

# Custom Image Processing
custom_processed = ImageProcessor.auto_process_image(
    input_image,
    max_width=800,      # Resize to max 800px width
    max_height=600,     # Resize to max 600px height
    target_format='avif',  # Convert to AVIF format
    quality=90          # High-quality compression
)

# CDN URL Generation
cdn_manager = ImageCdnManager('https://cdn.example.com')
cdn_url = cdn_manager.get_optimized_image_url(
    '/path/to/image.jpg', 
    {'resize': {'width': 800, 'height': 600}}
)
```

To run the full demo:
```bash
python examples/image_processing_demo.py
```

### API

## Project Overview
A FastAPI-based backend service for extracting key visual elements from design images.

## Prerequisites
- Python 3.8+
- Windows 10/11
- PowerShell

## Setup Instructions

1. Create a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Run the application
```powershell
uvicorn app.main:app --reload
```
## Running Tests

```powershell
pytest tests/
```

## Features
- Color palette extraction
- Font detection
  - Detect font family
  - Estimate font weight (Light, Regular, Bold)
  - Determine font size
- Shape analysis
- Shadow analysis
- Text recognition

## API Endpoints
- /: Root endpoint
- /extract-colors: Extract color palette
- /extract-fonts: Detect font details
  - Supports PNG, JPEG, BMP formats
  - Returns font family, weight, and size
- /extract-shapes: Analyze design shapes
- /extract-shadows: Analyze shadow properties
- /extract-text: Recognize text in images

## API Documentation

Access Swagger UI at http://localhost:8000/docs

## Font Detection Limitations

The font detection service has the following limitations:
- Best performance with clear, high-contrast text
- Limited accuracy with handwritten or stylized fonts
- May struggle with very small or very large text regions
- Supports primarily Latin character sets
- Font family detection is an estimation, not a guarantee

### Expected Accuracy
- Font Weight: 85-90% accuracy
- Font Size: ±2-3 points precision
- Font Family: Identification confidence varies

## Troubleshooting
- Ensure Python and pip are in your system PATH
- Check virtual environment activation
- Verify all dependencies are installed correctly