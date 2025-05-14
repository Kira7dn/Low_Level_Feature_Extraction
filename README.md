# Low-Level Feature Extraction API

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