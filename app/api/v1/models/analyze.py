from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl,validator


class FeatureType(str, Enum):
    """Enumeration of available feature types for image analysis."""
    COLORS = "colors"
    TEXT = "text"
    FONTS = "fonts"


class FeatureError(BaseModel):
    """Error details for a failed feature extraction.
    
    Attributes:
        code: A unique identifier for the error type in SCREAMING_SNAKE_CASE.
        message: Human-readable explanation of the error.
        severity: Indicates error severity level.
            - 'error': Critical issue that prevents feature extraction.
            - 'warning': Non-critical issue that allows partial processing.
    """
    code: str = Field(..., description="Error code in SCREAMING_SNAKE_CASE")
    message: str = Field(..., description="Human-readable error message")
    severity: Literal["error", "warning"] = Field(
        "error",
        description="Error severity level ('error' or 'warning')",
    )


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process.
    
    Attributes:
        total_features_requested: Total number of features requested for extraction.
            Must be >= 0.
        features_processed: Number of features successfully processed.
            Must be >= 0 and <= total_features_requested.
        features_failed: Number of features that failed to process.
            Must be >= 0 and <= total_features_requested.
        processing_time_ms: Total processing time in milliseconds.
            Includes all feature extraction and preprocessing time.
    """
    total_features_requested: int = Field(
        ..., 
        description="Total number of features requested for extraction",
        ge=0
    )
    features_processed: int = Field(
        ..., 
        description="Number of features successfully processed",
        ge=0
    )
    features_failed: int = Field(
        ..., 
        description="Number of features that failed to process",
        ge=0
    )
    processing_time_ms: float = Field(
        ..., 
        description="Total processing time in milliseconds",
        ge=0.0
    )


class UnifiedAnalysisResponse(BaseModel):
    """Response model for unified image analysis.
    
    This model represents the standardized response format for all image analysis requests,
    containing the extracted features, any errors that occurred, and processing metadata.
    """
    status: Literal["success", "partial", "failure"]
    request_id: str
    features: Dict[
        str,
        Optional[Union[Dict[str, Any], "ColorFeatures", "TextFeatures", "FontFeatures"]],
    ] = Field(..., description="Extracted features keyed by feature name")
    errors: Dict[str, FeatureError] = Field(
        default_factory=dict,
        description="Errors that occurred during processing, keyed by feature name",
    )
    metadata: AnalysisMetadata = Field(..., description="Metadata about the analysis process")

    class Config:
        json_encoders = {
            "ColorFeatures": lambda v: v.dict() if hasattr(v, "dict") else v,
            "TextFeatures": lambda v: v.dict() if hasattr(v, "dict") else v,
            "FontFeatures": lambda v: v.dict() if hasattr(v, "dict") else v,
        }
        json_schema_extra = {
            "example": {
                "status": "success",
                "request_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
                "features": {
                    "colors": {
                        "primary": "#FF5733",
                        "background": "#FFFFFF",
                        "accent": ["#33FF57", "#3357FF"],
                    },
                    "text": {
                        "lines": ["Sample text"],
                        "details": [],
                        "metadata": {},
                    },
                },
                "errors": {},
                "metadata": {
                    "total_features_requested": 2,
                    "features_processed": 2,
                    "features_failed": 0,
                    "processing_time_ms": 123.45,
                },
            }
        }





class TextFeatures(BaseModel):
    """Model for text extraction results."""
    lines: List[str] = Field(
        default_factory=list,
        description="List of extracted text lines"
    )
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed text extraction results including bounding boxes and confidence scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the text extraction process"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextFeatures':
        """Create TextFeatures from a dictionary.
        
        Args:
            data: Dictionary containing text extraction results
            
        Returns:
            TextFeatures instance
        """
        return cls(
            lines=data.get('lines', []),
            details=data.get('details', []),
            metadata={
                'confidence': data.get('metadata', {}).get('confidence', 0.0),
                'success': data.get('metadata', {}).get('success', True),
                'timestamp': data.get('metadata', {}).get('timestamp', 0.0),
                'processing_time': data.get('metadata', {}).get('processing_time', 0.0)
            }
        )


class ColorFeatures(BaseModel):
    """Model for color extraction results."""
    primary: Optional[str] = Field(
        None,
        description="Primary color in hex format (e.g., '#007BFF')",
        pattern=r'^#(?:[0-9a-fA-F]{3}){1,2}$'
    )
    background: Optional[str] = Field(
        None,
        description="Background color in hex format",
        pattern=r'^#(?:[0-9a-fA-F]{3}){1,2}$'
    )
    accent: List[str] = Field(
        default_factory=list,
        description="List of accent colors in hex format"
    )
    
    @validator('accent', each_item=True)
    def validate_accent_colors(cls, v):
        import re
        if not re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', v):
            raise ValueError(f'Invalid hex color code: {v}')
        return v
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the color extraction process"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorFeatures':
        """Create ColorFeatures from a dictionary.
        
        Args:
            data: Dictionary containing color extraction results
            
        Returns:
            ColorFeatures instance
        """
        return cls(
            primary=data.get('primary'),
            background=data.get('background'),
            accent=data.get('accent', []),
            metadata={
                'success': data.get('metadata', {}).get('success', True),
                'timestamp': data.get('metadata', {}).get('timestamp', 0.0),
                'processing_time': data.get('metadata', {}).get('processing_time', 0.0)
            }
        )


class FontFeatures(BaseModel):
    """Model for font analysis results."""
    font_family: Optional[str] = Field(
        None,
        description="Detected font family"
    )
    font_size: Optional[float] = Field(
        None,
        description="Detected font size in points"
    )
    font_style: Optional[str] = Field(
        None,
        description="Detected font style (e.g., 'normal', 'bold', 'italic')"
    )
    confidence: Optional[float] = Field(
        None,
        description="Confidence score for the font detection (0-1)"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FontFeatures':
        """Create FontFeatures from a dictionary.
        
        Args:
            data: Dictionary containing font analysis results
            
        Returns:
            FontFeatures instance
        """
        return cls(
            font_family=data.get('font_family'),
            font_size=data.get('font_size'),
            font_style=data.get('font_style'),
            confidence=data.get('confidence')
        )