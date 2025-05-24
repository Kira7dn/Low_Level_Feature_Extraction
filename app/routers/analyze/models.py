from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


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


# Import these here to avoid circular imports
from app.services.models import ColorFeatures, TextFeatures, FontFeatures  # noqa: E402
