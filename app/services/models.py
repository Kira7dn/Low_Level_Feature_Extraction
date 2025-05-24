from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

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
