from dataclasses import dataclass
from typing import List, Dict, Optional, Union

@dataclass
class ColorFeatures:
    primary: str
    background: str
    accent: List[str]

@dataclass
class ShapeFeatures:
    total_shapes: int
    rectangles: int
    circles: int
    polygons: int
    borders: Dict[str, str]  # border_type: border_style

@dataclass
class TextFeatures:
    content: List[str]
    languages: List[str]
    text_areas: int

@dataclass
class FontFeatures:
    primary_font: str
    secondary_fonts: List[str]
    font_sizes: List[int]
    font_styles: List[str]

@dataclass
class ShadowFeatures:
    has_shadows: bool
    shadow_types: List[str]
    shadow_colors: List[str]

@dataclass
class TestCase:
    image_path: str
    image_type: str
    colors: ColorFeatures
    shapes: ShapeFeatures
    text: TextFeatures
    fonts: FontFeatures
    shadows: ShadowFeatures
    
    @classmethod
    def create_visa_card_test(cls) -> 'TestCase':
        """Create test case for the VISA card image"""
        return cls(
            image_path="tests/test_images/image.png",
            image_type="png",
            colors=ColorFeatures(
                primary="#CC0000",  # Deep Red
                background="#000000",  # Rich Black
                accent=["#FFFFFF", "#CCCCCC"]  # White and Light Gray
            ),
            shapes=ShapeFeatures(
                total_shapes=2,
                rectangles=2,  # Card shape and chip
                circles=0,
                polygons=0,
                borders={"card": "rounded"}
            ),
            text=TextFeatures(
                content=["VISA", "CARD NUMBER", "9891", "EXP.", "CVV", "Show info"],
                languages=["en"],
                text_areas=6
            ),
            fonts=FontFeatures(
                primary_font="Arial",
                secondary_fonts=["Helvetica", "Sans-serif"],
                font_sizes=[24, 12, 16],  # Different sizes for different text elements
                font_styles=["bold", "regular"]
            ),
            shadows=ShadowFeatures(
                has_shadows=True,
                shadow_types=["drop-shadow"],
                shadow_colors=["rgba(0,0,0,0.3)"]
            )
        )

# Dictionary of test cases
TEST_CASES = {
    "visa_card": TestCase.create_visa_card_test()
}
