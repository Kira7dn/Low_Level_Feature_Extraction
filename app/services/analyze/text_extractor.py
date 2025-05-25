import cv2
import pytesseract
import re
import numpy as np
from typing import Any, Dict, List, Union, Optional
import time

from app.api.v1.models.analyze import (
    TextFeatures,
    FontFeatures,
)  # Import the new models


class TextExtractor:
    @classmethod
    def preprocess_image(cls, image):
        """Simplified image preprocessing for better text detection

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Preprocessed binary image optimized for OCR (black text on white background)
        """
        # Convert to grayscale if not already
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize image if it's too small
        height, width = gray.shape[:2]
        if height < 30 or width < 100:
            scale = max(2, 300 / width, 100 / height)
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        # Simple thresholding - no blur, no adaptive thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure black text on white background
        if np.mean(binary) > 127:  # If mostly white (inverted)
            binary = cv2.bitwise_not(binary)

        return binary

    @classmethod
    def extract_text(
        cls,
        image,
        lang: str = "eng",
        config: str = None,
        confidence_threshold: float = 0.1,
    ) -> TextFeatures:
        """Extract text from image using Tesseract OCR with enhanced parameters

        Args:
            image: Input image (numpy array)
            lang: Language for OCR (default: English)
            config: Tesseract configuration (defaults to optimized settings)
            confidence_threshold: Minimum confidence score (0-1) for text to be included

        Returns:
            TextFeatures: Object containing extracted text and metadata
        """
        start_time = time.time()

        # Default Tesseract configuration - using a list of arguments instead of a string
        config_args = [
            "--oem",
            "3",  # Use LSTM OCR Engine
            "--psm",
            "6",  # Assume a single uniform block of text
            "-c",
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.!?():;'\"- ",
            "-c",
            "preserve_interword_spaces=1",
            "--dpi",
            "300",
        ]

        # Preprocess the image
        processed_img = cls.preprocess_image(image)

        # Extract text with confidence
        try:
            # First try with the default configuration
            data = pytesseract.image_to_data(
                processed_img,
                lang=lang,
                config=" ".join(config_args),  # Convert list to space-separated string
                output_type=pytesseract.Output.DICT,
            )

            # If no text found, try with a different page segmentation mode
            if not any(data["text"]):
                config_args[config_args.index("--psm") + 1] = (
                    "11"  # Try sparse text mode
                )
                data = pytesseract.image_to_data(
                    processed_img,
                    lang=lang,
                    config=" ".join(config_args),
                    output_type=pytesseract.Output.DICT,
                )

        except Exception as e:
            error_msg = str(e)
            # Try a simpler configuration if the first one fails
            try:
                data = pytesseract.image_to_data(
                    processed_img,
                    lang=lang,
                    config="--psm 6",
                    output_type=pytesseract.Output.DICT,
                )
            except Exception as inner_e:
                error_msg = f"{error_msg}; Fallback config also failed: {str(inner_e)}"
                return TextFeatures(
                    lines=[],
                    details=[],
                    metadata={
                        "error": error_msg,
                        "success": False,
                        "timestamp": time.time(),
                        "confidence": 0.0,
                        "processing_time": time.time() - start_time,
                    },
                )

        # Process the results
        text_blocks = []
        for i in range(len(data["text"])):
            conf = float(data["conf"][i]) / 100.0  # Convert to 0-1 range
            text = data["text"][i].strip()

            if conf >= confidence_threshold and text:
                text_blocks.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "bbox": {
                            "x": int(data["left"][i]),
                            "y": int(data["top"][i]),
                            "width": int(data["width"][i]),
                            "height": int(data["height"][i]),
                        },
                        "block_num": int(data["block_num"][i]),
                        "line_num": int(data["line_num"][i]),
                        "word_num": int(data["word_num"][i]),
                    }
                )

        # Group text by lines
        lines = {}
        for block in text_blocks:
            line_key = f"{block['block_num']}-{block['line_num']}"
            if line_key not in lines:
                lines[line_key] = {
                    "text": [],
                    "confidence": [],
                    "bbox": {
                        "x": block["bbox"]["x"],
                        "y": block["bbox"]["y"],
                        "width": block["bbox"]["width"],
                        "height": block["bbox"]["height"],
                    },
                }
            lines[line_key]["text"].append(block["text"])
            lines[line_key]["confidence"].append(block["confidence"])

        # Format lines and details for the response
        lines_list = []
        details_list = []

        # Sort lines by Y position (top to bottom)
        sorted_lines = sorted(lines.values(), key=lambda x: x["bbox"]["y"])

        for line in sorted_lines:
            line_text = " ".join(line["text"])
            line_conf = sum(line["confidence"]) / len(line["confidence"])

            # Add to lines list (just the text)
            lines_list.append(line_text)

            # Add to details list
            details_list.append(
                {"text": line_text, "confidence": line_conf, "bbox": line["bbox"]}
            )

        # Process text and return results
        processed = cls.postprocess_text(" ".join(lines_list), confidence_threshold)

        # Ensure we have the expected structure
        if not isinstance(processed, dict):
            processed = {}

        # Create TextFeatures with default values if needed
        return TextFeatures(
            lines=processed.get("lines", lines_list),
            details=processed.get("details", details_list),
            metadata={
                "confidence": processed.get("metadata", {}).get(
                    "confidence",
                    (
                        sum(line["confidence"] for line in details_list)
                        / len(details_list)
                        if details_list
                        else 0.0
                    ),
                ),
                "success": bool(processed.get("lines", lines_list)),
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
            },
        )

    @classmethod
    def extract_from_file(cls, file_path: str, **kwargs) -> TextFeatures:
        """Extract text from an image file

        Args:
            file_path: Path to the image file
            **kwargs: Additional arguments to pass to extract_text

        Returns:
            TextFeatures: Object containing extracted text and metadata
        """
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image from {file_path}")

            # Extract text
            result = cls.extract_text(image, **kwargs)

            # Ensure we have a valid TextFeatures object
            if not isinstance(result, TextFeatures):
                # If result is a dict, convert it to TextFeatures
                if isinstance(result, dict):
                    return TextFeatures(
                        lines=result.get("lines", []),
                        details=result.get("details", result.get("blocks", [])),
                        metadata={
                            "confidence": result.get("metadata", {}).get(
                                "confidence",
                                result.get("metadata", {}).get("avg_confidence", 0.0),
                            ),
                            "success": bool(result.get("lines")),
                            "timestamp": time.time(),
                            "processing_time": result.get("metadata", {}).get(
                                "processing_time", 0.0
                            ),
                        },
                    )
                else:
                    # If result is not a dict, create a new TextFeatures with defaults
                    return TextFeatures(
                        lines=[],
                        details=[],
                        metadata={
                            "error": "Invalid result format",
                            "success": False,
                            "timestamp": time.time(),
                            "confidence": 0.0,
                            "processing_time": 0.0,
                        },
                    )
            return result

        except Exception as e:
            return TextFeatures(
                lines=[],
                details=[],
                metadata={
                    "error": str(e),
                    "success": False,
                    "timestamp": time.time(),
                    "confidence": 0.0,
                    "processing_time": 0.0,
                },
            )

    @staticmethod
    def postprocess_text(
        text: str, confidence_threshold: float = 0.6
    ) -> Dict[str, Union[List[str], List[Dict[str, Union[str, float]]]]]:
        """
        Clean and structure extracted text

        Args:
            text: Raw extracted text

        Returns:
            List of cleaned text lines
        """
        # If no text, return empty dictionary
        if not text:
            return {
                "lines": [],
                "details": [],
                "metadata": {"confidence": 0.0, "timestamp": time.time()},
            }

        # Split by newlines and remove empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Process and clean lines
        processed_lines = []
        for line in lines:
            # Normalize character substitutions
            substitutions = {
                "0": "O",
                "1": "I",
                "5": "S",
                "—": "-",
                "–": "-",
                """: "'", """: "'",
                '"': '"',
                '"': '"',
            }
            for original, replacement in substitutions.items():
                line = line.replace(original, replacement)

            # Remove noise characters
            line = "".join(char for char in line if char.isprintable())
            line = re.sub(r"\s+", " ", line).strip()

            # Keep only alphanumeric chars, spaces, and basic punctuation
            final_line = re.sub(r'[^\w\s.,!?:;\'"\-]', "", line)

            if final_line:
                processed_lines.append(
                    {"text": final_line, "original": line, "length": len(final_line)}
                )

        return {
            "lines": [line["text"] for line in processed_lines],
            "details": processed_lines,
            "metadata": {
                "confidence": 1.0,  # Text post-processing always succeeds
                "timestamp": time.time(),
            },
        }
