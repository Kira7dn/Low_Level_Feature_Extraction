import cv2
import pytesseract
import re
import numpy as np
from typing import Any, Dict
from typing import List, Dict, Union

class TextExtractor:
    @staticmethod
    def normalize_image(image):
        """
        Normalize image intensity to improve OCR accuracy
        
        Args:
            image: Input image (grayscale)
        
        Returns:
            Normalized image with pixel intensities scaled to 0-255 range
        """
        # Normalize the image to stretch the intensity range
        normalized = cv2.normalize(
            image, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        return normalized

    @classmethod
    def preprocess_image(cls, image):
        """Preprocess image for text detection
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if not already
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize image intensity
        normalized = cls.normalize_image(gray)
        
        # Apply adaptive thresholding for better text segmentation
        binary = cv2.adaptiveThreshold(
            normalized, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply dilation to connect text components
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        return dilated
    
    @classmethod
    def extract_text(cls, image, lang: str = 'eng', config: str = '--oem 3 --psm 6', confidence_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image: OpenCV image to process
            lang: Language for OCR (default: English)
            config: Tesseract configuration options
        
        Returns:
            Dictionary containing extracted text lines and their confidence scores
        """
        # Preprocess the image
        processed_img = cls.preprocess_image(image)
        
        # Extract text with confidence using image_to_data
        data = pytesseract.image_to_data(processed_img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        
        # Process extracted text
        text_results = []
        for i in range(len(data['text'])):
            # Calculate confidence
            confidence = float(data['conf'][i]) / 100
            
            # Skip empty, irrelevant entries, or low-confidence lines
            if (int(data['conf'][i]) > -1 and 
                data['text'][i].strip() and 
                confidence >= confidence_threshold):
                text_results.append({
                    'text': data['text'][i].strip(),
                    'confidence': confidence,
                    'bbox': {
                        'x': int(data['left'][i]),
                        'y': int(data['top'][i]),
                        'width': int(data['width'][i]),
                        'height': int(data['height'][i])
                    }
                })
        
        return {
            'lines': [result['text'] for result in text_results],
            'details': text_results,
            'metadata': {
                'total_lines': len(data['text']),
                'extracted_lines': len(text_results),
                'confidence_threshold': confidence_threshold
            }
        }
    
    @staticmethod
    def postprocess_text(text: str, confidence_threshold: float = 0.6) -> Dict[str, Union[List[str], List[Dict[str, Union[str, float]]]]]:
        """
        Clean and structure extracted text
        
        Args:
            text: Raw extracted text
        
        Returns:
            List of cleaned text lines
        """
        # If no text, return empty dictionary
        if not text:
            return {"lines": [], "details": []}

        # Split by newlines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Process and clean lines
        processed_lines = []
        for line in lines:
            # Normalize character substitutions
            substitutions = {
                '0': 'O', '1': 'I', '5': 'S', 
                '—': '-', '–': '-', 
                ''': "'", ''': "'", 
                '"': '"', '"': '"'
            }
            for original, replacement in substitutions.items():
                line = line.replace(original, replacement)
            
            # Remove noise characters
            line = ''.join(char for char in line if char.isprintable())
            line = re.sub(r'\s+', ' ', line).strip()
            
            # Keep only alphanumeric chars, spaces, and basic punctuation
            final_line = re.sub(r'[^\w\s.,!?:;\'"\-]', '', line)
            
            if final_line:
                processed_lines.append({
                    "text": final_line,
                    "original": line,
                    "length": len(final_line)
                })

        return {
            "lines": [line["text"] for line in processed_lines],
            "details": processed_lines
        }