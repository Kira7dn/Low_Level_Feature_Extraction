import numpy as np
import cv2
import tempfile

def test_cv2_imwrite():
    # Create a simple white image
    height, width = 400, 600
    channels = 3
    image = np.zeros((height, width, channels), dtype=np.uint8)
    image[:] = 255  # Fill with white

    # Print detailed image information
    print("Original Image Details:")
    print(f"Image type: {type(image)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image shape: {image.shape}")
    print(f"Image is contiguous: {image.flags['C_CONTIGUOUS']}")
    print(f"Image is writeable: {image.flags['WRITEABLE']}")
    print(f"Image strides: {image.strides}")
    print(f"Image memory layout: {image.__array_interface__}")
    print(f"Image memory address: {image.__array_interface__['data'][0]}")
    
    # Detailed color channel check
    print("\nColor Channel Check:")
    print(f"First pixel values (RGB): {image[0, 0, :]}")

    # Ensure the image is contiguous
    image = np.ascontiguousarray(image)

    # Use a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        test_image_path = temp_file.name
        
        try:
            # Convert to BGR (OpenCV's default color space)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Print details of BGR image
            print("\nBGR Image Details:")
            print(f"Image type: {type(image_bgr)}")
            print(f"Image dtype: {image_bgr.dtype}")
            print(f"Image shape: {image_bgr.shape}")
            print(f"Image is contiguous: {image_bgr.flags['C_CONTIGUOUS']}")
            print(f"Image is writeable: {image_bgr.flags['WRITEABLE']}")
            print(f"Image strides: {image_bgr.strides}")
            print(f"First pixel values (BGR): {image_bgr[0, 0, :]}")
            
            # Write the image
            success = cv2.imwrite(test_image_path, image_bgr)
            
            if not success:
                print(f"Failed to write image to {test_image_path}")
                return False
            
            print(f"\nSuccessfully wrote image to {test_image_path}")
            return True
        
        except Exception as e:
            print(f"Error writing image: {e}")
            return False

# Run the test
test_cv2_imwrite()
