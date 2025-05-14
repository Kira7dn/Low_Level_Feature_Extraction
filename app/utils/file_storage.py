import os
import uuid

def save_uploaded_file(file_id: str, file_bytes: bytes, filename: str) -> str:
    """
    Save uploaded file to storage and return file URL.
    
    Args:
        file_id (str): Unique identifier for the file
        file_bytes (bytes): File content
        filename (str): Original filename
    
    Returns:
        str: URL or path to the saved file
    """
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(uploads_dir, unique_filename)
    
    # Save file
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    
    # Return relative URL
    return f"/uploads/{unique_filename}"
