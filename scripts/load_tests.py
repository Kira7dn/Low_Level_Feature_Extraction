from locust import HttpUser, task, between
import base64
import os

class FeatureExtractionUser(HttpUser):
    """
    Simulated user for load testing the Feature Extraction API
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def __init__(self, *args, **kwargs):
        """
        Initialize test user with sample images
        """
        super().__init__(*args, **kwargs)
        self.sample_images = self._load_sample_images()
    
    def _load_sample_images(self):
        """
        Load sample images for testing
        
        :return: List of base64 encoded images
        """
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'sample_images')
        images = []
        
        for filename in os.listdir(sample_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                with open(os.path.join(sample_dir, filename), 'rb') as f:
                    images.append(base64.b64encode(f.read()).decode('utf-8'))
        
        return images or [None]  # Fallback to None if no images found
    
    @task(3)  # Weighted task, more frequent
    def upload_and_extract_text(self):
        """
        Simulate uploading an image and extracting text
        """
        image = self._get_random_image()
        if image:
            self.client.post("/upload", json={"image": image})
            self.client.post("/extract-text", json={"image": image})
    
    @task(2)
    def generate_cdn_url(self):
        """
        Simulate CDN URL generation requests
        """
        image = self._get_random_image()
        if image:
            self.client.post("/generate-cdn-url", json={
                "image": image,
                "max_width": 800,
                "max_height": 600,
                "quality": 85
            })
    
    @task(1)
    def get_performance_metrics(self):
        """
        Simulate requesting performance metrics
        """
        self.client.get("/performance-metrics")
    
    def _get_random_image(self):
        """
        Get a random sample image
        
        :return: Base64 encoded image or None
        """
        import random
        return random.choice(self.sample_images)

def main():
    """
    Entry point for load testing
    """
    # Note: Actual execution is handled by Locust CLI
    pass

if __name__ == "__main__":
    main()
