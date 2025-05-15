from prometheus_client import Counter, Histogram, Gauge
import time

# Feature Extraction Metrics
# Counters for tracking total number of requests and successful/failed extractions
TEXT_EXTRACTION_TOTAL = Counter(
    'text_extraction_total', 
    'Total number of text extraction requests'
)
TEXT_EXTRACTION_SUCCESS = Counter(
    'text_extraction_success', 
    'Number of successful text extractions'
)
TEXT_EXTRACTION_FAILURES = Counter(
    'text_extraction_failures', 
    'Number of failed text extractions'
)

# Histogram to track processing time for different feature extraction methods
EXTRACTION_PROCESSING_TIME = Histogram(
    'feature_extraction_processing_seconds', 
    'Processing time for feature extraction',
    ['feature_type']
)

# Gauge for tracking current image processing load
CURRENT_IMAGE_PROCESSING_LOAD = Gauge(
    'current_image_processing_load', 
    'Number of images currently being processed'
)

# Decorator for tracking metrics
def track_feature_extraction(feature_type):
    """
    Decorator to track metrics for feature extraction methods
    
    Args:
        feature_type (str): Type of feature extraction (e.g., 'text', 'shape', 'font')
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Increment total requests
            TEXT_EXTRACTION_TOTAL.inc()
            
            # Increase current processing load
            CURRENT_IMAGE_PROCESSING_LOAD.inc()
            
            start_time = time.time()
            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Track successful extraction
                TEXT_EXTRACTION_SUCCESS.inc()
                
                return result
            except Exception as e:
                # Track failed extraction
                TEXT_EXTRACTION_FAILURES.inc()
                raise
            finally:
                # Record processing time
                processing_time = time.time() - start_time
                EXTRACTION_PROCESSING_TIME.labels(feature_type=feature_type).observe(processing_time)
                
                # Decrease current processing load
                CURRENT_IMAGE_PROCESSING_LOAD.dec()
        
        return wrapper
    return decorator

# Method to reset metrics if needed
def reset_metrics():
    """
    Reset all custom metrics to their initial state
    """
    TEXT_EXTRACTION_TOTAL._metrics.clear()
    TEXT_EXTRACTION_SUCCESS._metrics.clear()
    TEXT_EXTRACTION_FAILURES._metrics.clear()
    EXTRACTION_PROCESSING_TIME._metrics.clear()
    CURRENT_IMAGE_PROCESSING_LOAD.set(0)
