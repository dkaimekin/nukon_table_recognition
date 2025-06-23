import time
import logging
from functools import wraps
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_predictions(func):
    """Decorator to log prediction requests"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Log successful prediction
            duration = time.time() - start_time
            logger.info(f"Prediction successful - Duration: {duration:.3f}s")
            
            return result
            
        except Exception as e:
            # Log failed prediction
            duration = time.time() - start_time
            logger.error(f"Prediction failed - Duration: {duration:.3f}s - Error: {str(e)}")
            raise
    
    return wrapper

# Simple metrics storage (in production, use proper monitoring)
class SimpleMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.start_time = datetime.now()
    
    def record_request(self, success: bool, processing_time: float):
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def get_stats(self):
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_processing_time = (
            self.total_processing_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "average_processing_time": avg_processing_time
        }

# Global metrics instance
metrics = SimpleMetrics()