"""
Utilities - Common functions, logging, configuration, and decorators
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional
import platform

def setup_logging() -> logging.Logger:
    """Setup application logging"""
    
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    logger = logging.getLogger('data_analyst_agent')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get logger for specific module"""
    return logging.getLogger(f'data_analyst_agent.{name}')

def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    
    config = {
        # API Configuration
        'max_request_timeout': int(os.getenv('REQUEST_TIMEOUT', '180')),
        'scraping_timeout': int(os.getenv('SCRAPING_TIMEOUT', '30')),
        'llm_timeout': int(os.getenv('LLM_TIMEOUT', '45')),
        
        # Rate Limiting
        'max_requests_per_minute': int(os.getenv('MAX_REQUESTS_PER_MINUTE', '12')),
        'min_request_interval': float(os.getenv('MIN_REQUEST_INTERVAL', '4.0')),
        
        # Data Processing
        'max_data_rows': int(os.getenv('MAX_DATA_ROWS', '200')),
        'max_plot_size_kb': int(os.getenv('MAX_PLOT_SIZE_KB', '80')),
        
        # System Info
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'environment': os.getenv('ENVIRONMENT', 'development'),
        
        # API Keys (validation only)
        'has_gemini_key': bool(os.getenv('GEMINI_API_KEY')),
        
        # Debug Mode
        'debug': os.getenv('DEBUG', 'false').lower() == 'true'
    }
    
    return config

def timer(func):
    """Decorator to time function execution"""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger('timer')
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger('timer')
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__call__'):
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
    
    return sync_wrapper

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int with default"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "Unknown") -> str:
    """Safely convert value to string with default"""
    try:
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def validate_base64_image(data: str) -> bool:
    """Validate base64 image data"""
    try:
        if not data.startswith("data:image/"):
            return False
        
        # Extract base64 part
        if ';base64,' in data:
            base64_part = data.split(';base64,')[1]
        else:
            return False
        
        # Try to decode
        import base64
        decoded = base64.b64decode(base64_part)
        
        # Check size (should be reasonable for an image)
        return 100 <= len(decoded) <= 200000  # 100 bytes to 200KB
        
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe use"""
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:90] + ext
    
    return filename

def get_file_type(filename: str) -> str:
    """Get file type from filename"""
    _, ext = os.path.splitext(filename.lower())
    
    type_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.txt': 'text',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.gif': 'image',
        '.pdf': 'document',
        '.xlsx': 'excel',
        '.xls': 'excel'
    }
    
    return type_map.get(ext, 'unknown')

def format_bytes(bytes_value: int) -> str:
    """Format bytes as human readable string"""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.1f} GB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds as human readable string"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'total': sum(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics statistics"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        """
        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_make_call(self) -> bool:
        """Check if a call can be made"""
        now = time.time()
        
        # Remove old calls
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def make_call(self):
        """Register a call"""
        self.calls.append(time.time())
    
    def time_until_next_call(self) -> float:
        """Time until next call is allowed"""
        if self.can_make_call():
            return 0.0
        
        if not self.calls:
            return 0.0
        
        oldest_call = min(self.calls)
        return self.time_window - (time.time() - oldest_call)

def create_error_response(error_type: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        'error': {
            'type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    if details:
        response['error']['details'] = details
    
    return response

def validate_environment() -> Dict[str, Any]:
    """Validate environment setup"""
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ is required")
    
    # Check required environment variables
    if not os.getenv('GEMINI_API_KEY'):
        issues.append("GEMINI_API_KEY environment variable is required")
    
    # Check optional but recommended settings
    if not os.getenv('LOG_LEVEL'):
        warnings.append("LOG_LEVEL not set, using default INFO")
    
    # Check system resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 1:
            warnings.append(f"Low memory: {memory_gb:.1f}GB (recommend 2GB+)")
    except ImportError:
        warnings.append("psutil not available for memory monitoring")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'environment': os.getenv('ENVIRONMENT', 'unknown'),
        'python_version': platform.python_version()
    }

def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check"""
    
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Environment validation
    env_check = validate_environment()
    health['checks']['environment'] = env_check
    
    if not env_check['valid']:
        health['status'] = 'unhealthy'
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        health['checks']['memory'] = {
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'status': 'ok' if memory.percent < 90 else 'warning'
        }
        
        if memory.percent > 95:
            health['status'] = 'degraded'
            
    except ImportError:
        health['checks']['memory'] = {'status': 'unknown', 'error': 'psutil not available'}
    
    # Disk space check (if applicable)
    try:
        disk = psutil.disk_usage('/')
        health['checks']['disk'] = {
            'usage_percent': (disk.used / disk.total) * 100,
            'free_gb': disk.free / (1024**3),
            'status': 'ok' if disk.free > 1024**3 else 'warning'  # 1GB free
        }
    except:
        health['checks']['disk'] = {'status': 'unknown'}
    
    return health

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Default rate limiter for LLM calls
llm_rate_limiter = RateLimiter(max_calls=12, time_window=60)  # 12 calls per minute
