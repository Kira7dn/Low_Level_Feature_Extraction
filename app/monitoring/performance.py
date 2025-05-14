import time
import logging
import functools
import os
import json
from typing import Callable, Any, Dict

class PerformanceMonitor:
    """
    Performance monitoring utility for image processing pipeline
    
    Provides decorators and utilities to track:
    - Method execution times
    - Resource utilization
    - Performance metrics logging
    """
    
    _METRICS_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'performance_metrics.json')
    
    @classmethod
    def _ensure_log_directory(cls):
        """Ensure the logs directory exists"""
        log_dir = os.path.dirname(cls._METRICS_LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
    
    @classmethod
    def track_performance(cls, log_level=logging.INFO):
        """
        Decorator to track method performance
        
        Args:
            log_level (int): Logging level for performance metrics
        
        Returns:
            Callable: Decorated function with performance tracking
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Prepare performance tracking
                start_time = time.perf_counter()
                memory_before = os.getpid()
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Calculate performance metrics
                    end_time = time.perf_counter()
                    memory_after = os.getpid()
                    
                    # Calculate metrics
                    execution_time = end_time - start_time
                    memory_usage = memory_after - memory_before
                    
                    # Log performance metrics
                    logging.log(log_level, 
                        f"Performance: {func.__name__} "
                        f"Execution Time: {execution_time:.4f} seconds, "
                        f"Memory Usage: {memory_usage} bytes"
                    )
                    
                    # Store metrics
                    cls._store_performance_metrics({
                        'method': func.__name__,
                        'execution_time': execution_time,
                        'memory_usage': memory_usage,
                        'timestamp': time.time()
                    })
                    
                    return result
                
                except Exception as e:
                    logging.error(f"Performance tracking error in {func.__name__}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    @classmethod
    def _store_performance_metrics(cls, metrics: Dict[str, Any]):
        """
        Store performance metrics in a JSON log file
        
        Args:
            metrics (Dict[str, Any]): Performance metrics to store
        """
        cls._ensure_log_directory()
        
        try:
            # Read existing metrics
            if os.path.exists(cls._METRICS_LOG_FILE):
                with open(cls._METRICS_LOG_FILE, 'r') as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = []
            
            # Append new metrics
            existing_metrics.append(metrics)
            
            # Limit log size (keep last 1000 entries)
            existing_metrics = existing_metrics[-1000:]
            
            # Write back to file
            with open(cls._METRICS_LOG_FILE, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
        
        except Exception as e:
            logging.error(f"Error storing performance metrics: {e}")
    
    @classmethod
    def analyze_performance_metrics(cls, method_name: str = None) -> Dict[str, Any]:
        """
        Analyze stored performance metrics
        
        Args:
            method_name (str, optional): Specific method to analyze
        
        Returns:
            Dict[str, Any]: Performance analysis results
        """
        try:
            with open(cls._METRICS_LOG_FILE, 'r') as f:
                metrics = json.load(f)
            
            # Filter by method if specified
            if method_name:
                metrics = [m for m in metrics if m['method'] == method_name]
            
            # Calculate basic statistics
            if not metrics:
                return {}
            
            return {
                'total_runs': len(metrics),
                'avg_execution_time': sum(m['execution_time'] for m in metrics) / len(metrics),
                'avg_memory_usage': sum(m['memory_usage'] for m in metrics) / len(metrics),
                'max_execution_time': max(m['execution_time'] for m in metrics),
                'max_memory_usage': max(m['memory_usage'] for m in metrics)
            }
        
        except FileNotFoundError:
            return {}
        except Exception as e:
            logging.error(f"Error analyzing performance metrics: {e}")
            return {}
