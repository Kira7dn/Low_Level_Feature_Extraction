import os
import time
import json
import logging
import pytest
from app.monitoring.performance import PerformanceMonitor

# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

class TestPerformanceMonitor:
    @PerformanceMonitor.track_performance()
    def sample_method(self, sleep_time=0.1):
        """Sample method for performance testing"""
        time.sleep(sleep_time)
        return True
    
    def test_performance_tracking(self):
        """Test performance tracking decorator"""
        # Clear existing metrics
        metrics_file = PerformanceMonitor._METRICS_LOG_FILE
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        
        # Run method multiple times
        for _ in range(5):
            self.sample_method()
        
        # Verify metrics were logged
        assert os.path.exists(metrics_file), "Performance metrics file was not created"
        
        # Read and verify metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert len(metrics) == 5, "Incorrect number of metrics logged"
        
        # Verify each metric entry
        for metric in metrics:
            assert 'method' in metric
            assert 'execution_time' in metric
            assert 'memory_usage' in metric
            assert 'timestamp' in metric
            assert metric['method'] == 'sample_method'
    
    def test_performance_analysis(self):
        """Test performance metrics analysis"""
        # Clear existing metrics
        metrics_file = PerformanceMonitor._METRICS_LOG_FILE
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        
        # Run method with different sleep times
        self.sample_method(0.1)
        self.sample_method(0.2)
        self.sample_method(0.3)
        
        # Analyze metrics
        analysis = PerformanceMonitor.analyze_performance_metrics('sample_method')
        
        # Verify analysis results
        assert 'total_runs' in analysis
        assert 'avg_execution_time' in analysis
        assert 'avg_memory_usage' in analysis
        assert 'max_execution_time' in analysis
        assert 'max_memory_usage' in analysis
        
        assert analysis['total_runs'] == 3
        
        # Verify execution times are roughly correct
        assert 0.1 < analysis['avg_execution_time'] < 0.4
        assert 0.3 < analysis['max_execution_time'] < 0.5
    
    def test_performance_tracking_error_handling(self):
        """Test performance tracking with an error-raising method"""
        @PerformanceMonitor.track_performance()
        def error_method():
            raise ValueError("Test error")
        
        # Verify error is still raised
        with pytest.raises(ValueError, match="Test error"):
            error_method()
