import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
import asyncio
import json
import os
from datetime import datetime

# Ensure the app module can be imported
import app

# Import test report generator
from .test_report_generator import TestReportGenerator

# Global variable to store test metrics
test_metrics = {
    'timestamp': datetime.now().isoformat(),
    'endpoints': {}
}

# Modify Python import system to help with imports
sys.path.append(os.path.join(project_root, 'app'))

# Add tests directory to Python path
sys.path.append(os.path.dirname(__file__))

import json
import os
from datetime import datetime

# Global test metrics collection
test_metrics = {
    'timestamp': datetime.now().isoformat(),
    'endpoints': {}
}

def _extract_endpoint_from_test_name(test_name: str) -> str:
    """Extract endpoint name from test function name"""
    endpoints = ['colors', 'text', 'shapes', 'shadows', 'fonts']
    for endpoint in endpoints:
        if endpoint in test_name:
            return endpoint
    return None

def pytest_runtest_makereport(item, call):
    """Collect test metrics for report generation"""
    if call.when == 'call':
        test_name = item.name
        endpoint = _extract_endpoint_from_test_name(test_name)
        
        if endpoint:
            performance_time = getattr(item, 'performance_time', None)
            performance_rating = getattr(item, 'performance_rating', 'Unknown')
            
            if endpoint not in test_metrics['endpoints']:
                test_metrics['endpoints'][endpoint] = {
                    'performance': {
                        'times': [],
                        'performance_ratings': []
                    },
                    'validation': {
                        'total_tests': 0,
                        'passed_tests': 0
                    }
                }
            
            endpoint_metrics = test_metrics['endpoints'][endpoint]
            
            if performance_time is not None:
                endpoint_metrics['performance']['times'].append(performance_time)
                endpoint_metrics['performance']['performance_ratings'].append(performance_rating)
            
            endpoint_metrics['validation']['total_tests'] += 1
            
            if call.excinfo is None:
                endpoint_metrics['validation']['passed_tests'] += 1

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate test report at the end of test session"""
    report_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(report_dir, exist_ok=True)
    
    report_filename = os.path.join(
        report_dir, 
        f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(report_filename, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    terminalreporter.write_line("\n===== Test Performance Summary =====")
    for endpoint, data in test_metrics['endpoints'].items():
        terminalreporter.write_line(f"\n{endpoint.upper()} Endpoint:")
        
        perf_times = data['performance']['times']
        if perf_times:
            terminalreporter.write_line(f"  Mean Performance Time: {sum(perf_times)/len(perf_times):.2f}s")
            terminalreporter.write_line(f"  Performance Ratings: {', '.join(data['performance']['performance_ratings'])}")
        
        validation = data['validation']
        terminalreporter.write_line(f"  Total Tests: {validation['total_tests']}")
        terminalreporter.write_line(f"  Passed Tests: {validation['passed_tests']}")
        terminalreporter.write_line(f"  Pass Rate: {validation['passed_tests']/validation['total_tests']*100:.2f}%")
    
    terminalreporter.write_line("\nFull report generated: " + report_filename)

# Configure async event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
