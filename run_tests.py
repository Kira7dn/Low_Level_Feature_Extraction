import subprocess
import sys

def run_tests():
    """Run tests with coverage reporting"""
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            '--cov=app', 
            '--cov-report=html', 
            '--cov-report=xml', 
            '--cov-report=term',
            'tests/'
        ], capture_output=False, text=True)
        
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests())
