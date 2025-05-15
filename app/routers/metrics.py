from fastapi import APIRouter, Request
from prometheus_client import (
    generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge,
    CollectorRegistry, multiprocess
)
from starlette.responses import Response
from typing import Dict
import psutil
import time

# Create a custom registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# Response latency
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint'],
    registry=registry
)

# Error rate
ERROR_COUNT = Counter(
    'http_errors_total',
    'Total number of HTTP errors',
    ['endpoint', 'error_type'],
    registry=registry
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['endpoint'],
    registry=registry
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['endpoint'],
    registry=registry
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

router = APIRouter(prefix="/metrics", tags=["Monitoring"])

@router.get("/", response_model=Dict)
async def metrics():
    """
    Endpoint to expose Prometheus metrics.
    
    Returns:
        Response: Prometheus metrics in plain text format
    """
    # Update system metrics
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    CPU_USAGE.set(psutil.cpu_percent())
    
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/health", response_model=Dict)
async def health_check():
    """
    Health check endpoint that returns system status.
    
    Returns:
        Dict: System health metrics
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "status": "healthy",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_used": memory_info.rss,
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads()
        },
        "timestamp": time.time()
    }
