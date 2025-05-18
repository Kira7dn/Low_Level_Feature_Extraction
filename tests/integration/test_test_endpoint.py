"""Test the test endpoint to verify route registration."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_test_endpoint():
    """Test that the test endpoint returns the expected response."""
    response = client.get("/api/v1/analyze/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Test endpoint is working!"}
