import sys
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_hello_endpoint():
    response = client.get("/hello")
    assert response.status_code == 200
    response_text = response.text.strip('"')  # Remove quotes from JSON string
    assert response_text.startswith("Hello, World! - Test ")
