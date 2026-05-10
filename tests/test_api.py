from fastapi.testclient import TestClient

from app.api.server import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_endpoint():
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["service"] == "stock-market-automation"

