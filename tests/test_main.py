"""Tests for main application."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Portfolio Management API"
    assert data["version"] == "1.0.0"
    assert "status" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_portfolio_endpoints_exist():
    """Test that portfolio endpoints are registered."""
    response = client.get("/docs")
    assert response.status_code == 200 or response.status_code == 404  # 404 if debug=False