"""
Basic API tests for the portfolio management system.
"""

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test the root endpoint returns basic app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Portfolio Management API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"
    assert "environment" in data
    assert "dry_run" in data


@pytest.mark.asyncio
async def test_health_check(client):
    """Test the health check endpoint."""
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_portfolio_current_empty(client):
    """Test getting current portfolio when empty."""
    response = await client.get("/portfolio/current")
    # Expect 404 when no positions exist (this is correct behavior)
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "No positions found"


@pytest.mark.asyncio
async def test_portfolio_targets_empty(client):
    """Test getting portfolio targets when empty."""
    response = await client.get("/portfolio/targets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should return empty list when no targets exist
    assert len(data) == 0


@pytest.mark.asyncio
async def test_portfolio_recommendations_empty(client):
    """Test getting recommendations when empty."""
    response = await client.get("/portfolio/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should return empty list when no recommendations exist
    assert len(data) == 0


@pytest.mark.asyncio
async def test_portfolio_orders_empty(client):
    """Test getting order history when empty."""
    response = await client.get("/portfolio/orders")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should return empty list when no orders exist
    assert len(data) == 0


@pytest.mark.asyncio
async def test_portfolio_history_empty(client):
    """Test getting decision history when empty."""
    response = await client.get("/portfolio/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should return empty list when no history exists
    assert len(data) == 0


@pytest.mark.asyncio
async def test_portfolio_performance_empty(client):
    """Test getting performance metrics when empty."""
    response = await client.get("/portfolio/performance")
    # Expect 404 when no performance data exists (this is correct behavior)
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "No performance data available" 