"""
Pytest configuration and fixtures for the portfolio API tests.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, List
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models import Base
from app.database import get_db
from app.config import settings
from app.services.llm import AdvisorType


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Remove custom event_loop fixture to avoid conflicts with pytest-asyncio


@pytest_asyncio.fixture
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with AsyncSession(test_engine) as session:
        yield session


@pytest_asyncio.fixture
async def client(test_session):
    """Create a test client with database dependency override."""
    
    async def get_test_db():
        yield test_session
    
    app.dependency_overrides[get_db] = get_test_db
    
    async with AsyncClient(
        transport=ASGITransport(app=app), 
        base_url="http://test"
    ) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture(params=[AdvisorType.VALUE, AdvisorType.MACRO])
def advisor_type(request):
    """Parametrized fixture for advisor types."""
    return request.param


@pytest_asyncio.fixture
async def all_recommendations():
    """Mock fixture for all advisor recommendations."""
    return {
        "value": [
            {
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "allocation": 15,
                "confidence": 0.8,
                "risk": 0.5,
                "justification": "Strong value metrics"
            },
            {
                "ticker": "MSFT", 
                "company": "Microsoft Corp.",
                "allocation": 12,
                "confidence": 0.75,
                "risk": 0.45,
                "justification": "Solid fundamentals"
            }
        ],
        "macro": [
            {
                "ticker": "NVDA",
                "company": "NVIDIA Corp.",
                "allocation": 10,
                "confidence": 0.85,
                "risk": 0.7,
                "justification": "AI growth trend"
            },
            {
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "allocation": 8,
                "confidence": 0.7,
                "risk": 0.5,
                "justification": "Market position"
            }
        ]
    }


@pytest_asyncio.fixture
async def top_picks(all_recommendations):
    """Mock fixture for top stock picks."""
    # Create consensus data structure similar to what test_portfolio_consensus would create
    ticker_data = {}
    
    for advisor, recommendations in all_recommendations.items():
        for rec in recommendations:
            ticker = rec.get('ticker')
            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    'company': rec.get('company', ticker),
                    'advisors': [],
                    'total_weight': 0,
                    'avg_confidence': 0,
                    'avg_risk': 0,
                    'advisor_count': 0
                }
            
            ticker_data[ticker]['advisors'].append({
                'advisor': advisor,
                'allocation': rec.get('allocation', 0),
                'confidence': rec.get('confidence', 0.5),
                'risk': rec.get('risk', 0.5),
                'justification': rec.get('justification', '')
            })
    
    # Calculate consensus metrics
    for ticker, data in ticker_data.items():
        advisors = data['advisors']
        data['total_weight'] = sum(a['allocation'] for a in advisors)
        data['avg_confidence'] = sum(a['confidence'] for a in advisors) / len(advisors)
        data['avg_risk'] = sum(a['risk'] for a in advisors) / len(advisors) 
        data['advisor_count'] = len(advisors)
    
    # Sort by advisor agreement and total weight
    sorted_tickers = sorted(
        ticker_data.items(),
        key=lambda x: (x[1]['advisor_count'], x[1]['total_weight']),
        reverse=True
    )
    
    return sorted_tickers 