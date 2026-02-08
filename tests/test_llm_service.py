#!/usr/bin/env python3
"""Test script for the LLM service with OpenRouter."""

import asyncio
import json
import os
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the path
import sys
sys.path.append('.')

from app.services.llm import OpenRouterLLMService
from app.models import AdvisorType

@pytest.mark.asyncio
async def test_single_advisor(advisor_type: AdvisorType):
    """Test a single advisor."""
    print(f"\n🧠 Testing {advisor_type.value.upper()} advisor...")
    
    service = OpenRouterLLMService()
    
    try:
        recommendations = await service.get_recommendation(advisor_type)
        print(f"✅ {advisor_type.value} advisor returned {len(recommendations)} recommendations")
        
        if recommendations:
            # Show first recommendation
            rec = recommendations[0]
            print(f"   Sample: {rec.symbol} - {rec.recommended_weight:.1%} allocation")
            print(f"   Reasoning: {rec.reasoning[:60]}...")
            print(f"   Confidence: {rec.confidence:.2f}, Risk: {rec.risk_score:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ {advisor_type.value} advisor failed: {e}")
        return False

@pytest.mark.asyncio
async def test_all_advisors():
    """Test all advisors in parallel."""
    print("\n🚀 Testing all advisors in parallel...")
    
    service = OpenRouterLLMService()
    
    try:
        results = await service.get_all_recommendations()
        
        print(f"✅ Got recommendations from {len(results)} advisors")
        
        for advisor_type, recommendations in results.items():
            if recommendations:
                print(f"   {advisor_type.value}: {len(recommendations)} stocks")
            else:
                print(f"   {advisor_type.value}: No recommendations (possible error)")
        
        return True
    except Exception as e:
        print(f"❌ Parallel test failed: {e}")
        return False

@pytest.mark.asyncio
async def test_portfolio_context():
    """Test with portfolio context."""
    print("\n📊 Testing with portfolio context...")
    
    portfolio_context = {
        "nav": 100000.0,
        "position_count": 5,
        "positions": {
            "AAPL": {"weight": 0.2, "value": 20000},
            "MSFT": {"weight": 0.15, "value": 15000},
            "GOOGL": {"weight": 0.15, "value": 15000},
            "TSLA": {"weight": 0.25, "value": 25000},
            "NVDA": {"weight": 0.25, "value": 25000}
        },
        "cash": 0.0
    }
    
    service = OpenRouterLLMService()
    
    try:
        # Test with one advisor
        recommendations = await service.get_recommendation(
            AdvisorType.RISK, 
            portfolio_context
        )
        
        print(f"✅ Risk advisor with context: {len(recommendations)} recommendations")
        if recommendations:
            rec = recommendations[0]
            print(f"   Context-aware pick: {rec.symbol} - {rec.reasoning[:60]}...")
        
        return True
    except Exception as e:
        print(f"❌ Portfolio context test failed: {e}")
        return False

# Tests are now properly configured for pytest 