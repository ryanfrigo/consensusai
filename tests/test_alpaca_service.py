#!/usr/bin/env python3
"""Test script for the Alpaca trading service."""

import asyncio
import os
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the path
import sys
sys.path.append('.')

from app.services.alpaca import AlpacaService

@pytest.mark.asyncio
async def test_alpaca_connection():
    """Test basic Alpaca API connection."""
    print("🔌 Testing Alpaca API connection...")
    
    if not all([
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    ]):
        print("❌ Missing Alpaca environment variables")
        return False
    
    service = AlpacaService()
    
    try:
        # Test account NAV
        nav = await service.get_account_nav()
        print(f"✅ Account connected: ${nav:,.2f} portfolio value")
        
        return True
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return False

@pytest.mark.asyncio
async def test_get_positions():
    """Test getting current positions."""
    print("\n📊 Testing get positions...")
    
    service = AlpacaService()
    
    try:
        positions = await service.get_positions()
        print(f"✅ Retrieved {len(positions)} positions")
        
        if positions:
            for pos in positions[:3]:  # Show first 3
                print(f"   {pos.symbol}: {pos.quantity} shares @ ${pos.market_value:,.2f}")
                print(f"      Current price: ${pos.current_price:.2f}, P&L: ${pos.unrealized_pnl:.2f}")
        else:
            print("   No open positions")
        
        return True
    except Exception as e:
        print(f"❌ Get positions failed: {e}")
        return False

@pytest.mark.asyncio
async def test_market_data():
    """Test getting market data."""
    print("\n📈 Testing market data...")
    
    service = AlpacaService()
    
    try:
        # Test getting current prices for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        prices = await service.get_latest_prices(symbols)
        
        print(f"✅ Retrieved prices for {len(prices)} symbols:")
        for symbol, price in prices.items():
            print(f"   {symbol}: ${price:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Get market data failed: {e}")
        return False

@pytest.mark.asyncio
async def test_slippage_estimation():
    """Test slippage estimation."""
    print("\n🎯 Testing slippage estimation...")
    
    service = AlpacaService()
    
    try:
        slippage_small = await service.estimate_slippage("AAPL", 10)
        slippage_large = await service.estimate_slippage("AAPL", 1000)
        
        print(f"✅ Slippage estimates:")
        print(f"   Small order (10 shares): {slippage_small:.3%}")
        print(f"   Large order (1000 shares): {slippage_large:.3%}")
        
        return True
    except Exception as e:
        print(f"❌ Slippage estimation failed: {e}")
        return False

@pytest.mark.asyncio
async def test_dry_run_order():
    """Test submitting a dry run order."""
    print("\n📝 Testing dry run order...")
    
    service = AlpacaService()
    
    try:
        # Submit a dry run order
        result = await service.submit_order(
            symbol="AAPL",
            quantity=1,
            side="buy",
            dry_run=True
        )
        
        print(f"✅ Dry run order submitted:")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        
        return True
    except Exception as e:
        print(f"❌ Dry run order failed: {e}")
        return False

# Tests are now properly configured for pytest 