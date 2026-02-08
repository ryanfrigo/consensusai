#!/usr/bin/env python3
"""Check the status of recent orders placed by the portfolio manager."""

import sys
sys.path.append('.')

from app.config import settings

def check_recent_orders():
    """Check recent orders and their status."""
    print("🔍 Checking Recent Orders")
    print("=" * 50)
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        from datetime import datetime, timedelta
        
        client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True
        )
        
        # Get account info
        account = client.get_account()
        print(f"Account Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        
        # Get recent orders (last 24 hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        # Get all recent orders
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=50
        )
        
        orders = client.get_orders(filter=request)
        
        if not orders:
            print("\nℹ️  No recent orders found")
            return
        
        print(f"\n📋 Found {len(orders)} recent orders:")
        print("-" * 80)
        
        for order in orders:
            filled_qty = getattr(order, 'filled_qty', 0) or 0
            avg_fill_price = getattr(order, 'filled_avg_price', 0) or 0
            
            print(f"Order ID: {order.id}")
            print(f"  Symbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Quantity: {order.qty}")
            print(f"  Status: {order.status}")
            print(f"  Order Type: {order.order_type}")
            print(f"  Time in Force: {order.time_in_force}")
            print(f"  Submitted: {order.submitted_at}")
            print(f"  Filled Qty: {filled_qty}")
            if avg_fill_price:
                print(f"  Avg Fill Price: ${float(avg_fill_price):.2f}")
                total_value = float(filled_qty) * float(avg_fill_price)
                print(f"  Total Value: ${total_value:.2f}")
            
            if hasattr(order, 'canceled_at') and order.canceled_at:
                print(f"  Canceled: {order.canceled_at}")
            if hasattr(order, 'filled_at') and order.filled_at:
                print(f"  Filled: {order.filled_at}")
            
            print("-" * 40)
        
        # Check current positions
        print("\n📊 Current Positions:")
        positions = client.get_all_positions()
        
        if not positions:
            print("ℹ️  No current positions")
        else:
            print(f"Found {len(positions)} positions:")
            for pos in positions:
                current_value = float(pos.qty) * float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                pl_percent = (unrealized_pl / float(pos.cost_basis)) * 100 if float(pos.cost_basis) != 0 else 0
                
                print(f"  {pos.symbol}: {pos.qty} shares")
                print(f"    Current Price: ${float(pos.current_price):.2f}")
                print(f"    Market Value: ${current_value:,.2f}")
                print(f"    Cost Basis: ${float(pos.cost_basis):,.2f}")
                print(f"    Unrealized P/L: ${unrealized_pl:.2f} ({pl_percent:+.1f}%)")
                print()
        
    except Exception as e:
        print(f"❌ Error checking orders: {e}")

if __name__ == "__main__":
    check_recent_orders() 