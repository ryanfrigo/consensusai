"""Alpaca trading API integration."""

import asyncio
from typing import Dict, List, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, OrderSide
from alpaca.trading.enums import TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import logging

from ..config import settings
from ..models import PositionSchema, OrderLogSchema

logger = logging.getLogger(__name__)


class AlpacaService:
    """Service for interacting with Alpaca trading API."""
    
    def __init__(self):
        """Initialize Alpaca clients."""
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True if "paper" in settings.alpaca_base_url else False
        )
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key
        )    
    async def get_positions(self) -> List[PositionSchema]:
        """Fetch current portfolio positions."""
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(
                None, self.trading_client.get_all_positions
            )
            
            position_schemas = []
            for pos in positions:
                position_schemas.append(PositionSchema(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pnl=float(pos.unrealized_pl or 0),
                    current_price=float(pos.current_price or 0),
                    last_updated=None  # Alpaca Position object doesn't have created_at
                ))
            
            logger.info(f"Retrieved {len(position_schemas)} positions")
            return position_schemas
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    async def get_account_nav(self) -> float:
        """Get total portfolio NAV."""
        try:
            loop = asyncio.get_event_loop()
            account = await loop.run_in_executor(
                None, self.trading_client.get_account
            )
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error fetching account NAV: {e}")
            raise    
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for symbols."""
        try:
            loop = asyncio.get_event_loop()
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = await loop.run_in_executor(
                None, self.data_client.get_stock_latest_quote, request
            )
            
            prices = {}
            for symbol, quote in quotes.items():
                prices[symbol] = float(quote.bid_price + quote.ask_price) / 2
                
            return prices
        except Exception as e:
            logger.error(f"Error fetching prices for {symbols}: {e}")
            raise
    
    async def estimate_slippage(self, symbol: str, quantity: float) -> float:
        """Estimate slippage for an order."""
        try:
            # Simple slippage model - 0.1% for small orders, scale with size
            base_slippage = 0.001
            size_factor = min(abs(quantity) / 1000, 0.01)  # Max 1% additional
            return base_slippage + size_factor
        except Exception as e:
            logger.error(f"Error estimating slippage for {symbol}: {e}")
            return 0.005  # Default 0.5% slippage
    
    async def submit_order(
        self, 
        symbol: str, 
        quantity: float, 
        side: str,
        dry_run: bool = True
    ) -> Dict[str, any]:
        """Submit an order to Alpaca."""
        try:
            if dry_run:
                logger.info(f"DRY RUN: Would submit {side} order for {quantity} shares of {symbol}")
                return {
                    "order_id": f"dry_run_{symbol}_{quantity}",
                    "status": "accepted",
                    "message": "Dry run order"
                }
            
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(quantity),
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None, self.trading_client.submit_order, order_request
            )
            
            logger.info(f"Submitted order {order.id} for {symbol}")
            return {
                "order_id": order.id,
                "status": order.status.value,
                "message": f"Order submitted successfully"
            }
            
        except Exception as e:
            error_msg = f"Error submitting order for {symbol}: {e}"
            logger.error(error_msg)
            return {
                "order_id": None,
                "status": "rejected",
                "message": error_msg
            }