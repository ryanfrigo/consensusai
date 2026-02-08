"""Sell trigger monitoring and management service."""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models import SellTrigger, PositionSchema
from ..config import settings

logger = logging.getLogger(__name__)


class TriggerService:
    """Service for managing and evaluating sell triggers."""
    
    def __init__(self):
        self.settings = settings
    
    async def create_stop_loss_trigger(
        self,
        db: AsyncSession,
        symbol: str,
        trigger_price: float
    ) -> SellTrigger:
        """Create a stop-loss trigger for a position."""
        
        trigger = SellTrigger(
            symbol=symbol,
            trigger_type="stop_loss",
            trigger_price=trigger_price,
            is_active=True
        )
        
        db.add(trigger)
        await db.commit()
        await db.refresh(trigger)
        
        logger.info(f"Created stop-loss trigger for {symbol} at ${trigger_price}")
        return trigger
    
    async def create_take_profit_trigger(
        self,
        db: AsyncSession,
        symbol: str,
        trigger_price: float
    ) -> SellTrigger:
        """Create a take-profit trigger for a position."""
        
        trigger = SellTrigger(
            symbol=symbol,
            trigger_type="take_profit",
            trigger_price=trigger_price,
            is_active=True
        )
        
        db.add(trigger)
        await db.commit()
        await db.refresh(trigger)
        
        logger.info(f"Created take-profit trigger for {symbol} at ${trigger_price}")
        return trigger
    
    async def evaluate_triggers(
        self,
        db: AsyncSession,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Evaluate all active triggers against current prices."""
        
        # Get all active triggers
        result = await db.execute(
            select(SellTrigger).where(SellTrigger.is_active == True)
        )
        triggers = result.scalars().all()
        
        triggered_orders = []
        
        for trigger in triggers:
            if trigger.symbol not in current_prices:
                continue
                
            current_price = current_prices[trigger.symbol]
            should_trigger = False
            
            if trigger.trigger_type == "stop_loss":
                # Trigger if price falls below stop price
                should_trigger = current_price <= trigger.trigger_price
            elif trigger.trigger_type == "take_profit":
                # Trigger if price rises above take profit price
                should_trigger = current_price >= trigger.trigger_price
            
            if should_trigger:
                # Mark trigger as triggered
                trigger.is_active = False
                trigger.triggered_at = datetime.now()
                
                triggered_orders.append({
                    "symbol": trigger.symbol,
                    "trigger_type": trigger.trigger_type,
                    "trigger_price": trigger.trigger_price,
                    "current_price": current_price,
                    "action": "sell",
                    "reason": f"{trigger.trigger_type} triggered at ${current_price}"
                })
                
                logger.info(f"Trigger activated for {trigger.symbol}: {trigger.trigger_type} at ${current_price}")
        
        await db.commit()
        return triggered_orders    
    async def cleanup_old_triggers(
        self,
        db: AsyncSession,
        days_old: int = 30
    ) -> int:
        """Clean up old inactive triggers."""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        result = await db.execute(
            select(SellTrigger).where(
                and_(
                    SellTrigger.is_active == False,
                    SellTrigger.triggered_at < cutoff_date
                )
            )
        )
        
        old_triggers = result.scalars().all()
        count = len(old_triggers)
        
        for trigger in old_triggers:
            await db.delete(trigger)
        
        await db.commit()
        logger.info(f"Cleaned up {count} old triggers")
        return count
    
    async def get_active_triggers(
        self,
        db: AsyncSession,
        symbol: Optional[str] = None
    ) -> List[SellTrigger]:
        """Get all active triggers, optionally filtered by symbol."""
        
        query = select(SellTrigger).where(SellTrigger.is_active == True)
        
        if symbol:
            query = query.where(SellTrigger.symbol == symbol)
        
        result = await db.execute(query)
        return result.scalars().all()