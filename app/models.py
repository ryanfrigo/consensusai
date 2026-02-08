"""Database models and Pydantic schemas."""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

from .database import Base


# SQLAlchemy Models
class Position(Base):
    """Current portfolio positions."""
    __tablename__ = "positions"
    
    symbol = Column(String(10), primary_key=True)
    quantity = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())


class Recommendation(Base):
    """LLM advisor recommendations."""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)    
    timestamp = Column(DateTime, default=func.now())
    advisor_type = Column(String(50), nullable=False)  # value, macro, risk
    symbol = Column(String(10), nullable=False)
    recommended_weight = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    context_used = Column(JSON)


class PortfolioTarget(Base):
    """Canonical target weights after aggregation."""
    __tablename__ = "portfolio_targets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    symbol = Column(String(10), nullable=False)
    target_weight = Column(Float, nullable=False)
    current_weight = Column(Float, nullable=False)
    weight_delta = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)


class OrderLog(Base):
    """All orders sent to Alpaca."""
    __tablename__ = "order_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    symbol = Column(String(10), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    order_type = Column(String(20), nullable=False)
    alpaca_order_id = Column(String(50))
    status = Column(String(20), nullable=False)
    filled_price = Column(Float)
    filled_quantity = Column(Float)
    error_message = Column(Text)


class DecisionRecord(Base):
    """Complete decision record for each run."""
    __tablename__ = "decision_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    portfolio_nav = Column(Float, nullable=False)
    total_turnover = Column(Float, nullable=False)
    orders_generated = Column(Integer, nullable=False)
    orders_executed = Column(Integer, nullable=False)
    risk_violations = Column(JSON)
    execution_summary = Column(JSON)
    dry_run = Column(Boolean, default=False)


class SellTrigger(Base):
    """Active sell triggers to monitor."""
    __tablename__ = "sell_triggers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    trigger_type = Column(String(20), nullable=False)  # stop_loss, take_profit
    trigger_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime)


# Pydantic Schemas
class AdvisorType(str, Enum):
    VALUE = "value"
    MACRO = "macro" 
    RISK = "risk"
    WILDCARD = "wildcard"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSchema(BaseModel):
    """Pydantic schema for positions."""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    current_price: float
    last_updated: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class RecommendationSchema(BaseModel):
    """Pydantic schema for recommendations."""
    id: Optional[int] = None
    timestamp: datetime
    advisor_type: AdvisorType
    symbol: str
    recommended_weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    sell_trigger: Optional[str] = None
    context_used: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class PortfolioTargetSchema(BaseModel):
    """Pydantic schema for portfolio targets."""
    id: Optional[int] = None
    timestamp: datetime
    symbol: str
    target_weight: float = Field(ge=0.0, le=1.0)
    current_weight: float = Field(ge=0.0, le=1.0)
    weight_delta: float
    is_active: bool = True
    
    class Config:
        from_attributes = True


class OrderLogSchema(BaseModel):
    """Pydantic schema for order logs."""
    id: Optional[int] = None
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str
    alpaca_order_id: Optional[str] = None
    status: str
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    """Summary of current portfolio state."""
    total_nav: float
    positions: List[PositionSchema]
    position_count: int
    largest_position_weight: float
    cash_balance: float
    last_updated: datetime


class RebalanceRequest(BaseModel):
    """Request schema for manual rebalancing."""
    force: bool = False
    dry_run: bool = True
    max_turnover_override: Optional[float] = None
    fresh_start: bool = False


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    total_return: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_return: float