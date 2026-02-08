"""Portfolio management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from typing import List, Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models import (
    PositionSchema, RecommendationSchema, PortfolioTargetSchema, 
    OrderLogSchema, PortfolioSummary, RebalanceRequest, PerformanceMetrics,
    Position, Recommendation, PortfolioTarget, OrderLog, DecisionRecord
)
from ..tasks import PortfolioOrchestrator
from ..services.alpaca import AlpacaService

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/current", response_model=PortfolioSummary)
async def get_current_portfolio(db: AsyncSession = Depends(get_db)) -> PortfolioSummary:
    """Get current portfolio summary."""
    
    # Get positions from database
    result = await db.execute(select(Position))
    positions = result.scalars().all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    # Convert to schemas
    position_schemas = [
        PositionSchema(
            symbol=pos.symbol,
            quantity=pos.quantity,
            market_value=pos.market_value,
            cost_basis=pos.cost_basis,
            unrealized_pnl=pos.unrealized_pnl,
            current_price=pos.current_price,
            last_updated=pos.last_updated
        ) for pos in positions
    ]    
    # Calculate summary metrics
    total_nav = sum(pos.market_value for pos in position_schemas)
    weights = [pos.market_value / total_nav for pos in position_schemas]
    largest_weight = max(weights) if weights else 0.0
    
    # Get cash balance from Alpaca
    alpaca_service = AlpacaService()
    try:
        account_nav = await alpaca_service.get_account_nav()
        cash_balance = account_nav - total_nav
    except Exception:
        cash_balance = 0.0
    
    return PortfolioSummary(
        total_nav=total_nav,
        positions=position_schemas,
        position_count=len(position_schemas),
        largest_position_weight=largest_weight,
        cash_balance=cash_balance,
        last_updated=max(pos.last_updated for pos in position_schemas)
    )


@router.post("/rebalance")
async def trigger_rebalance(
    request: RebalanceRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Trigger a portfolio rebalance."""
    
    orchestrator = PortfolioOrchestrator()
    
    # Run rebalance in background
    background_tasks.add_task(
        orchestrator.run_daily_rebalance,
        force=request.force,
        dry_run=request.dry_run,
        max_turnover_override=request.max_turnover_override,
        fresh_start=getattr(request, 'fresh_start', None)
    )
    
    return {
        "status": "started",
        "message": "Rebalance initiated",
        "dry_run": request.dry_run,
        "fresh_start": getattr(request, 'fresh_start', False),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/history")
async def get_portfolio_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
) -> List[dict]:
    """Get portfolio decision history."""
    
    result = await db.execute(
        select(DecisionRecord)
        .order_by(desc(DecisionRecord.timestamp))
        .limit(limit)
    )
    
    records = result.scalars().all()
    
    return [
        {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "portfolio_nav": record.portfolio_nav,
            "total_turnover": record.total_turnover,
            "orders_generated": record.orders_generated,
            "orders_executed": record.orders_executed,
            "risk_violations": record.risk_violations,
            "dry_run": record.dry_run
        } for record in records
    ]


@router.get("/targets", response_model=List[PortfolioTargetSchema])
async def get_portfolio_targets(db: AsyncSession = Depends(get_db)) -> List[PortfolioTargetSchema]:
    """Get current portfolio targets."""
    
    result = await db.execute(
        select(PortfolioTarget)
        .where(PortfolioTarget.is_active == True)
        .order_by(desc(PortfolioTarget.timestamp))
    )
    
    targets = result.scalars().all()
    
    return [
        PortfolioTargetSchema(
            id=target.id,
            timestamp=target.timestamp,
            symbol=target.symbol,
            target_weight=target.target_weight,
            current_weight=target.current_weight,
            weight_delta=target.weight_delta,
            is_active=target.is_active
        ) for target in targets
    ]


@router.get("/recommendations", response_model=List[RecommendationSchema])
async def get_recommendations(
    advisor_type: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
) -> List[RecommendationSchema]:
    """Get LLM advisor recommendations."""
    
    query = select(Recommendation).order_by(desc(Recommendation.timestamp))
    
    if advisor_type:
        query = query.where(Recommendation.advisor_type == advisor_type)
    
    query = query.limit(limit)
    result = await db.execute(query)
    recommendations = result.scalars().all()
    
    return [
        RecommendationSchema(
            id=rec.id,
            timestamp=rec.timestamp,
            advisor_type=rec.advisor_type,
            symbol=rec.symbol,
            recommended_weight=rec.recommended_weight,
            confidence=rec.confidence,
            reasoning=rec.reasoning,
            context_used=rec.context_used
        ) for rec in recommendations
    ]


@router.get("/orders")
async def get_order_history(
    symbol: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[dict]:
    """Get order execution history."""
    
    query = select(OrderLog).order_by(desc(OrderLog.timestamp))
    
    if symbol:
        query = query.where(OrderLog.symbol == symbol)
    
    query = query.limit(limit)
    result = await db.execute(query)
    orders = result.scalars().all()
    
    return [
        {
            "id": order.id,
            "timestamp": order.timestamp.isoformat(),
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "alpaca_order_id": order.alpaca_order_id,
            "status": order.status,
            "filled_price": order.filled_price,
            "filled_quantity": order.filled_quantity,
            "error_message": order.error_message
        } for order in orders
    ]


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(db: AsyncSession = Depends(get_db)) -> PerformanceMetrics:
    """Get portfolio performance metrics."""
    
    # Get recent decision records for performance calculation
    result = await db.execute(
        select(DecisionRecord)
        .order_by(desc(DecisionRecord.timestamp))
        .limit(30)  # Last 30 rebalances
    )
    
    records = result.scalars().all()
    
    if not records:
        raise HTTPException(status_code=404, detail="No performance data available")
    
    # Simple performance metrics calculation
    nav_values = [record.portfolio_nav for record in reversed(records)]
    
    if len(nav_values) < 2:
        return PerformanceMetrics(
            total_return=0.0,
            daily_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_trade_return=0.0
        )
    
    # Calculate basic metrics
    total_return = (nav_values[-1] - nav_values[0]) / nav_values[0]
    daily_return = (nav_values[-1] - nav_values[-2]) / nav_values[-2] if len(nav_values) > 1 else 0.0
    
    # Calculate volatility (simplified)
    daily_returns = []
    for i in range(1, len(nav_values)):
        daily_ret = (nav_values[i] - nav_values[i-1]) / nav_values[i-1]
        daily_returns.append(daily_ret)
    
    if daily_returns:
        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = variance ** 0.5
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        
        # Max drawdown calculation
        peak = nav_values[0]
        max_drawdown = 0.0
        for value in nav_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        positive_days = sum(1 for r in daily_returns if r > 0)
        win_rate = positive_days / len(daily_returns)
        
        avg_trade_return = avg_return
    else:
        volatility = sharpe_ratio = max_drawdown = win_rate = avg_trade_return = 0.0
    
    return PerformanceMetrics(
        total_return=total_return,
        daily_return=daily_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_trade_return=avg_trade_return
    )


@router.post("/sync-positions")
async def sync_positions(db: AsyncSession = Depends(get_db)) -> dict:
    """Manually sync positions from Alpaca."""
    
    try:
        alpaca_service = AlpacaService()
        positions = await alpaca_service.get_positions()
        
        # Clear existing positions
        await db.execute("DELETE FROM positions")
        
        # Insert new positions  
        for pos_schema in positions:
            position = Position(
                symbol=pos_schema.symbol,
                quantity=pos_schema.quantity,
                market_value=pos_schema.market_value,
                cost_basis=pos_schema.cost_basis,
                unrealized_pnl=pos_schema.unrealized_pnl,
                current_price=pos_schema.current_price,
                last_updated=pos_schema.last_updated
            )
            db.add(position)
        
        await db.commit()
        
        return {
            "status": "success",
            "positions_synced": len(positions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing positions: {str(e)}")