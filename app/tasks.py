"""Portfolio orchestration and task management."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from .database import AsyncSessionLocal
from .services.alpaca import AlpacaService
from .services.llm import LLMService
from .services.rebalance import RebalanceService
from .services.triggers import TriggerService
from .models import (
    Position, Recommendation, PortfolioTarget, OrderLog, DecisionRecord,
    PositionSchema, RecommendationSchema, PortfolioTargetSchema, AdvisorType
)
from .config import settings

logger = logging.getLogger(__name__)


class PortfolioOrchestrator:
    """Main orchestrator for portfolio management workflow."""
    
    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.llm_service = LLMService()
        self.rebalance_service = RebalanceService()
        self.trigger_service = TriggerService()
        self.settings = settings
    
    async def run_daily_rebalance(
        self,
        force: bool = False,
        dry_run: Optional[bool] = None,
        max_turnover_override: Optional[float] = None,
        fresh_start: Optional[bool] = None
    ) -> Dict[str, any]:
        """Execute the complete daily rebalancing workflow."""
        
        if dry_run is None:
            dry_run = self.settings.dry_run
        
        start_time = datetime.now()
        rebalance_mode = "fresh start" if fresh_start else "portfolio-aware"
        logger.info(f"Starting daily rebalance ({rebalance_mode} mode, dry_run={dry_run})")
        
        async with AsyncSessionLocal() as db:
            try:
                # Step 1: Fetch current portfolio state
                positions = await self.alpaca_service.get_positions()
                portfolio_nav = await self.alpaca_service.get_account_nav()
                
                logger.info(f"Portfolio NAV: ${portfolio_nav:,.2f}, Positions: {len(positions)}")
                
                # Step 2: Update positions in database
                await self._update_positions_in_db(db, positions)                
                # Step 3: Get previous targets for comparison
                previous_targets = await self._get_previous_targets(db)
                
                # Step 4: Build portfolio context for LLMs (unless fresh start)
                portfolio_context = None
                if not fresh_start:
                    portfolio_context = await self._build_portfolio_context(
                        positions, portfolio_nav, previous_targets
                    )
                
                # Step 5: Get recommendations from all advisors
                recommendations = await self.llm_service.get_all_recommendations(
                    portfolio_context, fresh_start
                )
                
                # Step 6: Save recommendations to database
                await self._save_recommendations(db, recommendations)
                
                # Step 7: Aggregate recommendations
                target_weights = self.rebalance_service.aggregate_recommendations(recommendations)
                
                # Step 8: Apply risk controls
                adjusted_weights, risk_violations = self.rebalance_service.apply_risk_controls(
                    target_weights, positions, portfolio_nav
                )
                
                # Step 9: Check turnover limits
                turnover = self.rebalance_service.calculate_turnover(
                    adjusted_weights, positions, portfolio_nav
                )
                
                max_turnover = max_turnover_override or self.settings.max_daily_turnover
                
                if turnover > max_turnover and not force:
                    logger.warning(f"Turnover {turnover:.1%} exceeds limit {max_turnover:.1%}")
                    return {
                        "status": "blocked",
                        "reason": f"Turnover limit exceeded ({turnover:.1%} > {max_turnover:.1%})",
                        "turnover": turnover,
                        "max_turnover": max_turnover
                    }
                
                # Step 10: Generate orders
                symbols = list(set(list(adjusted_weights.keys()) + [p.symbol for p in positions]))
                current_prices = await self.alpaca_service.get_latest_prices(symbols)
                
                orders = self.rebalance_service.generate_orders(
                    adjusted_weights, positions, current_prices, portfolio_nav
                )
                
                # Step 11: Execute orders
                execution_results = await self._execute_orders(orders, dry_run)
                
                # Step 12: Save targets and decision record
                await self._save_targets(db, adjusted_weights, positions, portfolio_nav)
                decision_record = await self._save_decision_record(
                    db, portfolio_nav, turnover, orders, execution_results, 
                    risk_violations, dry_run
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Daily rebalance completed in {duration:.1f}s")
                
                return {
                    "status": "success",
                    "duration_seconds": duration,
                    "portfolio_nav": portfolio_nav,
                    "turnover": turnover,
                    "orders_generated": len(orders),
                    "orders_executed": len([r for r in execution_results if r["status"] == "accepted"]),
                    "risk_violations": risk_violations,
                    "dry_run": dry_run,
                    "decision_record_id": decision_record.id
                }
                
            except Exception as e:
                logger.error(f"Error in daily rebalance: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "duration_seconds": (datetime.now() - start_time).total_seconds()
                }    
    async def _update_positions_in_db(
        self, 
        db: AsyncSession, 
        positions: List[PositionSchema]
    ) -> None:
        """Update current positions in database."""
        
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
        logger.info(f"Updated {len(positions)} positions in database")
    
    async def _get_previous_targets(self, db: AsyncSession) -> Dict[str, float]:
        """Get the most recent portfolio targets."""
        
        result = await db.execute(
            select(PortfolioTarget)
            .where(PortfolioTarget.is_active == True)
            .order_by(desc(PortfolioTarget.timestamp))
        )
        
        targets = result.scalars().all()
        return {target.symbol: target.target_weight for target in targets}
    
    async def _build_portfolio_context(
        self,
        positions: List[PositionSchema],
        portfolio_nav: float,
        previous_targets: Dict[str, float]
    ) -> Dict[str, any]:
        """Build context for LLM advisors."""
        
        # Calculate basic metrics
        position_dict = {pos.symbol: {
            "weight": pos.market_value / portfolio_nav,
            "value": pos.market_value,
            "pnl": pos.unrealized_pnl,
            "price": pos.current_price
        } for pos in positions}
        
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        cash_balance = portfolio_nav - sum(pos.market_value for pos in positions)        
        return {
            "nav": portfolio_nav,
            "position_count": len(positions),
            "positions": position_dict,
            "cash": cash_balance,
            "total_pnl": total_pnl,
            "previous_targets": previous_targets,
            "recent_performance": f"Unrealized P&L: ${total_pnl:,.2f}",
            "risk_metrics": "Within normal parameters",
            "market_conditions": "Current market environment"
        }
    
    async def _save_recommendations(
        self,
        db: AsyncSession,
        recommendations: Dict[AdvisorType, List[RecommendationSchema]]
    ) -> None:
        """Save LLM recommendations to database."""
        
        for advisor_type, recs in recommendations.items():
            for rec_schema in recs:
                recommendation = Recommendation(
                    timestamp=rec_schema.timestamp,
                    advisor_type=rec_schema.advisor_type.value,
                    symbol=rec_schema.symbol,
                    recommended_weight=rec_schema.recommended_weight,
                    confidence=rec_schema.confidence,
                    reasoning=rec_schema.reasoning,
                    context_used=rec_schema.context_used
                )
                db.add(recommendation)
        
        await db.commit()
        total_recs = sum(len(recs) for recs in recommendations.values())
        logger.info(f"Saved {total_recs} recommendations to database")
    
    async def _execute_orders(
        self,
        orders: List[Dict[str, any]],
        dry_run: bool
    ) -> List[Dict[str, any]]:
        """Execute orders through Alpaca."""
        
        results = []
        for order in orders:
            try:
                result = await self.alpaca_service.submit_order(
                    symbol=order["symbol"],
                    quantity=order["quantity"],
                    side=order["side"].value,
                    dry_run=dry_run
                )
                results.append({
                    **order,
                    "alpaca_order_id": result["order_id"],
                    "status": result["status"],
                    "message": result["message"]
                })
            except Exception as e:
                results.append({
                    **order,
                    "alpaca_order_id": None,
                    "status": "error",
                    "message": str(e)
                })
        
        return results    
    async def _save_targets(
        self,
        db: AsyncSession,
        target_weights: Dict[str, float],
        positions: List[PositionSchema],
        portfolio_nav: float
    ) -> None:
        """Save portfolio targets to database."""
        
        # Deactivate previous targets
        await db.execute(
            "UPDATE portfolio_targets SET is_active = false WHERE is_active = true"
        )
        
        # Calculate current weights
        current_weights = {}
        for pos in positions:
            current_weights[pos.symbol] = pos.market_value / portfolio_nav
        
        # Save new targets
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_delta = target_weight - current_weight
            
            target = PortfolioTarget(
                symbol=symbol,
                target_weight=target_weight,
                current_weight=current_weight,
                weight_delta=weight_delta,
                is_active=True
            )
            db.add(target)
        
        await db.commit()
        logger.info(f"Saved {len(target_weights)} portfolio targets")
    
    async def _save_decision_record(
        self,
        db: AsyncSession,
        portfolio_nav: float,
        turnover: float,
        orders: List[Dict[str, any]],
        execution_results: List[Dict[str, any]],
        risk_violations: List[str],
        dry_run: bool
    ) -> DecisionRecord:
        """Save complete decision record."""
        
        orders_executed = len([r for r in execution_results if r["status"] == "accepted"])
        
        execution_summary = {
            "orders": execution_results,
            "total_orders": len(orders),
            "successful_orders": orders_executed,
            "failed_orders": len(orders) - orders_executed
        }
        
        decision_record = DecisionRecord(
            portfolio_nav=portfolio_nav,
            total_turnover=turnover,
            orders_generated=len(orders),
            orders_executed=orders_executed,
            risk_violations=risk_violations,
            execution_summary=execution_summary,
            dry_run=dry_run
        )
        
        db.add(decision_record)
        await db.commit()
        await db.refresh(decision_record)
        
        logger.info(f"Saved decision record {decision_record.id}")
        return decision_record