"""Portfolio rebalancing and risk management service."""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from ..config import settings
from ..models import (
    PositionSchema, RecommendationSchema, PortfolioTargetSchema, 
    AdvisorType, OrderLogSchema, OrderSide
)

logger = logging.getLogger(__name__)


class RebalanceService:
    """Service for portfolio rebalancing with risk controls."""
    
    def __init__(self):
        self.settings = settings
    
    def aggregate_recommendations(
        self, 
        recommendations: Dict[AdvisorType, List[RecommendationSchema]]
    ) -> Dict[str, float]:
        """Aggregate recommendations from all advisors into target weights."""
        
        # Weight each advisor's recommendations
        advisor_weights = {
            AdvisorType.VALUE: 0.4,   # 40% weight
            AdvisorType.MACRO: 0.35,  # 35% weight  
            AdvisorType.RISK: 0.25    # 25% weight
        }
        
        symbol_weights = defaultdict(float)
        symbol_confidence = defaultdict(list)
        
        # Aggregate weighted recommendations
        for advisor_type, recs in recommendations.items():
            advisor_weight = advisor_weights[advisor_type]
            
            for rec in recs:
                # Weight by advisor importance and confidence
                effective_weight = (
                    rec.recommended_weight * 
                    advisor_weight * 
                    rec.confidence
                )
                symbol_weights[rec.symbol] += effective_weight
                symbol_confidence[rec.symbol].append(rec.confidence)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(symbol_weights.values())
        if total_weight > 0:
            for symbol in symbol_weights:
                symbol_weights[symbol] /= total_weight        
        # Filter out positions with very low weights (< 0.5%)
        filtered_weights = {
            symbol: weight for symbol, weight in symbol_weights.items()
            if weight >= 0.005
        }
        
        # Enforce maximum number of target symbols based on weight, if configured
        max_positions = getattr(self.settings, 'target_position_count', None)
        capped_weights = filtered_weights
        if max_positions and len(filtered_weights) > max_positions:
            # Keep top-N by weight
            sorted_items = sorted(filtered_weights.items(), key=lambda kv: kv[1], reverse=True)
            capped_weights = dict(sorted_items[:max_positions])
            logger.info(f"Capped targets to top {max_positions} symbols (from {len(filtered_weights)})")

        # Renormalize after filtering and capping
        total_filtered = sum(capped_weights.values())
        if total_filtered > 0:
            for symbol in list(capped_weights.keys()):
                capped_weights[symbol] = capped_weights[symbol] / total_filtered
        
        logger.info(f"Aggregated recommendations for {len(capped_weights)} symbols")
        return dict(capped_weights)
    
    def apply_risk_controls(
        self,
        target_weights: Dict[str, float],
        current_positions: List[PositionSchema],
        portfolio_nav: float
    ) -> Tuple[Dict[str, float], List[str]]:
        """Apply risk controls to target weights."""
        
        risk_violations = []
        adjusted_weights = target_weights.copy()
        
        # Calculate current weights
        current_weights = {}
        for pos in current_positions:
            current_weights[pos.symbol] = pos.market_value / portfolio_nav
        
        # Risk Control 1: Maximum position size
        for symbol, weight in list(adjusted_weights.items()):
            if weight > self.settings.max_position_weight:
                risk_violations.append(
                    f"Capped {symbol} from {weight:.1%} to {self.settings.max_position_weight:.1%}"
                )
                adjusted_weights[symbol] = self.settings.max_position_weight
        
        # Risk Control 2: Maximum weight delta
        for symbol, target_weight in list(adjusted_weights.items()):
            current_weight = current_weights.get(symbol, 0.0)
            weight_delta = abs(target_weight - current_weight)
            
            if weight_delta > self.settings.max_weight_delta:
                # Limit the change to max_weight_delta
                if target_weight > current_weight:
                    new_weight = current_weight + self.settings.max_weight_delta
                else:
                    new_weight = max(0, current_weight - self.settings.max_weight_delta)
                
                risk_violations.append(
                    f"Limited {symbol} weight change from {weight_delta:.1%} to {self.settings.max_weight_delta:.1%}"
                )
                adjusted_weights[symbol] = new_weight        
        # Renormalize after risk adjustments
        total_adjusted = sum(adjusted_weights.values())
        if total_adjusted > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_adjusted
        
        logger.info(f"Applied risk controls, {len(risk_violations)} violations")
        return adjusted_weights, risk_violations
    
    def calculate_turnover(
        self,
        target_weights: Dict[str, float],
        current_positions: List[PositionSchema],
        portfolio_nav: float
    ) -> float:
        """Calculate portfolio turnover for the rebalance."""
        
        current_weights = {}
        for pos in current_positions:
            current_weights[pos.symbol] = pos.market_value / portfolio_nav
        
        total_turnover = 0.0
        
        # Calculate turnover as sum of absolute weight changes
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            turnover = abs(target_weight - current_weight)
            total_turnover += turnover
        
        # Divide by 2 since we double-count (buy and sell)
        return total_turnover / 2
    
    def generate_orders(
        self,
        target_weights: Dict[str, float],
        current_positions: List[PositionSchema],
        current_prices: Dict[str, float],
        portfolio_nav: float
    ) -> List[Dict[str, any]]:
        """Generate orders to achieve target weights."""
        
        orders = []
        current_weights = {}
        
        # Build current weight map
        for pos in current_positions:
            current_weights[pos.symbol] = pos.market_value / portfolio_nav        
        # Generate orders for all symbols
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            
            if symbol not in current_prices:
                logger.warning(f"No price data for {symbol}, skipping")
                continue
            
            # Calculate target value and shares
            target_value = target_weight * portfolio_nav
            current_value = current_weight * portfolio_nav
            value_delta = target_value - current_value
            
            # Skip small trades, but always liquidate positions with zero target weight
            if target_weight == 0.0:
                pass  # allow liquidation regardless of thresholds
            else:
                if abs(value_delta) < self.settings.min_trade_value:
                    continue
                
                if abs(value_delta) / portfolio_nav < self.settings.min_trade_percent:
                    continue
            
            # Calculate shares to trade
            price = current_prices[symbol]
            shares_delta = value_delta / price
            
            # Determine order side and quantity
            if shares_delta > 0:
                side = OrderSide.BUY
                quantity = abs(shares_delta)
            else:
                side = OrderSide.SELL
                quantity = abs(shares_delta)
            
            orders.append({
                "symbol": symbol,
                "side": side,
                "quantity": round(quantity, 6),
                "estimated_value": abs(value_delta),
                "target_weight": target_weight,
                "current_weight": current_weight,
                "price": price
            })
        
        logger.info(f"Generated {len(orders)} orders")
        return orders