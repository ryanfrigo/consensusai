#!/usr/bin/env python3
"""
Advanced Consensus Engine for Multi-Agent Portfolio Management
Implements sophisticated rank-based weighting and cross-agent fusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
try:
    # Prefer centralized app settings when available
    from app.config import settings as app_settings
except Exception:  # pragma: no cover - fallback if app package unavailable
    app_settings = None

@dataclass
class AgentPick:
    """Represents a single stock pick from an agent."""
    ticker: str
    rank: int  # 1-10, where 1 is strongest
    allocation: float  # Original allocation percentage
    confidence: float  # Agent's confidence score
    justification: str = ""
    sell_trigger: str = ""

@dataclass
class AgentRecommendations:
    """All recommendations from a single agent."""
    agent_name: str
    picks: List[AgentPick]
    performance_weight: float = 1.0  # W_i in the formula
    success_rate: float = 1.0

class AdvancedConsensusEngine:
    """Advanced consensus engine using mathematical rank fusion."""
    
    def __init__(self,
                 lambda_decay: float = 0.25,
                 max_single_weight: Optional[float] = None,  # Use config if None
                 min_weight_threshold: float = 0.005,  # 0.5% minimum
                 winsorize_percentiles: Tuple[float, float] = (5, 95),
                 turnover_brake: float = 0.20,  # 20% max position change
                 max_positions: Optional[int] = None):
        
        self.lambda_decay = lambda_decay
        # Use config setting if not explicitly provided
        if max_single_weight is None and app_settings:
            self.max_single_weight = getattr(app_settings, 'max_position_weight', 0.08)
        else:
            self.max_single_weight = max_single_weight or 0.08
        self.min_weight_threshold = min_weight_threshold
        self.winsorize_percentiles = winsorize_percentiles
        self.turnover_brake = turnover_brake
        # Default to configured target count when not explicitly provided
        if max_positions is not None:
            self.max_positions = max_positions
        else:
            self.max_positions = (
                getattr(app_settings, "target_position_count", None) if app_settings else None
            )
        
    def _allocation_to_rank(self, allocations: List[float]) -> List[int]:
        """Convert allocation percentages to ranks (1=highest allocation)."""
        # Sort by allocation descending and assign ranks
        sorted_indices = np.argsort(allocations)[::-1]
        ranks = [0] * len(allocations)
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        return ranks
    
    def _rank_to_weight(self, rank: int) -> float:
        """Convert rank to weight using exponential decay: exp(-λ(r-1))."""
        return np.exp(-self.lambda_decay * (rank - 1))
    
    def _process_agent_recommendations(self, agent_recs: AgentRecommendations) -> Dict[str, float]:
        """Process a single agent's recommendations into normalized weights."""
        logger.info(f"Processing {agent_recs.agent_name} with {len(agent_recs.picks)} picks")
        
        agent_weights = {}
        raw_weights = []
        tickers = []
        
        for pick in agent_recs.picks:
            # Calculate raw weight from rank
            w_raw = self._rank_to_weight(pick.rank)
            
            # Multiply by confidence
            w_agent = w_raw * pick.confidence
            
            raw_weights.append(w_agent)
            tickers.append(pick.ticker)
            
        # Normalize so agent's picks sum to 1
        total_weight = sum(raw_weights)
        if total_weight > 0:
            for i, ticker in enumerate(tickers):
                agent_weights[ticker] = raw_weights[i] / total_weight
        
        logger.info(f"{agent_recs.agent_name} normalized weights: {len(agent_weights)} stocks")
        return agent_weights
    
    def _apply_cross_agent_weights(self, 
                                   all_agent_weights: Dict[str, Dict[str, float]], 
                                   agent_performance_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply cross-agent performance weights and fuse ballots."""
        logger.info("Fusing ballots across agents...")
        
        # Collect all unique tickers
        all_tickers = set()
        for agent_weights in all_agent_weights.values():
            all_tickers.update(agent_weights.keys())
        
        # Calculate fused scores: S_t = Σ W_i × w_i,t
        fused_scores = {}
        for ticker in all_tickers:
            score = 0.0
            contributing_agents = []
            
            for agent_name, agent_weights in all_agent_weights.items():
                if ticker in agent_weights:
                    agent_contribution = agent_performance_weights[agent_name] * agent_weights[ticker]
                    score += agent_contribution
                    contributing_agents.append(agent_name)
            
            fused_scores[ticker] = score
            logger.debug(f"{ticker}: score={score:.4f} from agents: {contributing_agents}")
        
        logger.info(f"Fused scores for {len(fused_scores)} unique tickers")
        return fused_scores
    
    def _trim_craziness(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Winsorize scores and drop low-conviction picks."""
        if not scores:
            return scores
            
        score_values = list(scores.values())
        
        # Winsorize at specified percentiles
        p_low, p_high = self.winsorize_percentiles
        low_threshold = np.percentile(score_values, p_low)
        high_threshold = np.percentile(score_values, p_high)
        
        winsorized_scores = {}
        for ticker, score in scores.items():
            winsorized_score = np.clip(score, low_threshold, high_threshold)
            winsorized_scores[ticker] = winsorized_score
        
        # Drop low-conviction stragglers
        mean_score = np.mean(list(winsorized_scores.values()))
        threshold = 0.5 * mean_score
        
        trimmed_scores = {
            ticker: score for ticker, score in winsorized_scores.items()
            if score >= threshold
        }
        
        dropped_count = len(scores) - len(trimmed_scores)
        logger.info(f"Trimmed {dropped_count} low-conviction picks (threshold: {threshold:.4f})")
        
        return trimmed_scores
    
    def _convert_to_target_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to target portfolio weights."""
        if not scores:
            return {}
            
        total_score = sum(scores.values())
        
        raw_weights = {}
        for ticker, score in scores.items():
            raw_weights[ticker] = score / total_score
        
        return raw_weights
    
    def _apply_guardrails(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply single-name caps and minimum weight filters."""
        # Single-name cap
        capped_weights = {}
        excess_weight = 0.0
        
        for ticker, weight in weights.items():
            if weight > self.max_single_weight:
                capped_weights[ticker] = self.max_single_weight
                excess_weight += weight - self.max_single_weight
                logger.info(f"Capped {ticker} at {self.max_single_weight:.1%} (was {weight:.1%})")
            else:
                capped_weights[ticker] = weight
        
        # Redistribute excess weight proportionally to non-capped positions
        if excess_weight > 0:
            non_capped_tickers = [t for t, w in capped_weights.items() if w < self.max_single_weight]
            if non_capped_tickers:
                non_capped_total = sum(capped_weights[t] for t in non_capped_tickers)
                if non_capped_total > 0:
                    for ticker in non_capped_tickers:
                        proportion = capped_weights[ticker] / non_capped_total
                        capped_weights[ticker] += excess_weight * proportion
        
        # Apply minimum weight threshold
        final_weights = {}
        cash_sweep = 0.0
        
        for ticker, weight in capped_weights.items():
            if weight >= self.min_weight_threshold:
                final_weights[ticker] = weight
            else:
                cash_sweep += weight
                logger.debug(f"Swept {ticker} ({weight:.3%}) to cash")
        
        # Renormalize
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            for ticker in final_weights:
                final_weights[ticker] /= total_weight
        
        if cash_sweep > 0:
            logger.info(f"Cash sweep: {cash_sweep:.1%} from {len(capped_weights) - len(final_weights)} positions")
        
        return final_weights
    
    def build_consensus_portfolio(self, 
                                  all_agent_recommendations: List[AgentRecommendations],
                                  target_portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """Build consensus portfolio from all agent recommendations."""
        
        logger.info(f"Building consensus from {len(all_agent_recommendations)} agents")
        logger.info("=" * 60)
        
        # Step 1: Process each agent's recommendations
        all_agent_weights = {}
        agent_performance_weights = {}
        
        for agent_recs in all_agent_recommendations:
            agent_weights = self._process_agent_recommendations(agent_recs)
            all_agent_weights[agent_recs.agent_name] = agent_weights
            agent_performance_weights[agent_recs.agent_name] = agent_recs.performance_weight
        
        # Step 2: Fuse ballots across agents
        fused_scores = self._apply_cross_agent_weights(all_agent_weights, agent_performance_weights)
        
        # Step 3: Trim craziness
        trimmed_scores = self._trim_craziness(fused_scores)
        
        # Step 4: Convert to target weights
        raw_weights = self._convert_to_target_weights(trimmed_scores)
        
        # Step 5: Apply guardrails
        final_weights = self._apply_guardrails(raw_weights)

        # Step 5b: Enforce maximum number of positions (keep top-N by weight)
        if self.max_positions and len(final_weights) > self.max_positions:
            sorted_by_weight = sorted(final_weights.items(), key=lambda kv: kv[1], reverse=True)
            trimmed = dict(sorted_by_weight[: self.max_positions])
            dropped_count = len(final_weights) - len(trimmed)
            # Renormalize
            total = sum(trimmed.values())
            if total > 0:
                for t in trimmed:
                    trimmed[t] /= total
            logger.info(
                f"Limited positions to top {self.max_positions}; dropped {dropped_count} smaller positions"
            )
            final_weights = trimmed
        
        # Step 6: Calculate dollar amounts
        portfolio_allocations = {}
        for ticker, weight in final_weights.items():
            dollar_amount = target_portfolio_value * weight
            portfolio_allocations[ticker] = {
                'weight': weight,
                'dollar_amount': dollar_amount,
                'consensus_score': trimmed_scores.get(ticker, 0),
                'contributing_agents': [
                    agent for agent, weights in all_agent_weights.items() 
                    if ticker in weights
                ]
            }
        
        # Summary statistics
        total_positions = len(final_weights)
        max_position = max(final_weights.values()) if final_weights else 0
        total_coverage = sum(final_weights.values())
        
        consensus_summary = {
            'portfolio_allocations': portfolio_allocations,
            'total_positions': total_positions,
            'max_position_weight': max_position,
            'total_coverage': total_coverage,
            'participating_agents': len(all_agent_recommendations),
            'successful_agents': len([a for a in all_agent_recommendations if len(a.picks) > 0]),
            'total_unique_picks': len(fused_scores),
            'final_portfolio_picks': len(final_weights),
            'algorithm_params': {
                'lambda_decay': self.lambda_decay,
                'max_single_weight': self.max_single_weight,
                'min_weight_threshold': self.min_weight_threshold,
                'max_positions': self.max_positions or 0
            }
        }
        
        logger.info("=" * 60)
        logger.info("CONSENSUS PORTFOLIO BUILT")
        logger.info(f"Total positions: {total_positions}")
        logger.info(f"Max position: {max_position:.1%}")
        logger.info(f"Coverage: {total_coverage:.1%}")
        logger.info("=" * 60)
        
        return consensus_summary

def convert_raw_recommendations_to_agent_recs(raw_recommendations: Dict[str, List[Dict]]) -> List[AgentRecommendations]:
    """Convert raw LLM recommendations to AgentRecommendations objects."""
    agent_recs_list = []
    
    # Performance weights for each agent (can be adjusted based on historical performance)
    agent_performance_weights = {
        'risk': 1.2,      # Risk analyst gets higher weight
        'value': 1.0,     # Value investor baseline
        'macro': 0.8,     # Macro has been having JSON issues
        'wildcard': 1.1   # Wildcard provides good diversity
    }
    
    for agent_name, recommendations in raw_recommendations.items():
        if not recommendations:
            continue
        
        # Normalize allocations per-advisor to sum to 100%
        try:
            allocs = [float(r.get('allocation', 0)) for r in recommendations if isinstance(r, dict)]
            total_alloc = sum(allocs)
            if total_alloc > 0 and abs(total_alloc - 100.0) > 1e-6:
                scale = 100.0 / total_alloc
                for r in recommendations:
                    if isinstance(r, dict):
                        r['allocation'] = float(r.get('allocation', 0)) * scale
                logger.info(f"Renormalized {agent_name} allocations from {total_alloc:.1f}% to 100.0%")
        except Exception:
            # If normalization fails, proceed with raw values
            pass

        picks = []
        for i, rec in enumerate(recommendations):
            # Use allocation percentage to determine rank (higher allocation = better rank)
            allocations = [r.get('allocation', 0) for r in recommendations]
            ranks = sorted(range(len(allocations)), key=lambda x: allocations[x], reverse=True)
            rank = ranks.index(i) + 1
            
            pick = AgentPick(
                ticker=rec.get('ticker', ''),
                rank=rank,
                allocation=rec.get('allocation', 0),
                confidence=rec.get('confidence', 0.5),
                justification=rec.get('justification', ''),
                sell_trigger=rec.get('sell_trigger', '')
            )
            picks.append(pick)
        
        agent_recs = AgentRecommendations(
            agent_name=agent_name,
            picks=picks,
            performance_weight=agent_performance_weights.get(agent_name, 1.0)
        )
        agent_recs_list.append(agent_recs)
    
    return agent_recs_list 