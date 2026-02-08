#!/usr/bin/env python3
"""Enhanced LLM service with robust JSON parsing and Gemini JSON mode support."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from app.config import settings
from app.models import AdvisorType, RecommendationSchema
from app.services.json_parser import LLMJSONParser, GeminiJSONConfig, JSONParseError

logger = logging.getLogger(__name__)

class OpenRouterLLMService:
    """Enhanced LLM service with robust JSON handling."""
    
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        self.json_parser = LLMJSONParser()
    
    def _get_advisor_prompt(
        self, 
        advisor_type: AdvisorType, 
        use_json_mode: bool = True,
        portfolio_context: Optional[Dict[str, Any]] = None,
        fresh_start: bool = False
    ) -> str:
        """Get specialized prompt for each advisor with optional portfolio context."""
        
        # Build portfolio context section if available and not fresh start
        portfolio_section = ""
        if portfolio_context and not fresh_start:
            nav = portfolio_context.get('nav', 0)
            cash = portfolio_context.get('cash', 0)
            total_pnl = portfolio_context.get('total_pnl', 0)
            positions = portfolio_context.get('positions', {})
            previous_targets = portfolio_context.get('previous_targets', {})
            
            portfolio_section = f"""
## CURRENT PORTFOLIO CONTEXT
Portfolio NAV: ${nav:,.2f}
Available Cash: ${cash:,.2f}
Total P&L: ${total_pnl:,.2f}
Position Count: {len(positions)}

Current Holdings:
"""
            # Add current positions with performance
            for symbol, data in sorted(positions.items(), key=lambda x: x[1]['weight'], reverse=True):
                weight = data['weight'] * 100
                value = data['value']
                pnl = data['pnl']
                pnl_pct = (pnl / (value - pnl)) * 100 if (value - pnl) > 0 else 0
                portfolio_section += f"- {symbol}: {weight:.1f}% (${value:,.0f}) P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%)\n"
            
            if previous_targets:
                portfolio_section += f"\nPrevious Targets: {len(previous_targets)} stocks\n"
            
            portfolio_section += f"""
PORTFOLIO ANALYSIS INSTRUCTIONS:
- Consider current position sizes when recommending allocations
- Factor in P&L performance - trim winners, add to underperformers if justified
- Suggest strategic rebalancing rather than wholesale changes
- Account for available cash: ${cash:,.2f}
- Maintain your investment philosophy while being portfolio-aware
- If suggesting major changes, justify why current holdings should be reduced/eliminated

"""
        
        # Enhanced base prompt with portfolio awareness
        if use_json_mode:
            # All JSON mode models get the same prompt - they return arrays directly
            base_prompt = """You are {role}, a specialist equity researcher for an investment committee.

{portfolio_context}Your task: {task_description}

CRITICAL: Respond ONLY with a valid JSON array of exactly 10 objects. No markdown, no explanations, no other text.

Required format for each stock:
{{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "allocation": 12,
  "justification": "{justification_hint}",
  "confidence": 0.85,
  "risk": 0.60,
  "sell_trigger": "Clear exit condition"
}}

Requirements:
- Allocations must sum to exactly 100
- Confidence/risk: 0.0-1.0 scale
- Justification: ≤40 words{portfolio_rules}
- All 10 stocks must be different

{lens}

Generate your 10-stock JSON array now:"""
        else:
            # Fallback prompt for non-JSON mode models
            base_prompt = """You are {role}, a specialist equity researcher invited to a multi-model investment committee. {task_description}

{portfolio_context}Rules:
1. **Output valid JSON only**—no markdown, no prose, **no enclosing object**.
   **Your entire response must be exactly a JSON array of ten objects**, starting with `[` and ending with `]`.
   Example of each object:
   ```json
   {{
     "ticker":       "TSLA",
     "company":      "Tesla Inc.",
     "allocation":   12,
     "justification":"{justification_hint}",
     "confidence":   0.82,
     "risk":         0.74,
     "sell_trigger": "If autonomous-taxi rollout delayed > 2 yrs"
   }}
   ```
2. Allocations must sum to 100.
3. Keep each justification ≤ 40 words; avoid hype.{portfolio_rules}
4. Do not repeat another agent's picks verbatim; diversity is valued.
5. If unsure, still give your best estimate—no "cannot answer."

{lens}
Task: Draft your recommendations now."""

        # Determine task description and rules based on context
        if portfolio_context and not fresh_start:
            task_description = "Recommend portfolio adjustments considering current holdings and performance"
            justification_hint = "Brief reason for allocation/rebalancing (≤40 words)"
            portfolio_rules = "\n- Consider current holdings in your allocations"
        else:
            task_description = "Recommend exactly 10 high-upside stocks for an aggressive growth portfolio"
            justification_hint = "Concise reason (≤40 words)"
            portfolio_rules = ""

        lenses = {
            AdvisorType.RISK: "You are RiskAnalyst, focused on governance, balance-sheet health, and downside scenarios.\nLens: surface governance red flags, regulatory vulnerabilities, financing cliffs, or single-point-of-failure supply chains; penalise such risks in your \"risk\" score.",
            
            AdvisorType.MACRO: "You are MacroStrategist, an expert on macroeconomic and geopolitical trends.\nLens: identify stocks positioned to benefit from major themes—interest-rate shifts, inflation dynamics, energy transition, digital infrastructure rollouts, and global trade flows.",
            
            AdvisorType.WILDCARD: "You are WildcardAnalyst, harnessing Grok's contrarian intuition.\nLens: seek out-of-consensus, under-the-radar names in niche sectors or novel business models that the herd may overlook. Embrace bold, unconventional ideas.",
            
            AdvisorType.VALUE: "You are ValueInvestor, the o3 model trained in deep value investing.\nLens: hunt companies trading below intrinsic value with strong free-cash-flow yields, margin-of-safety, and identifiable catalysts for re-rating."
        }
        
        role_names = {
            AdvisorType.RISK: "RiskAnalyst",
            AdvisorType.MACRO: "MacroStrategist", 
            AdvisorType.WILDCARD: "WildcardAnalyst",
            AdvisorType.VALUE: "ValueInvestor"
        }
        
        return base_prompt.format(
            role=role_names[advisor_type],
            lens=lenses[advisor_type],
            portfolio_context=portfolio_section,
            task_description=task_description,
            justification_hint=justification_hint,
            portfolio_rules=portfolio_rules
        )
    
    def _should_use_json_mode(self, model: str) -> bool:
        """Determine if model supports structured JSON output."""
        # Models that support JSON mode
        json_mode_models = [
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-pro",
            "google/gemini-1.5-flash",
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "openai/gpt-oss-120b",
            "anthropic/claude-3"
        ]

        return any(json_model in model for json_model in json_mode_models)
    
    def _get_generation_config(self, model: str) -> Dict[str, Any]:
        """Get appropriate generation config based on model capabilities."""
        base_config = {
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60.0
        }
        
        # Add JSON mode config for supported models
        if self._should_use_json_mode(model) and "gemini" in model.lower():
            # Use Gemini's native JSON mode
            gemini_config = GeminiJSONConfig.get_generation_config()
            base_config.update({
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "response_mime_type": gemini_config["response_mime_type"],
                    "response_schema": gemini_config["response_schema"]
                }
            })
        elif self._should_use_json_mode(model):
            # Use standard JSON mode for other models
            base_config["response_format"] = {"type": "json_object"}
        
        return base_config
    
    async def get_recommendation(
        self, 
        advisor_type: AdvisorType,
        portfolio_context: Optional[Dict[str, any]] = None,
        fresh_start: Optional[bool] = None
    ) -> List[RecommendationSchema]:
        """Get recommendation from a specific advisor with enhanced JSON handling."""
        
        # Map advisor types to configuration keys
        advisor_to_config_key = {
            AdvisorType.RISK: "RISK_ANALYST",
            AdvisorType.MACRO: "MACRO_STRATEGIST", 
            AdvisorType.WILDCARD: "WILDCARD",
            AdvisorType.VALUE: "VALUE_INVESTOR"
        }
        
        config_key = advisor_to_config_key[advisor_type]
        model = settings.advisor_models[config_key]
        
        # Determine fresh start mode
        if fresh_start is None:
            fresh_start = settings.fresh_start_mode
        
        # Only use portfolio context if enabled and not fresh start
        use_context = portfolio_context and settings.use_portfolio_context and not fresh_start
        
        # Determine if we can use JSON mode
        use_json_mode = self._should_use_json_mode(model)
        generation_config = self._get_generation_config(model)
        
        prompt = self._get_advisor_prompt(
            advisor_type, 
            use_json_mode, 
            portfolio_context if use_context else None,
            fresh_start
        )
        
        context_status = "with portfolio context" if use_context else "fresh start mode" if fresh_start else "no context available"
        logger.info(f"Calling {model} for {advisor_type.value} ({context_status}, JSON mode: {use_json_mode})")
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **generation_config
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError(f"Empty response from {advisor_type.value} advisor")
            
            logger.debug(f"Raw response from {advisor_type.value}: {len(content)} chars")

            # Parse JSON with robust error handling
            try:
                recommendations_data = self.json_parser.parse_recommendation_list(
                    content=content, 
                    expected_count=10
                )
                
                if not recommendations_data:
                    logger.warning(f"No valid recommendations from {advisor_type.value}")
                    return []
                    
                logger.info(f"Successfully parsed {len(recommendations_data)} recommendations from {advisor_type.value}")
                
            except JSONParseError as e:
                logger.error(f"JSON parsing failed for {advisor_type.value}: {e}")
                logger.error(f"Raw content: {content[:500]}...")
                return []
            
            # Convert to RecommendationSchema objects
            recommendations = []
            total_allocation = 0
            
            for i, rec_data in enumerate(recommendations_data):
                try:
                    # Validate required fields
                    required_fields = ["ticker", "company", "allocation", "justification", "confidence", "risk"]
                    missing_fields = [field for field in required_fields if field not in rec_data]
                    
                    if missing_fields:
                        logger.warning(f"Recommendation {i} missing fields: {missing_fields}")
                        continue
                    
                    # Create RecommendationSchema object
                    recommendation = RecommendationSchema(
                        ticker=str(rec_data["ticker"]).upper(),
                        company=str(rec_data["company"]),
                        allocation=int(rec_data["allocation"]),
                        justification=str(rec_data["justification"])[:200],  # Truncate if too long
                        confidence=float(rec_data["confidence"]),
                        risk=float(rec_data["risk"]),
                        sell_trigger=rec_data.get("sell_trigger", "Review quarterly"),
                        advisor_type=advisor_type,
                        timestamp=None  # Will be set by calling code
                    )
                    
                    recommendations.append(recommendation)
                    total_allocation += recommendation.allocation
                    
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to parse recommendation {i} from {advisor_type.value}: {e}")
                    continue
            
            # Validate total allocation
            if abs(total_allocation - 100) > 5:  # Allow 5% tolerance
                logger.warning(f"{advisor_type.value} allocations sum to {total_allocation}%, not 100%")
            
            logger.info(f"{advisor_type.value} returned {len(recommendations)} valid recommendations (total allocation: {total_allocation}%)")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendation from {advisor_type.value}: {e}")
            return []

    async def get_all_recommendations(
        self,
        portfolio_context: Optional[Dict[str, any]] = None,
        fresh_start: Optional[bool] = None
    ) -> Dict[AdvisorType, List[RecommendationSchema]]:
        """Get recommendations from all advisors concurrently."""
        
        context_mode = "fresh start" if fresh_start else "portfolio-aware" if portfolio_context else "no context"
        logger.info(f"Getting recommendations from all advisors ({context_mode} mode)...")
        
        # Create tasks for all advisors
        tasks = []
        for advisor_type in AdvisorType:
            task = self.get_recommendation(advisor_type, portfolio_context, fresh_start)
            tasks.append((advisor_type, task))
        
        # Execute all tasks concurrently
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        # Process results
        for (advisor_type, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Failed to get recommendations from {advisor_type.value}: {result}")
                results[advisor_type] = []
            else:
                results[advisor_type] = result
        
        successful_advisors = sum(1 for recs in results.values() if recs)
        logger.info(f"Successfully got recommendations from {successful_advisors}/{len(AdvisorType)} advisors")
        
        return results


# Legacy compatibility - keep the old class name
LLMService = OpenRouterLLMService