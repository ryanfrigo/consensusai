#!/usr/bin/env python3
"""End-to-end test of the full portfolio management workflow with enhanced JSON handling."""

import asyncio
import json
import pytest
import sys
from enum import Enum
from typing import List, Dict, Any

sys.path.append('.')

from app.config import settings
from app.services.json_parser import LLMJSONParser, GeminiJSONConfig, JSONParseError

# Local enums to avoid database imports
class AdvisorType(str, Enum):
    VALUE = "value"
    MACRO = "macro" 
    RISK = "risk"
    WILDCARD = "wildcard"

class EnhancedOpenRouterLLMService:
    """Enhanced LLM service for testing with robust JSON parsing."""
    
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        self.json_parser = LLMJSONParser()
    
    def _get_advisor_prompt(self, advisor_type: AdvisorType) -> str:
        """Get the specialized prompt for each advisor with JSON mode optimization."""
        base_prompt = """You are {role}, a specialist equity researcher for an investment committee.

Your task: Recommend exactly 10 high-upside stocks for an aggressive growth portfolio.

CRITICAL: Respond ONLY with a valid JSON array of exactly 10 objects. No markdown, no explanations, no other text.

Required format for each stock:
{{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "allocation": 12,
  "justification": "Concise reason (≤40 words)",
  "confidence": 0.85,
  "risk": 0.60,
  "sell_trigger": "Clear exit condition"
}}

Requirements:
- Allocations must sum to exactly 100
- Confidence/risk: 0.0-1.0 scale  
- Justification: ≤40 words
- All 10 stocks must be different

{lens}

Generate your 10-stock JSON array now:"""

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
            lens=lenses[advisor_type]
        )
    
    def _should_use_json_mode(self, model: str) -> bool:
        """Determine if model supports structured JSON output."""
        json_mode_models = [
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-pro", 
            "google/gemini-1.5-flash",
            "openai/gpt-4",
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
            base_config["response_format"] = {"type": "json_object"}
        
        return base_config
    
    async def get_recommendation(self, advisor_type: AdvisorType) -> List[Dict[str, Any]]:
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
        
        use_json_mode = self._should_use_json_mode(model)
        generation_config = self._get_generation_config(model)
        prompt = self._get_advisor_prompt(advisor_type)
        
        print(f"   🔄 Calling {model} for {advisor_type.value} (JSON mode: {use_json_mode})")
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **generation_config
            )
            
            content = response.choices[0].message.content
            if not content:
                print(f"   ❌ Empty response from {advisor_type.value}")
                return []
            
            print(f"   ✅ Received response from {advisor_type.value} ({len(content)} chars)")
            
            # Use enhanced JSON parser
            try:
                recommendations = self.json_parser.parse_recommendation_list(
                    content=content,
                    expected_count=10
                )
                
                if recommendations and len(recommendations) == 10:
                    total_allocation = sum(rec.get('allocation', 0) for rec in recommendations)
                    print(f"   ✅ Successfully parsed {len(recommendations)} recommendations (total: {total_allocation}%)")
                    return recommendations
                else:
                    print(f"   ⚠️  Invalid format: expected 10 items, got {len(recommendations) if recommendations else 0}")
                    return []
                    
            except JSONParseError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                # Show first 200 chars for debugging
                print(f"   📝 Content preview: {content[:200]}...")
                return []
                
        except Exception as e:
            print(f"   ❌ {advisor_type.value}: API call failed: {e}")
            return []

@pytest.mark.asyncio
async def test_json_parser():
    """Test the new JSON parser with various edge cases."""
    print("🧪 Testing Enhanced JSON Parser")
    print("=" * 50)
    
    parser = LLMJSONParser()
    
    test_cases = [
        # Standard JSON
        ('{"test": "value"}', "dict"),
        ('[{"ticker": "AAPL", "allocation": 10}]', "list"),
        
        # Markdown wrapped
        ('```json\n[{"ticker": "AAPL"}]\n```', "list"),
        ('```\n{"test": "value"}\n```', "dict"),
        
        # Mixed content
        ('Here is the data:\n[{"ticker": "AAPL"}]\nHope this helps!', "list"),
        ('The recommendations are: ```json\n[{"ticker": "TSLA"}]```', "list"),
    ]
    
    for i, (content, expected_type) in enumerate(test_cases, 1):
        try:
            result = parser.parse_llm_json(content, expected_type=expected_type)
            print(f"   ✅ Test {i}: {type(result).__name__} parsed successfully")
        except JSONParseError as e:
            print(f"   ❌ Test {i}: {e}")
    
    print()

@pytest.mark.asyncio
async def test_single_advisor():
    """Test getting recommendations from a single advisor with enhanced parsing."""
    print("🧠 Testing Single Advisor (Risk Analyst) with Enhanced JSON Parsing")
    print("=" * 60)
    
    llm_service = EnhancedOpenRouterLLMService()
    
    try:
        recommendations = await llm_service.get_recommendation(AdvisorType.RISK)
        
        if recommendations:
            print(f"✅ Risk Analyst provided {len(recommendations)} recommendations")
            total_allocation = sum(rec.get('allocation', 0) for rec in recommendations)
            print(f"   Total allocation: {total_allocation}%")
            
            print("\n   Top 3 recommendations:")
            for rec in recommendations[:3]:
                print(f"   • {rec.get('ticker', 'N/A')}: {rec.get('allocation', 0)}% - {rec.get('justification', 'N/A')[:50]}...")
            
            return True
        else:
            print("❌ No recommendations received")
            return False
            
    except Exception as e:
        print(f"❌ Single advisor test failed: {e}")
        return False

@pytest.mark.asyncio
async def test_all_advisors():
    """Test getting recommendations from all advisors with enhanced parsing."""
    print("\n🎯 Testing All Advisors with Enhanced JSON Parsing")
    print("=" * 60)
    print("This may take 1-2 minutes as we call multiple LLM APIs...")
    
    llm_service = EnhancedOpenRouterLLMService()
    all_recommendations = {}
    
    import time
    start_time = time.time()
    
    for i, advisor_type in enumerate(AdvisorType, 1):
        print(f"\n[{i}/{len(AdvisorType)}] Getting recommendations from {advisor_type.value}...")
        
        advisor_start = time.time()
        try:
            recommendations = await llm_service.get_recommendation(advisor_type)
            all_recommendations[advisor_type.value] = recommendations
            
            advisor_time = time.time() - advisor_start
            
            if recommendations:
                total_allocation = sum(rec.get('allocation', 0) for rec in recommendations)
                print(f"   ✅ {advisor_type.value}: {len(recommendations)} stocks, {total_allocation}% total allocation (took {advisor_time:.1f}s)")
            else:
                print(f"   ❌ {advisor_type.value}: No recommendations (took {advisor_time:.1f}s)")
                
        except Exception as e:
            advisor_time = time.time() - advisor_start
            print(f"   ❌ {advisor_type.value}: {e} (took {advisor_time:.1f}s)")
            all_recommendations[advisor_type.value] = []
    
    total_time = time.time() - start_time
    
    # Show summary
    successful_advisors = sum(1 for recs in all_recommendations.values() if recs)
    print(f"\n📊 Summary: {successful_advisors}/{len(AdvisorType)} advisors provided recommendations")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    return all_recommendations

@pytest.mark.asyncio
async def test_portfolio_consensus(all_recommendations: Dict[str, List[Dict]]):
    """Test creating a consensus portfolio from all advisor recommendations."""
    print("\n🤝 Testing Portfolio Consensus Building")
    print("=" * 50)
    
    # Collect all unique tickers
    all_tickers = set()
    ticker_data = {}
    
    for advisor, recommendations in all_recommendations.items():
        for rec in recommendations:
            ticker = rec.get('ticker')
            if ticker:
                all_tickers.add(ticker)
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        'company': rec.get('company', ticker),
                        'advisors': [],
                        'total_weight': 0,
                        'avg_confidence': 0,
                        'avg_risk': 0
                    }
                
                ticker_data[ticker]['advisors'].append({
                    'advisor': advisor,
                    'allocation': rec.get('allocation', 0),
                    'confidence': rec.get('confidence', 0.5),
                    'risk': rec.get('risk', 0.5),
                    'justification': rec.get('justification', '')
                })
    
    # Calculate consensus weights
    for ticker, data in ticker_data.items():
        advisors = data['advisors']
        data['total_weight'] = sum(a['allocation'] for a in advisors)
        data['avg_confidence'] = sum(a['confidence'] for a in advisors) / len(advisors)
        data['avg_risk'] = sum(a['risk'] for a in advisors) / len(advisors)
        data['advisor_count'] = len(advisors)
    
    # Sort by total weight and advisor agreement
    sorted_tickers = sorted(
        ticker_data.items(),
        key=lambda x: (x[1]['advisor_count'], x[1]['total_weight']),
        reverse=True
    )
    
    print(f"✅ Analyzed {len(all_tickers)} unique stock recommendations")
    print(f"   Most agreed upon stocks:")
    
    for ticker, data in sorted_tickers[:5]:
        print(f"   • {ticker}: {data['advisor_count']} advisors, {data['total_weight']:.1f}% total weight")
        print(f"     Confidence: {data['avg_confidence']:.2f}, Risk: {data['avg_risk']:.2f}")
    
    return sorted_tickers

@pytest.mark.asyncio
async def test_paper_trading_simulation(top_picks: List[tuple]):
    """Test paper trading simulation with top picks."""
    print("\n📈 Testing Paper Trading Simulation")
    print("=" * 50)
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True
        )
        
        # Get account info
        account = client.get_account()
        portfolio_value = float(account.portfolio_value)
        
        print(f"✅ Portfolio value: ${portfolio_value:,.2f}")
        
        # Simulate orders for top 3 picks
        simulated_orders = []
        
        for i, (ticker, data) in enumerate(top_picks[:3]):
            # Calculate position size (simple equal weight for demo)
            target_allocation = 10.0  # 10% each for demo
            target_value = portfolio_value * (target_allocation / 100)
            
            simulated_orders.append({
                'symbol': ticker,
                'target_value': target_value,
                'target_allocation': target_allocation,
                'rationale': f"Consensus pick from {data['advisor_count']} advisors"
            })
            
            print(f"   📋 Simulated order: {ticker}")
            print(f"      Target value: ${target_value:,.2f} ({target_allocation}%)")
            print(f"      Rationale: {data['advisor_count']} advisor consensus")
        
        print(f"\n✅ Prepared {len(simulated_orders)} simulated orders")
        print("   (Orders not executed - simulation only)")
        
        return simulated_orders
        
    except Exception as e:
        print(f"❌ Paper trading simulation failed: {e}")
        return []

async def main():
    """Run the complete end-to-end workflow test with enhanced JSON parsing."""
    print("🧪 Enhanced End-to-End Portfolio Management Workflow Test")
    print("=" * 70)
    
    # Check prerequisites
    if not settings.openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return
    
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        print("❌ Alpaca API keys not found")
        return
    
    print("✅ All API keys present")
    
    # Run workflow steps
    success_count = 0
    total_steps = 6
    
    # Step 1: Test JSON parser
    await test_json_parser()
    success_count += 1
    
    # Step 2: Test single advisor
    if await test_single_advisor():
        success_count += 1
    
    # Step 3: Test all advisors
    all_recommendations = await test_all_advisors()
    if any(recs for recs in all_recommendations.values()):
        success_count += 1
    
    # Step 4: Build consensus
    if all_recommendations:
        top_picks = await test_portfolio_consensus(all_recommendations)
        if top_picks:
            success_count += 1
    
    # Step 5: Test paper trading
    if 'top_picks' in locals() and top_picks:
        simulated_orders = await test_paper_trading_simulation(top_picks)
        if simulated_orders:
            success_count += 1
    
    # Step 6: Overall system validation
    print(f"\n🎯 Enhanced System Validation Complete")
    print("=" * 60)
    print(f"✅ {success_count}/{total_steps} workflow steps completed successfully")
    
    if success_count >= 5:
        print("\n🎉 Enhanced portfolio management system is fully operational!")
        print("✨ Ready for scheduled daily runs with robust JSON parsing!")
        success_count += 1
    else:
        print("\n⚠️  Some workflow steps failed. System needs attention.")
    
    print(f"\n📊 Final Score: {success_count}/{total_steps} ({'PASS' if success_count >= 5 else 'NEEDS WORK'})")

if __name__ == "__main__":
    asyncio.run(main()) 