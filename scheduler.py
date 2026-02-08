#!/usr/bin/env python3
"""
Portfolio Management Scheduler
Runs the full portfolio management workflow daily at market open.
"""

import asyncio
import json
import logging
import sys
import time as time_module
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List

import schedule

sys.path.append(".")

from app.config import settings
from consensus_engine import (
    AdvancedConsensusEngine,
    convert_raw_recommendations_to_agent_recs,
)
from app.services.json_parser import GeminiJSONConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("portfolio_scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Local enums to avoid database imports
class AdvisorType(str, Enum):
    VALUE = "value"
    MACRO = "macro"
    RISK = "risk"
    WILDCARD = "wildcard"


class PortfolioManager:
    """Main portfolio management class."""

    def __init__(self):
        """Initialize the portfolio manager."""
        from openai import AsyncOpenAI

        self.llm_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url
        )

        # Track daily run statistics
        self.daily_stats = {
            "run_date": None,
            "advisors_successful": 0,
            "total_advisors": len(AdvisorType),
            "recommendations_count": 0,
            "consensus_picks": [],
            "simulated_trades": [],
            "execution_time_seconds": 0,
            "errors": [],
        }

    def _get_advisor_prompt(self, advisor_type: AdvisorType) -> str:
        """Get the specialized prompt for each advisor."""
        base_prompt = """You are {role}, a specialist equity researcher invited to a multi-model investment committee. Your job in **Round 0** is to propose exactly **ten** high-upside "moonshot" stocks for an aggressive, decade-long growth portfolio.

Rules:
1. **Output valid JSON only**—no markdown, no prose, **no enclosing object**.
   **Your entire response must be exactly a JSON array of ten objects**, starting with `[` and ending with `]`.
   Example of each object:
   ```json
   {{
     "ticker":       "TSLA",
     "company":      "Tesla Inc.",
     "allocation":   12,
     "justification":"Short (≤ 40 words) reason for the investment. MUST BE a single line with no newlines.",
     "confidence":   0.82,
     "risk":         0.74,
     "sell_trigger": "If autonomous-taxi rollout delayed > 2 yrs"
   }}
   ```
2. Allocations must sum to 100.
3. Keep each justification ≤ 40 words; avoid hype. **CRITICAL: The justification value must not contain any newline characters.**
4. Do not repeat another agent's picks verbatim; diversity is valued.
5. If unsure, still give your best estimate—no "cannot answer."

{lens}
Task: Draft your Round 0 proposal now."""

        lenses = {
            AdvisorType.RISK: 'You are RiskAnalyst, focused on governance, balance-sheet health, and downside scenarios.\nLens: surface governance red flags, regulatory vulnerabilities, financing cliffs, or single-point-of-failure supply chains; penalise such risks in your "risk" score.',
            AdvisorType.MACRO: "You are MacroStrategist, an expert on macroeconomic and geopolitical trends.\nLens: identify stocks positioned to benefit from major themes—interest-rate shifts, inflation dynamics, energy transition, digital infrastructure rollouts, and global trade flows.",
            AdvisorType.WILDCARD: "You are WildcardAnalyst, harnessing Grok's contrarian intuition.\nLens: seek out-of-consensus, under-the-radar names in niche sectors or novel business models that the herd may overlook. Embrace bold, unconventional ideas.",
            AdvisorType.VALUE: "You are ValueInvestor, the o3 model trained in deep value investing.\nLens: hunt companies trading below intrinsic value with strong free-cash-flow yields, margin-of-safety, and identifiable catalysts for re-rating.",
        }

        role_names = {
            AdvisorType.RISK: "RiskAnalyst",
            AdvisorType.MACRO: "MacroStrategist",
            AdvisorType.WILDCARD: "WildcardAnalyst",
            AdvisorType.VALUE: "ValueInvestor",
        }

        return base_prompt.format(
            role=role_names[advisor_type], lens=lenses[advisor_type]
        )

    async def get_advisor_recommendation(
        self, advisor_type: AdvisorType
    ) -> List[Dict[str, Any]]:
        """Get recommendation from a specific advisor using enhanced JSON parsing."""
        advisor_to_config_key = {
            AdvisorType.VALUE: "VALUE_INVESTOR",
            AdvisorType.MACRO: "MACRO_STRATEGIST",
            AdvisorType.RISK: "RISK_ANALYST",
            AdvisorType.WILDCARD: "WILDCARD",
        }

        config_key = advisor_to_config_key[advisor_type]
        model = settings.advisor_models[config_key]
        prompt = self._get_advisor_prompt(advisor_type)

        logger.info(f"Getting recommendation from {advisor_type.value} using {model}")

        try:
            # Import JSON parser
            from app.services.json_parser import JSONParseError, LLMJSONParser

            json_parser = LLMJSONParser()

            # Configure JSON-mode for models that support it (esp. Gemini)
            generation_kwargs: Dict[str, Any] = {
                "max_tokens": 2000,
                "temperature": 0.7,
                "timeout": 60.0,
            }
            model_lower = (model or "").lower()
            supports_json_mode = any(
                tag in model_lower for tag in [
                    "gemini", "gpt-4", "gpt-3.5", "claude-3", "gpt-4o", "gpt-4-turbo"
                ]
            )
            if supports_json_mode:
                generation_kwargs["response_format"] = {"type": "json_object"}
                # Gemini via OpenRouter supports structured schema in extra_body
                if "gemini" in model_lower:
                    generation_kwargs["extra_body"] = {
                        "response_mime_type": "application/json",
                        "response_schema": GeminiJSONConfig.get_recommendation_schema(),
                    }

            response = await self.llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **generation_kwargs,
            )

            content = response.choices[0].message.content.strip()
            logger.debug(
                f"Raw response from {advisor_type.value}: {len(content)} characters"
            )

            # Use enhanced JSON parser instead of manual parsing
            try:
                recommendations = json_parser.parse_recommendation_list(
                    content=content, expected_count=10
                )

                if recommendations:
                    total_allocation = sum(
                        rec.get("allocation", 0) for rec in recommendations
                    )
                    logger.info(
                        f"{advisor_type.value} returned {len(recommendations)} recommendations (total: {total_allocation}%)"
                    )
                    if len(recommendations) < 10:
                        logger.warning(
                            f"{advisor_type.value}: Received {len(recommendations)} (<10); accepting partial list"
                        )
                    elif len(recommendations) > 10:
                        logger.warning(
                            f"{advisor_type.value}: Received >10; trimmed to 10 by parser"
                        )
                    return recommendations
                else:
                    logger.warning(f"{advisor_type.value}: No recommendations received")
                    return []

            except JSONParseError as e:
                logger.error(f"JSON parsing failed for {advisor_type.value}: {e}")
                logger.debug(f"Raw content: {content[:500]}...")
                return []

        except Exception as e:
            logger.error(f"Failed to get recommendation from {advisor_type.value}: {e}")
            return []

    async def get_all_recommendations(self) -> Dict[str, List[Dict]]:
        """Get recommendations from all advisors."""
        logger.info("Getting recommendations from all advisors...")
        all_recommendations = {}

        for advisor_type in AdvisorType:
            try:
                recommendations = await self.get_advisor_recommendation(advisor_type)
                all_recommendations[advisor_type.value] = recommendations

                if recommendations:
                    self.daily_stats["advisors_successful"] += 1
                    self.daily_stats["recommendations_count"] += len(recommendations)
                    total_allocation = sum(
                        rec.get("allocation", 0) for rec in recommendations
                    )
                    logger.info(
                        f"{advisor_type.value}: {len(recommendations)} stocks, {total_allocation}% total allocation"
                    )
                else:
                    logger.warning(f"{advisor_type.value}: No recommendations received")

            except Exception as e:
                error_msg = (
                    f"Failed to get recommendations from {advisor_type.value}: {e}"
                )
                logger.error(error_msg)
                self.daily_stats["errors"].append(error_msg)
                all_recommendations[advisor_type.value] = []

        logger.info(
            f"Summary: {self.daily_stats['advisors_successful']}/{self.daily_stats['total_advisors']} advisors provided recommendations"
        )
        return all_recommendations

    def build_consensus_portfolio(
        self, all_recommendations: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Build consensus portfolio using advanced mathematical algorithm."""
        logger.info("Building advanced consensus portfolio...")

        # Convert raw recommendations to structured format
        agent_recommendations = convert_raw_recommendations_to_agent_recs(
            all_recommendations
        )

        if not agent_recommendations:
            logger.warning("No agent recommendations to process")
            return {}

        # Initialize advanced consensus engine
        consensus_engine = AdvancedConsensusEngine(
            lambda_decay=0.25,  # Exponential decay for rank weighting
            max_single_weight=settings.max_position_weight,  # Use config setting (15%)
            min_weight_threshold=0.005,  # 0.5% minimum position size
            winsorize_percentiles=(5, 95),  # Trim outliers
            turnover_brake=settings.max_weight_delta,  # Use config setting (5%)
        )

        # Build consensus portfolio
        consensus_result = consensus_engine.build_consensus_portfolio(
            agent_recommendations, target_portfolio_value=100000.0
        )

        # Log detailed results
        logger.info(f"Advanced consensus complete:")
        logger.info(f"  • Total positions: {consensus_result['total_positions']}")
        logger.info(
            f"  • Max single position: {consensus_result['max_position_weight']:.1%}"
        )
        logger.info(f"  • Portfolio coverage: {consensus_result['total_coverage']:.1%}")
        logger.info(
            f"  • Unique picks processed: {consensus_result['total_unique_picks']}"
        )
        logger.info(
            f"  • Final portfolio picks: {consensus_result['final_portfolio_picks']}"
        )

        # Log top 10 positions
        portfolio_allocations = consensus_result["portfolio_allocations"]
        sorted_positions = sorted(
            portfolio_allocations.items(), key=lambda x: x[1]["weight"], reverse=True
        )

        logger.info("Top 10 consensus positions:")
        for i, (ticker, data) in enumerate(sorted_positions[:10]):
            agents = ", ".join(data["contributing_agents"])
            logger.info(
                f"  {i+1:2d}. {ticker}: {data['weight']:.1%} (${data['dollar_amount']:,.0f}) - {agents}"
            )

        # Store consensus picks for daily stats
        self.daily_stats["consensus_picks"] = []
        for ticker, data in sorted_positions:
            pick_info = {
                "ticker": ticker,
                "weight": data["weight"],
                "dollar_amount": data["dollar_amount"],
                "consensus_score": data["consensus_score"],
                "contributing_agents": data["contributing_agents"],
                "agent_count": len(data["contributing_agents"]),
            }
            self.daily_stats["consensus_picks"].append(pick_info)

        return consensus_result

    async def simulate_trades(self, consensus_result: Dict[str, Any]) -> List[Dict]:
        """Place actual paper trades based on consensus portfolio."""
        logger.info("Placing paper trades...")

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=True,
            )

            # Get account info & market clock
            account = client.get_account()
            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)
            try:
                clock = client.get_clock()  # precise market clock
                market_open = bool(getattr(clock, "is_open", False))
            except Exception:
                market_open = False

            logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            logger.info(f"Buying power: ${buying_power:,.2f}")
            if market_open:
                logger.info(
                    "Market open per Alpaca clock - attempting to place actual paper trading orders..."
                )
                execute_orders = True
            else:
                logger.info(
                    "Market is closed per Alpaca clock - simulating orders only"
                )
                execute_orders = False

            # Get portfolio allocations from consensus result
            portfolio_allocations = consensus_result.get("portfolio_allocations", {})

            if not portfolio_allocations:
                logger.warning("No portfolio allocations found in consensus result")
                return []

            # Fetch current positions to determine liquidations
            executed_orders = []
            total_order_value = 0
            try:
                current_positions = client.get_all_positions()
            except Exception as e:
                logger.warning(f"Failed to fetch current positions for liquidation planning: {e}")
                current_positions = []

            current_symbols = {p.symbol for p in current_positions}
            target_symbols = set(portfolio_allocations.keys())
            symbols_to_liquidate = sorted(list(current_symbols - target_symbols))

            # First, process liquidations for symbols not in target portfolio
            for pos in current_positions:
                if pos.symbol not in target_symbols:
                    order_info = {
                        "symbol": pos.symbol,
                        "target_value": 0.0,
                        "target_weight": 0.0,
                        "contributing_agents": [],
                        "agent_count": 0,
                        "consensus_score": 0.0,
                        "order_status": "not_executed",
                        "order_id": None,
                        "shares_ordered": 0,
                    }

                    try:
                        # Use fractional shares for accurate liquidation
                        shares_to_sell = abs(float(getattr(pos, "qty", 0)))
                        shares_to_sell = round(shares_to_sell, 6)
                    except Exception:
                        shares_to_sell = 0

                    if shares_to_sell <= 0.000001:  # Account for floating point precision
                        order_info["order_status"] = "no_shares"
                        executed_orders.append(order_info)
                        continue

                    if execute_orders:
                        try:
                            market_order_data = MarketOrderRequest(
                                symbol=pos.symbol,
                                qty=shares_to_sell,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                            )
                            order = client.submit_order(order_data=market_order_data)
                            order_info["order_status"] = "submitted"
                            order_info["order_id"] = str(order.id)
                            order_info["shares_ordered"] = shares_to_sell
                            logger.info(
                                f"✅ LIQUIDATE: {pos.symbol} - {shares_to_sell:.6f} shares (not in targets)"
                            )
                        except Exception as e:
                            order_info["order_status"] = "failed"
                            order_info["error"] = str(e)
                            logger.error(f"❌ LIQUIDATION FAILED: {pos.symbol} - {e}")
                    else:
                        order_info["order_status"] = "simulated_market_closed"
                        logger.info(f"📋 Simulated liquidation: {pos.symbol} - {shares_to_sell:.6f} shares")

                    executed_orders.append(order_info)

            # Then, process target buys sorted by weight (largest positions first)
            sorted_allocations = sorted(
                portfolio_allocations.items(),
                key=lambda x: x[1]["weight"],
                reverse=True,
            )

            for ticker, allocation_data in sorted_allocations:
                target_value = allocation_data["dollar_amount"]
                weight = allocation_data["weight"]
                contributing_agents = allocation_data["contributing_agents"]

                order_info = {
                    "symbol": ticker,
                    "target_value": target_value,
                    "target_weight": weight,
                    "contributing_agents": contributing_agents,
                    "agent_count": len(contributing_agents),
                    "consensus_score": allocation_data["consensus_score"],
                    "order_status": "not_executed",
                    "order_id": None,
                    "shares_ordered": 0,
                }

                # Dynamic scaling: adjust remaining targets to fit remaining BP (80% cap)
                conservative_buying_power = buying_power * 0.8
                remaining_power = max(
                    0.0, conservative_buying_power - total_order_value
                )
                if execute_orders and remaining_power > 0:
                    # Scale target if it exceeds remaining_power / remaining_positions
                    remaining_positions = max(
                        1, len(sorted_allocations) - len(executed_orders)
                    )
                    fair_share = remaining_power / remaining_positions
                    if target_value > fair_share:
                        logger.debug(
                            f"Scaling {ticker} target from ${target_value:,.0f} to ${fair_share:,.0f} due to BP"
                        )
                        target_value = fair_share

                if (
                    execute_orders
                    and target_value >= 100
                    and total_order_value + target_value <= conservative_buying_power
                ):
                    try:
                        # Pre-check asset tradability
                        try:
                            asset = client.get_asset(ticker)
                            if (
                                not getattr(asset, "tradable", True)
                                or getattr(asset, "status", "active") != "active"
                            ):
                                order_info["order_status"] = "asset_inactive"
                                order_info["error"] = (
                                    f"asset {ticker} is not active/tradable"
                                )
                                logger.warning(
                                    f"⚠️ Inactive/non-tradable asset: {ticker} - skipping"
                                )
                                executed_orders.append(order_info)
                                continue
                        except Exception:
                            # If asset lookup fails, continue and let submit fail if needed
                            pass

                        # Get current market price for accurate share calculation
                        try:
                            data_client = StockHistoricalDataClient(
                                api_key=settings.alpaca_api_key,
                                secret_key=settings.alpaca_secret_key,
                            )
                            quote_request = StockLatestQuoteRequest(
                                symbol_or_symbols=[ticker]
                            )
                            quotes = data_client.get_stock_latest_quote(quote_request)
                            quote = quotes[ticker]

                            # Use midpoint of bid/ask for more accurate pricing
                            bid_price = float(quote.bid_price or 0)
                            ask_price = float(quote.ask_price or 0)

                            if bid_price > 0 and ask_price > 0:
                                current_price = (bid_price + ask_price) / 2
                            elif ask_price > 0:
                                current_price = ask_price
                            elif bid_price > 0:
                                current_price = bid_price
                            else:
                                raise ValueError("No valid bid/ask prices")

                            # Calculate fractional shares - no minimum share requirement
                            estimated_shares = target_value / current_price

                            # Round to appropriate decimal places (Alpaca supports fractional shares)
                            estimated_shares = round(estimated_shares, 6)

                            logger.debug(
                                f"Using market price ${current_price:.2f} (${bid_price:.2f} bid / ${ask_price:.2f} ask) for {ticker}"
                            )

                        except Exception as e:
                            # Fallback to estimation if market data fails
                            logger.debug(
                                f"Market data unavailable for {ticker}, using estimation: {e}"
                            )
                            estimated_price_per_share = max(
                                10, target_value / 100
                            )  # Conservative estimate
                            estimated_shares = round(target_value / estimated_price_per_share, 6)

                        # Create market order
                        market_order_data = MarketOrderRequest(
                            symbol=ticker,
                            qty=estimated_shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )

                        # Submit order
                        order = client.submit_order(order_data=market_order_data)

                        order_info["order_status"] = "submitted"
                        order_info["order_id"] = str(order.id)
                        order_info["shares_ordered"] = estimated_shares
                        total_order_value += target_value

                        logger.info(
                            f"✅ ORDER PLACED: {ticker} - {estimated_shares:.6f} shares (${target_value:,.0f}, {weight:.1%}) - {len(contributing_agents)} agents"
                        )

                    except Exception as e:
                        order_info["order_status"] = "failed"
                        order_info["error"] = str(e)
                        logger.error(f"❌ ORDER FAILED: {ticker} - {e}")

                else:
                    if not execute_orders:
                        order_info["order_status"] = "simulated_market_closed"
                        logger.info(
                            f"📋 Simulated order: {ticker} - ${target_value:,.0f} ({weight:.1%}) - {len(contributing_agents)} agents"
                        )
                    elif target_value < 100:
                        order_info["order_status"] = "below_minimum"
                        logger.debug(
                            f"⚠️ Skipped {ticker}: below $100 minimum (${target_value:.0f})"
                        )
                    elif total_order_value + target_value > conservative_buying_power:
                        order_info["order_status"] = "insufficient_buying_power"
                        remaining_power = conservative_buying_power - total_order_value
                        logger.warning(
                            f"⚠️ Insufficient buying power for {ticker}: need ${target_value:,.0f}, have ${remaining_power:,.0f} remaining"
                        )
                    else:
                        order_info["order_status"] = "skipped_other"
                        logger.debug(
                            f"⚠️ Skipped {ticker}: ${target_value:,.0f} for other reasons"
                        )

                executed_orders.append(order_info)
                self.daily_stats["simulated_trades"].append(order_info)

            successful_orders = len(
                [o for o in executed_orders if o["order_status"] == "submitted"]
            )
            total_target_value = sum(a["target_value"] for a in executed_orders)

            logger.info(
                f"Order summary: {successful_orders}/{len(executed_orders)} orders successfully placed"
            )
            logger.info(
                f"Total target value: ${total_target_value:,.0f} (${total_order_value:,.0f} submitted)"
            )

            return executed_orders

        except Exception as e:
            error_msg = f"Paper trading execution failed: {e}"
            logger.error(error_msg)
            self.daily_stats["errors"].append(error_msg)
            return []

    def save_daily_report(self):
        """Save daily report to file."""
        report_filename = (
            f"daily_report_{self.daily_stats['run_date'].strftime('%Y%m%d')}.json"
        )

        try:
            with open(report_filename, "w") as f:
                json.dump(self.daily_stats, f, indent=2, default=str)

            logger.info(f"Daily report saved to {report_filename}")

            # Also create a human-readable summary
            summary_filename = (
                f"daily_summary_{self.daily_stats['run_date'].strftime('%Y%m%d')}.txt"
            )
            with open(summary_filename, "w") as f:
                f.write(
                    f"Portfolio Management Daily Report - {self.daily_stats['run_date'].strftime('%Y-%m-%d')}\n"
                )
                f.write("=" * 60 + "\n\n")

                f.write(f"Execution Summary:\n")
                f.write(
                    f"- Run time: {self.daily_stats['execution_time_seconds']:.1f} seconds\n"
                )
                f.write(
                    f"- Advisors successful: {self.daily_stats['advisors_successful']}/{self.daily_stats['total_advisors']}\n"
                )
                f.write(
                    f"- Total recommendations: {self.daily_stats['recommendations_count']}\n"
                )
                f.write(
                    f"- Simulated trades: {len(self.daily_stats['simulated_trades'])}\n"
                )
                f.write(f"- Errors: {len(self.daily_stats['errors'])}\n\n")

                if self.daily_stats["consensus_picks"]:
                    f.write("Top Consensus Picks:\n")
                    for pick in self.daily_stats["consensus_picks"][:5]:
                        f.write(
                            f"- {pick['ticker']}: {pick['agent_count']} advisors, {pick['weight']:.1%} weight\n"
                        )
                    f.write("\n")

                if self.daily_stats["simulated_trades"]:
                    f.write("Simulated Trades:\n")
                    for trade in self.daily_stats["simulated_trades"]:
                        agents_count = trade.get(
                            "agent_count", len(trade.get("contributing_agents", []))
                        )
                        target_weight = trade.get("target_weight", 0)
                        f.write(
                            f"- {trade['symbol']}: ${trade['target_value']:,.0f} ({target_weight:.1%}) - {agents_count} agents\n"
                        )
                    f.write("\n")

                if self.daily_stats["errors"]:
                    f.write("Errors:\n")
                    for error in self.daily_stats["errors"]:
                        f.write(f"- {error}\n")

            logger.info(f"Daily summary saved to {summary_filename}")

        except Exception as e:
            logger.error(f"Failed to save daily report: {e}")

    async def run_daily_workflow(self):
        """Run the complete daily portfolio management workflow."""
        import time

        start_time = time.time()

        self.daily_stats["run_date"] = datetime.now()
        logger.info(
            f"Starting daily portfolio management workflow at {self.daily_stats['run_date']}"
        )

        try:
            # Step 1: Get recommendations from all advisors
            all_recommendations = await self.get_all_recommendations()

            # Step 2: Build consensus portfolio
            if any(recs for recs in all_recommendations.values()):
                consensus_result = self.build_consensus_portfolio(all_recommendations)

                # Step 3: Execute trades
                if consensus_result and consensus_result.get("portfolio_allocations"):
                    await self.simulate_trades(consensus_result)
                else:
                    logger.warning("No consensus portfolio available for trading")
            else:
                logger.error("No recommendations received from any advisor")

            # Calculate execution time
            self.daily_stats["execution_time_seconds"] = time.time() - start_time

            # Save report
            self.save_daily_report()

            # Final summary
            success_rate = (
                self.daily_stats["advisors_successful"]
                / self.daily_stats["total_advisors"]
                * 100
            )
            logger.info(
                f"Daily workflow completed in {self.daily_stats['execution_time_seconds']:.1f}s"
            )
            logger.info(
                f"Success rate: {success_rate:.1f}% ({self.daily_stats['advisors_successful']}/{self.daily_stats['total_advisors']} advisors)"
            )

            if self.daily_stats["advisors_successful"] >= 2:
                logger.info("✅ Daily workflow SUCCESSFUL - Ready for next market day")
            else:
                logger.warning("⚠️ Daily workflow had issues - Check logs and reports")

        except Exception as e:
            self.daily_stats["execution_time_seconds"] = time.time() - start_time
            error_msg = f"Daily workflow failed: {e}"
            logger.error(error_msg)
            self.daily_stats["errors"].append(error_msg)
            self.save_daily_report()


def run_scheduled_workflow():
    """Run the scheduled workflow (synchronous wrapper for async function)."""
    logger.info("🔔 Scheduled portfolio management workflow triggered")

    try:
        manager = PortfolioManager()
        asyncio.run(manager.run_daily_workflow())
    except Exception as e:
        logger.error(f"Failed to run scheduled workflow: {e}")


def setup_scheduler():
    """Set up the daily scheduler."""
    # Schedule for market open (9:30 AM ET)
    # Note: Adjust timezone as needed for your location
    schedule.every().monday.at("09:30").do(run_scheduled_workflow)
    schedule.every().tuesday.at("09:30").do(run_scheduled_workflow)
    schedule.every().wednesday.at("09:30").do(run_scheduled_workflow)
    schedule.every().thursday.at("09:30").do(run_scheduled_workflow)
    schedule.every().friday.at("09:30").do(run_scheduled_workflow)

    logger.info("📅 Scheduler configured for market days at 9:30 AM")
    logger.info("Scheduled days: Monday-Friday at 09:30")


def main():
    """Main scheduler loop."""
    logger.info("🚀 Portfolio Management Scheduler Starting")
    logger.info("=" * 50)

    # Check configuration
    if not all(
        [
            settings.openrouter_api_key,
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
        ]
    ):
        logger.error("❌ Missing required API keys in configuration")
        return

    logger.info("✅ Configuration validated")

    # Set up scheduler
    setup_scheduler()

    # Option to run immediately for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--run-now":
        logger.info("🧪 Running workflow immediately for testing...")
        try:
            run_scheduled_workflow()
        except KeyboardInterrupt:
            logger.info("⛔ Immediate run interrupted by user")
        return

    # Start scheduler loop
    logger.info("⏰ Scheduler started - waiting for scheduled times...")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("👋 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")


if __name__ == "__main__":
    main()
