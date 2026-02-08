"""Configuration management for the portfolio API."""

import os
from enum import Enum
from typing import Dict

from dotenv import load_dotenv

# Load environment variables from .env if present (non-fatal when missing)
load_dotenv()


class AdvisorModel(str, Enum):
    """Enum for advisor model assignments."""

    RISK_ANALYST = "anthropic/claude-sonnet-4"
    MACRO_STRATEGIST = "google/gemini-2.5-pro"
    WILDCARD = "x-ai/grok-3-beta"
    VALUE_INVESTOR = "deepseek/deepseek-chat-v3-0324"


class Settings:
    """Application settings sourced from environment variables."""

    def __init__(self) -> None:
        # Core integrations
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_db",
        )
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca_base_url = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        # Model configurations for each advisor
        self.advisor_models: Dict[str, str] = {
            "VALUE_INVESTOR": os.getenv(
                "VALUE_INVESTOR_MODEL", "anthropic/claude-sonnet-4.5"
            ),
            "MACRO_STRATEGIST": os.getenv(
                "MACRO_STRATEGIST_MODEL", "google/gemini-2.5-flash"
            ),
            "RISK_ANALYST": os.getenv(
                "RISK_ANALYST_MODEL", "anthropic/claude-sonnet-4.5"
            ),
            "WILDCARD": os.getenv("WILDCARD_MODEL", "x-ai/grok-4-fast"),
        }

        # Risk management
        self.max_daily_turnover = self._get_float("MAX_DAILY_TURNOVER", 0.20)
        self.max_weight_delta = self._get_float("MAX_WEIGHT_DELTA", 0.05)
        self.min_trade_value = self._get_float("MIN_TRADE_VALUE", 100.0)
        self.min_trade_percent = self._get_float("MIN_TRADE_PERCENT", 0.001)
        self.max_position_weight = self._get_float("MAX_POSITION_WEIGHT", 0.15)
        self.min_hold_days = self._get_int("MIN_HOLD_DAYS", 1)

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE") or None

        # Flags
        self.debug = self._get_bool("DEBUG", False)
        self.dry_run = self._get_bool("DRY_RUN", True)
        self.use_portfolio_context = self._get_bool("USE_PORTFOLIO_CONTEXT", True)
        self.fresh_start_mode = self._get_bool("FRESH_START_MODE", False)

        # Portfolio Construction
        self.target_position_count = self._get_int("TARGET_POSITION_COUNT", 20)

    @staticmethod
    def _get_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _get_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


# Global settings instance
settings = Settings()