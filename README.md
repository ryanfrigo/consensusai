<div align="center">

# 🧠 ConsensusAI

**Multi-LLM Portfolio Management with Mathematical Consensus Engine**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*What if 4 AI advisors could vote on your portfolio? This is ConsensusAI.*

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [API Docs](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 What is ConsensusAI?

ConsensusAI is an **open-source portfolio management system** that uses **4 specialized LLM advisors** (Value Investor, Macro Strategist, Risk Analyst, and Wildcard) to generate investment recommendations, then mathematically fuses their opinions into a single consensus portfolio using a sophisticated rank-based algorithm.

Think of it as an **AI investment committee** where each advisor brings a different perspective, and ConsensusAI finds the optimal blend.

### Why ConsensusAI?

- 🧠 **Multi-Model Intelligence**: Leverages Claude, Gemini, Grok, and DeepSeek simultaneously
- 📊 **Mathematical Consensus**: Advanced rank-based fusion algorithm (not just averaging)
- 🛡️ **Built-in Risk Controls**: Turnover caps, position limits, weight deltas
- 📈 **Portfolio-Aware**: Advisors see your current holdings and performance
- 🔄 **Real Trading**: Direct Alpaca integration (paper or live)
- 🎯 **Production-Ready**: FastAPI, async/await, comprehensive tests

---

## ✨ Features

### 🤖 Four Specialized Advisors

| Advisor | Model | Focus |
|---------|-------|-------|
| **Value Investor** | Claude Sonnet 4.5 | Deep value, margin of safety, FCF yields |
| **Macro Strategist** | Gemini 2.5 Flash | Economic cycles, sector rotation, rates |
| **Risk Analyst** | Claude Sonnet 4.5 | Governance, balance sheets, downside protection |
| **Wildcard** | Grok 4 Fast | Contrarian picks, under-the-radar opportunities |

### 🧮 Consensus Engine

The system uses a **mathematical rank-based fusion algorithm**:

1. Each advisor ranks their 10 picks (by allocation %)
2. Ranks converted to exponential weights: `exp(-λ × (rank - 1))`
3. Cross-advisor performance weights applied
4. Scores fused: `S_t = Σ W_i × w_i,t`
5. Winsorization and guardrails applied
6. Final portfolio weights normalized

**Result**: A portfolio that reflects the collective wisdom of all advisors, not just a simple average.

### 🛡️ Risk Management

- **Position Limits**: Max 15% per position (configurable)
- **Turnover Caps**: Max 20% daily turnover
- **Weight Deltas**: Max 5% change per position per day
- **Minimum Trade Size**: $100 minimum
- **Hold Periods**: Configurable minimum holding periods

### 📊 Portfolio Intelligence

- **Context-Aware Recommendations**: Advisors see current holdings, P&L, cash
- **Fresh Start Mode**: Option to ignore current portfolio for major overhauls
- **Performance Tracking**: Built-in metrics and analytics
- **Decision Audit Trail**: Complete logging of all decisions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database
- [Alpaca](https://alpaca.markets) account (paper trading recommended)
- [OpenRouter](https://openrouter.ai) API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/consensus-ai.git
cd consensus-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# Or with dev tools:
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your API keys
nano .env
```

**Required variables:**
```bash
OPENROUTER_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/portfolio_db
```

### Run the Server

```bash
# Start development server
uvicorn app.main:app --reload

# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Trigger Your First Rebalance

```bash
# Via API
curl -X POST "http://localhost:8000/portfolio/rebalance" \
     -H "Content-Type: application/json" \
     -d '{"dry_run": true, "fresh_start": false}'

# Or use the scheduler
python scheduler.py --run-now
```

---

## 🧠 How It Works

### 1. Advisor Consultation

Each of the 4 advisors receives a specialized prompt and generates 10 stock recommendations with allocations, confidence scores, and justifications.

### 2. Consensus Building

The consensus engine:
- Converts allocations to ranks (higher allocation = better rank)
- Applies exponential decay: `weight = exp(-0.25 × (rank - 1))`
- Multiplies by advisor confidence scores
- Applies cross-advisor performance weights
- Fuses scores across all advisors

### 3. Risk Controls

The system applies multiple layers of protection:
- Single-name caps (15% max)
- Turnover limits (20% daily max)
- Minimum position sizes
- Weight delta restrictions

### 4. Order Execution

Trades are calculated and submitted to Alpaca (or simulated in dry-run mode).

---

## 📡 API Documentation

### Portfolio Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/current` | GET | Current portfolio summary |
| `/portfolio/rebalance` | POST | Trigger rebalancing |
| `/portfolio/history` | GET | Decision history |
| `/portfolio/targets` | GET | Portfolio targets |
| `/portfolio/recommendations` | GET | LLM recommendations |
| `/portfolio/orders` | GET | Order execution history |
| `/portfolio/performance` | GET | Performance metrics |
| `/portfolio/sync-positions` | POST | Sync positions from Alpaca |

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/` | GET | Basic health check |
| `/health/database` | GET | Database connectivity |

**Interactive API docs**: Visit `http://localhost:8000/docs` when the server is running.

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | _required_ |
| `ALPACA_API_KEY` | Alpaca API key | _required_ |
| `ALPACA_SECRET_KEY` | Alpaca secret key | _required_ |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `DRY_RUN` | Skip placing real orders | `true` |
| `USE_PORTFOLIO_CONTEXT` | Include holdings in prompts | `true` |
| `FRESH_START_MODE` | Force context-free prompts | `false` |
| `MAX_POSITION_WEIGHT` | Maximum position size | `0.15` (15%) |
| `MAX_DAILY_TURNOVER` | Daily turnover cap | `0.20` (20%) |

See `env.example` for all available options.

---

## 🏗️ Architecture

```
┌─────────────────┐
│  FastAPI App    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ LLM   │ │Alpaca │
│Service│ │Service│
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
┌────────▼────────┐
│ Consensus Engine│
│ (Rank Fusion)   │
└────────┬────────┘
         │
┌────────▼────────┐
│ Risk Controls   │
└────────┬────────┘
         │
┌────────▼────────┐
│ Order Execution │
└─────────────────┘
```

### Core Components

- **FastAPI Application**: REST API with async/await
- **PostgreSQL Database**: Persistent storage
- **LLM Service**: Multi-model OpenRouter integration
- **Consensus Engine**: Mathematical rank-based fusion
- **Alpaca Service**: Trading execution
- **Risk Controls**: Multi-layer protection

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test file
pytest tests/test_full_workflow.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Type checking
mypy app/
```

### Project Structure

```
consensus-ai/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── models.py            # SQLAlchemy & Pydantic models
│   ├── tasks.py             # Orchestration
│   ├── routers/             # API endpoints
│   └── services/            # Business logic
│       ├── llm.py           # LLM advisor service
│       ├── alpaca.py        # Trading integration
│       ├── rebalance.py     # Rebalancing logic
│       └── json_parser.py    # Robust JSON parsing
├── consensus_engine.py       # Mathematical consensus
├── scheduler.py             # Daily scheduler
├── tests/                   # Test suite
├── env.example              # Environment template
└── README.md                # This file
```

---

## 📈 Use Cases

- **Automated Portfolio Management**: Let AI advisors manage your portfolio
- **Research Tool**: Compare recommendations from different AI perspectives
- **Backtesting**: Test consensus strategies on historical data
- **Learning**: Understand how different investment philosophies combine
- **Trading Bot**: Fully automated trading with risk controls

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (`pytest`)
5. **Commit** (`git commit -m 'Add amazing feature'`)
6. **Push** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Ideas for Contributions

- Additional advisor types
- Consensus algorithm improvements
- More risk control mechanisms
- Performance analytics enhancements
- Documentation improvements
- Test coverage expansion

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always use paper trading first and never invest more than you can afford to lose.**

---

## 🙏 Acknowledgments

- [Alpaca](https://alpaca.markets) for trading API
- [OpenRouter](https://openrouter.ai) for LLM access
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
- The open-source community

---

<div align="center">

**Made with ❤️ by the ConsensusAI community**

[⭐ Star this repo](https://github.com/yourusername/consensus-ai) • [🐛 Report Bug](https://github.com/yourusername/consensus-ai/issues) • [💡 Request Feature](https://github.com/yourusername/consensus-ai/issues)

</div>
