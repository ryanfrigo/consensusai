# Portfolio Management API - CLI Commands Reference

This document provides a comprehensive reference for all command-line interface (CLI) commands available in the Portfolio Management API project.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Development Server](#development-server)
- [Portfolio Management](#portfolio-management)
- [Testing](#testing)
- [Code Quality & Formatting](#code-quality--formatting)
- [Database Operations](#database-operations)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Deactivate virtual environment
deactivate
```

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install specific packages
pip install fastapi uvicorn sqlalchemy
```

### Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit environment variables
nano .env
# or
vim .env
```

## Development Server

### FastAPI Server

```bash
# Start development server (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start server with custom settings
uvicorn app.main:app --reload --port 8080

# Run server directly with Python
python -m app.main

# Start server in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start with specific log level
uvicorn app.main:app --reload --log-level debug
```

### API Documentation

```bash
# Access API docs (after server is running)
open http://localhost:8000/docs        # Swagger UI
open http://localhost:8000/redoc       # ReDoc
```

## Portfolio Management

### Scheduler Commands

```bash
# Run portfolio scheduler daemon
python scheduler.py

# Run portfolio rebalance immediately (test mode)
python scheduler.py --run-now

# Background scheduler with logging
nohup python scheduler.py > scheduler.log 2>&1 &
```

### Order Management

```bash
# Check recent orders and positions
python check_orders.py

# Run with virtual environment
source .venv/bin/activate && python check_orders.py
```

### Manual Portfolio Operations

```bash
# Trigger rebalance via API (requires server running)
curl -X POST "http://localhost:8000/portfolio/rebalance" \
     -H "Content-Type: application/json" \
     -d '{"force": false, "dry_run": true}'

# Get current portfolio status
curl "http://localhost:8000/portfolio/current"

# Get portfolio performance metrics
curl "http://localhost:8000/portfolio/performance"

# Sync positions from Alpaca
curl -X POST "http://localhost:8000/portfolio/sync-positions"
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py

# Run specific test function
pytest tests/test_api.py::test_root_endpoint

# Run tests with coverage
pytest --cov=app tests/

# Run tests in parallel
pytest -n auto

# Run only fast tests (exclude slow markers)
pytest -m "not slow"
```

### Test Categories

```bash
# Run full workflow end-to-end test
python tests/test_full_workflow.py

# Test single LLM advisor
python tests/test_single_advisor.py

# Test Alpaca service integration
python tests/test_alpaca_service.py

# Test LLM service
python tests/test_llm_service.py
```

### Test Configuration

```bash
# Run with specific pytest configuration
pytest --asyncio-mode=auto -v --tb=short

# Run with custom test database
DATABASE_URL="sqlite:///test.db" pytest

# Run tests with environment override
DRY_RUN=true pytest tests/
```

## Code Quality & Formatting

### Code Formatting

```bash
# Format Python code with Black
black app/ tests/

# Format specific files
black app/main.py app/models.py

# Check formatting without making changes
black --check app/ tests/

# Format with specific line length
black --line-length 88 app/
```

### Import Sorting

```bash
# Sort imports with isort
isort app/ tests/

# Check import sorting
isort --check-only app/ tests/

# Sort imports with Black compatibility
isort --profile black app/ tests/
```

### Type Checking

```bash
# Run mypy type checking
mypy app/

# Type check specific files
mypy app/main.py app/services/

# Run with strict mode
mypy --strict app/

# Generate type checking report
mypy app/ --html-report mypy-report/
```

### Code Quality Tools

```bash
# Run all quality checks together
black app/ tests/ && isort app/ tests/ && mypy app/

# Check code quality without modifications
black --check app/ && isort --check-only app/ && mypy app/
```

## Database Operations

### Database Management

```bash
# Initialize database (tables created automatically)
python -c "from app.database import init_db; import asyncio; asyncio.run(init_db())"

# Reset database (careful - destroys data!)
python -c "from app.database import reset_db; import asyncio; asyncio.run(reset_db())"
```

### Database Utilities

```bash
# Connect to PostgreSQL database
psql $DATABASE_URL

# Backup database
pg_dump $DATABASE_URL > backup.sql

# Restore database
psql $DATABASE_URL < backup.sql
```

## Deployment

### Production Server

```bash
# Start production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start with Gunicorn (production ASGI server)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Start with systemd service
sudo systemctl start portfolio-api
sudo systemctl enable portfolio-api
```

### Docker Commands

```bash
# Build Docker image
docker build -t portfolio-api .

# Run container
docker run -p 8000:8000 portfolio-api

# Run with environment file
docker run --env-file .env -p 8000:8000 portfolio-api

# Docker Compose
docker-compose up -d
docker-compose logs -f portfolio-api
docker-compose down
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health/

# Database health check
curl http://localhost:8000/health/database

# Check application status
curl http://localhost:8000/
```

## Troubleshooting

### Process Management

```bash
# Check running Python processes
ps aux | grep python

# Find processes using port 8000
lsof -i :8000

# Kill process by PID
kill <PID>

# Kill all uvicorn processes
pkill -f uvicorn
```

### Logs and Debugging

```bash
# View application logs
tail -f portfolio_scheduler.log

# Check system logs
journalctl -u portfolio-api -f

# Debug mode startup
DEBUG=true uvicorn app.main:app --reload
```

### Environment Validation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Verify environment variables
python -c "from app.config import settings; print(f'Database: {settings.database_url}')"

# Test API connectivity
python -c "
import asyncio
from app.services.alpaca import AlpacaService
async def test():
    service = AlpacaService()
    nav = await service.get_account_nav()
    print(f'Account NAV: ${nav:,.2f}')
asyncio.run(test())
"
```

### Common Issue Resolution

```bash
# Fix import issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Regenerate requirements
pip freeze > requirements.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# Reset virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Reference

### Most Common Commands

```bash
# Development workflow
source .venv/bin/activate
uvicorn app.main:app --reload

# Code quality check
black app/ tests/ && isort app/ tests/ && mypy app/ && pytest

# Manual rebalance trigger
python scheduler.py --run-now

# Check orders and positions
python check_orders.py

# Run full test suite
pytest -v
```

### Environment Variables Quick Check

```bash
# Verify all required environment variables are set
python -c "
from app.config import settings
required = ['database_url', 'alpaca_api_key', 'alpaca_secret_key', 'openrouter_api_key']
for var in required:
    value = getattr(settings, var, None)
    status = '✅' if value else '❌'
    print(f'{status} {var.upper()}: {\"Set\" if value else \"Missing\"}')
"
```

---

## Notes

- Always activate the virtual environment (`.venv`) before running commands
- Use `dry_run=true` in environment for safe testing
- Paper trading is recommended for development (`ALPACA_BASE_URL=paper-api.alpaca.markets`)
- Check logs regularly for error debugging
- Ensure all API keys are properly configured before running portfolio operations

For more information, see the [README.md](README.md) file. 