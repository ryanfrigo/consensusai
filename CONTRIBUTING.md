# Contributing to ConsensusAI

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/ryanfrigo/consensusai.git
cd consensusai
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                        # all tests
pytest tests/test_api.py -v   # specific file
pytest --cov=app tests/       # with coverage
```

## Code Style

```bash
black app/ tests/
isort app/ tests/
mypy app/
```

## Pull Requests

1. Fork → branch → make changes → run tests → PR
2. Keep PRs focused — one feature or fix per PR
3. Write tests for new functionality
4. Follow existing code style (Black, isort)

## Ideas for Contributions

- Additional advisor personas
- Improved consensus algorithms
- More risk control mechanisms
- Better test coverage
- Documentation improvements
