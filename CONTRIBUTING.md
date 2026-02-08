# Contributing to ConsensusAI

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/consensus-ai.git
   cd consensus-ai
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

- **Formatting**: We use [Black](https://black.readthedocs.io/) with 88 character line length
- **Imports**: We use [isort](https://pycqa.github.io/isort/) with Black profile
- **Type Checking**: We use [mypy](https://mypy.readthedocs.io/) for static type checking

Run before committing:
```bash
black app/ tests/
isort app/ tests/
mypy app/
```

### Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Aim for high test coverage
- Test both success and error cases

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Remove)
- Be specific about what changed
- Reference issues when applicable

Examples:
```
Add support for custom advisor weights
Fix JSON parsing edge case in Gemini responses
Update README with new configuration options
```

## Areas for Contribution

### High Priority

- **Additional Advisor Types**: Add new specialized advisors (e.g., ESG, Technical Analysis)
- **Consensus Algorithm Improvements**: Enhance the rank-based fusion algorithm
- **Risk Controls**: Add new risk management features
- **Performance Analytics**: Expand metrics and reporting
- **Documentation**: Improve docs, add examples, tutorials

### Medium Priority

- **UI/Dashboard**: Web interface for monitoring and control
- **Backtesting**: Historical performance analysis
- **More Exchanges**: Support for additional trading platforms
- **Alerting**: Notifications for important events
- **Configuration UI**: Web-based configuration management

### Low Priority

- **Docker Support**: Containerization improvements
- **Kubernetes**: Helm charts and deployment configs
- **Monitoring**: Prometheus/Grafana integration
- **Internationalization**: Multi-language support

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure tests pass** (`pytest`)
4. **Run code quality checks** (black, isort, mypy)
5. **Update CHANGELOG.md** if applicable
6. **Create pull request** with clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## Code Review

- All PRs require at least one approval
- Address review comments promptly
- Be open to feedback and suggestions
- Keep PRs focused and reasonably sized

## Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Review the documentation

Thank you for contributing to ConsensusAI! 🚀


