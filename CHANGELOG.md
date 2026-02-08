# Changelog

All notable changes to ConsensusAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Multi-LLM advisor system (4 advisors)
- Mathematical consensus engine with rank-based fusion
- Alpaca trading integration
- Portfolio-aware recommendations
- Comprehensive risk controls
- FastAPI REST API
- PostgreSQL database integration
- Robust JSON parsing for LLM responses
- Daily scheduler for automated rebalancing
- Performance tracking and analytics

### Changed
- Configuration now uses environment variables exclusively
- All secrets moved to `.env` file

### Security
- Removed all hardcoded credentials
- Added `env.example` template
- Enhanced `.gitignore` for sensitive files

## [1.0.0] - 2025-01-XX

### Added
- Initial release of ConsensusAI
- Support for 4 LLM advisors (Value, Macro, Risk, Wildcard)
- Consensus engine with exponential rank weighting
- Alpaca paper and live trading support
- Portfolio context awareness
- Risk management controls
- Comprehensive test suite
- API documentation
- CLI tools for order checking and scheduling

[Unreleased]: https://github.com/yourusername/consensus-ai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/consensus-ai/releases/tag/v1.0.0


