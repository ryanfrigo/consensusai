# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please email security@yourdomain.com with the following information:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue before making it public.

## Security Best Practices

When using ConsensusAI:

1. **Never commit `.env` files** - Always use `.env.example` as a template
2. **Use paper trading first** - Test thoroughly before live trading
3. **Rotate API keys regularly** - Especially if keys are exposed
4. **Use strong database passwords** - Don't use defaults in production
5. **Enable HTTPS in production** - Never expose the API over HTTP
6. **Review risk controls** - Ensure limits are appropriate for your use case
7. **Monitor logs** - Watch for unusual activity
8. **Keep dependencies updated** - Run `pip install --upgrade` regularly

## Known Security Considerations

- API keys are stored in environment variables (never hardcoded)
- Database connections use parameterized queries (SQL injection protected)
- No user authentication built-in (add your own if needed)
- CORS is permissive in debug mode (restrict in production)

## Disclosure Policy

We follow responsible disclosure practices:
- Vulnerabilities will be patched promptly
- Security updates will be released as soon as possible
- Credit will be given to reporters (if desired)
- Public disclosure will occur after a patch is available

Thank you for helping keep ConsensusAI secure!


