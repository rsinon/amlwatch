# Contributing to amlwatch

Issues and pull requests are welcome.

## Getting started

```bash
git clone https://github.com/rsinon/amlwatch.git
cd amlwatch
pip install -e ".[dev]"
pytest
ruff check amlwatch tests
```

## Pull requests

- Keep each pull request to one change, with a short explanation of what it does and why.
- Add or update tests for any change in behaviour.
- Run `pytest` and `ruff check` before opening the pull request; CI runs both.
- Use clear names and docstrings. Readability matters more than brevity here.

## Reporting issues

Please include the version, a minimal example that reproduces the problem, and what you expected to happen. Do not attach real personal or confidential data; use the built-in synthetic data generator to build examples.
