"""amlwatch: open-source anti-money-laundering transaction monitoring."""

__version__ = "0.1.0"

from .config import RuleConfig
from .monitor import Monitor, MonitoringResult
from .rules import RULES, Alert
from .schema import SchemaError
from .synth import TYPOLOGIES, SynthConfig, generate_transactions

__all__ = [
    "RULES",
    "TYPOLOGIES",
    "Alert",
    "Monitor",
    "MonitoringResult",
    "RuleConfig",
    "SchemaError",
    "SynthConfig",
    "__version__",
    "generate_transactions",
]
