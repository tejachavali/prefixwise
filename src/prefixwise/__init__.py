"""Prefixwise: model-agnostic ledger of avoidable agent token spend."""

__version__ = "0.10.6"

from .classifier import MissCause, classify
from .diagnostics import AgreementScorer
from .fingerprint import Fingerprinter, fingerprint_blocks
from .ledger import Ledger
from .sdk import Recorder, session, wrap

__all__ = ["__version__", "MissCause", "classify", "AgreementScorer", "Fingerprinter", "fingerprint_blocks", "Ledger",
           "Recorder", "session", "wrap", "PrefixwiseMiddleware"]


def __getattr__(name: str):
    if name == "PrefixwiseMiddleware":  # `vllm serve --middleware prefixwise.PrefixwiseMiddleware`
        from .asgi import PrefixwiseMiddleware
        return PrefixwiseMiddleware
    raise AttributeError(f"module 'prefixwise' has no attribute {name!r}")
