"""Trade-capital bookkeeping and control package.

This package provides a self-contained bookkeeping engine that decides whether
a proposed trade can be executed against the currently available capital, while
keeping a complete, auditable ledger.

It is intentionally decoupled from the forecasting / prediction pipeline:
nothing in this package imports the forecasting modules, and the forecasting
modules do not need to import this package.

Public surface
--------------
* :class:`~app.bookkeeping.bookkeeping.BookkeepingEngine` -- the decision engine.
* :func:`~app.bookkeeping.bookkeeping_config.load_config` -- env-driven config.
* :mod:`~app.bookkeeping.bookkeeping_cli` -- CLI / JSON interface for Claude.
"""

from app.bookkeeping.bookkeeping import BookkeepingEngine
from app.bookkeeping.bookkeeping_config import BookkeepingConfig, load_config

__all__ = ["BookkeepingEngine", "BookkeepingConfig", "load_config"]
