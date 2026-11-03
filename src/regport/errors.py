"""Exceptions shared across the pipeline."""


class LookaheadError(RuntimeError):
    """A time-series consumer was handed rows available after its ``asof`` date (DQ-01)."""


class ConstraintViolation(RuntimeError):
    """Optimizer weights break a portfolio constraint beyond tolerance (FR-17, FR-24)."""


class DataSourceError(RuntimeError):
    """A data source failed. Messages never contain API keys or signed links."""
