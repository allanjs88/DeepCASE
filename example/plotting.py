"""Compatibility import; plot implementations live in reporting/plotting.py."""
try:
    from .reporting.plotting import ResultsPlotter
except ImportError:
    from reporting.plotting import ResultsPlotter
