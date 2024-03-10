try:
    # This variable is injected by setuptools-scm on build time
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
