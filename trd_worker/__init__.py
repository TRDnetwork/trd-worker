"""TRD Compute Network — worker CLI."""
import warnings as _warnings

# Silence macOS-specific urllib3 LibreSSL warning. urllib3 v2 prefers OpenSSL
# but works fine on LibreSSL for our HTTP-only use case (no fancy TLS features).
try:
    from urllib3.exceptions import NotOpenSSLWarning as _NotOpenSSLWarning
    _warnings.filterwarnings("ignore", category=_NotOpenSSLWarning)
except ImportError:
    pass

__version__ = '0.2.5'
