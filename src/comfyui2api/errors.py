from __future__ import annotations


class ConfigError(RuntimeError):
    """Raised when required process configuration is missing or invalid."""


class SlugError(ValueError):
    """Raised when a slug fails the locked grammar."""


class UrlError(ValueError):
    """Raised when a ComfyUI base URL fails morphological checks."""


class DuplicateError(ValueError):
    """Raised when a unique slug or normalized URL already exists."""


class NotFoundError(LookupError):
    """Raised when an instance or external model slug is missing."""


class ConflictError(ValueError):
    """Raised when a mutation is forbidden because work is in flight."""


class ValidationError(ValueError):
    """Raised when an admin mutation fails field or enablement checks."""


class InvalidParameterError(ValueError):
    """Raised when a request parameter fails its workflow sidecar's type or range check."""
