"""Helpers for legacy import namespaces.

Compatibility packages call :func:`reexport_package` when explicitly imported.
Canonical library imports never depend on this module.
"""

from importlib import import_module
import sys
import warnings


def reexport_package(namespace, target: str, aliases=None) -> None:
    """Expose a canonical package through a deprecated package namespace."""
    legacy_name = namespace["__name__"]
    target_module = import_module(target)
    public_names = list(getattr(target_module, "__all__", ()))

    for name in public_names:
        namespace[name] = getattr(target_module, name)
    namespace["__all__"] = public_names

    for suffix, module_name in (aliases or {}).items():
        sys.modules[f"{legacy_name}.{suffix}"] = import_module(module_name)

    warnings.warn(
        f"{legacy_name} is deprecated; import from {target} instead",
        DeprecationWarning,
        stacklevel=2,
    )
