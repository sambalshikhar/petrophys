"""Curve display-label helpers for visualization."""

from typing import Any


def is_apparent_porosity_curve(curve: Any) -> bool:
    """Return True when curve metadata says porosity is single-log apparent."""
    metadata = getattr(curve, "metadata", None)
    custom = getattr(metadata, "custom", {}) or {}
    if bool(custom.get("porosity_apparent")):
        return True

    description = str(getattr(metadata, "description", "") or "").lower()
    return "apparent" in description and "porosity" in description


def porosity_display_label(base_label: str, curve: Any) -> str:
    """Return compact PHIT/PHIE labels that expose apparent provenance."""
    label = str(base_label or getattr(curve, "mnemonic", "") or "")
    if label.upper() in {"PHIT", "PHIE"} and is_apparent_porosity_curve(curve):
        return f"{label} app"
    return label
