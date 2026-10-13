"""Model registry.

In-tree models use :func:`register`. Another repository can expose the same
protocols through the entry-point group ``volsurface.models`` without
importing this package at definition time. ``create`` checks the in-tree
registry first, then the entry points.
"""

from __future__ import annotations

from importlib.metadata import entry_points

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator: ``@register("heston")`` on a model class."""

    def decorate(cls: type) -> type:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"model {name!r} is already registered as {existing}")
        _REGISTRY[name] = cls
        return cls

    return decorate


def registered_names() -> tuple[str, ...]:
    """Names registered by import, not including unloaded entry points."""
    return tuple(sorted(_REGISTRY))


def create(name: str, **kwargs):
    """Build a model by name.

    Entry points are loaded only when ``name`` is not already in the in-tree
    registry, so importing this module does not import every plugin.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        matches = [ep for ep in entry_points(group="volsurface.models") if ep.name == name]
        if not matches:
            points = entry_points(group="volsurface.models")
            known = sorted(set(_REGISTRY) | {ep.name for ep in points})
            raise KeyError(f"unknown model {name!r}; known: {known}")
        cls = matches[0].load()
    return cls(**kwargs)
