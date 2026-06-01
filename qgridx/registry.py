"""Component registry and factory for qGridX.

Any stage of the pipeline can be swapped by registering a new class:

    @register_component("quantum_master", "my_qaoa")
    class MyQAOA(QuantumMasterBase):
        ...

Then set ``quantum_master.name: my_qaoa`` in your YAML config.
No changes to pipeline code are needed.
"""
from __future__ import annotations

from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")

# _REGISTRY maps stage_name -> {component_name -> class}
_REGISTRY: dict[str, dict[str, type]] = {}


def register_component(stage: str, name: str) -> Callable[[type], type]:
    """Class decorator that registers *cls* under *stage* / *name*.

    Args:
        stage: Pipeline stage identifier (e.g. ``"quantum_master"``).
        name:  Component name used in the YAML config (e.g. ``"pce_gqe"``).
    """
    def decorator(cls: type) -> type:
        _REGISTRY.setdefault(stage, {})[name] = cls
        return cls
    return decorator


def get_component(stage: str, name: str) -> type:
    """Retrieve a registered component class.

    Args:
        stage: Pipeline stage identifier.
        name:  Component name.

    Returns:
        The registered class.

    Raises:
        KeyError: If the stage or name is not registered.
    """
    if stage not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown pipeline stage '{stage}'. Available stages: {available}"
        )
    if name not in _REGISTRY[stage]:
        available = list(_REGISTRY[stage].keys())
        raise KeyError(
            f"Unknown component '{name}' for stage '{stage}'. "
            f"Available components: {available}"
        )
    return _REGISTRY[stage][name]


def build_component(stage: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a registered component.

    Args:
        stage:  Pipeline stage identifier.
        name:   Component name.
        *args:  Positional arguments forwarded to the component constructor.
        **kwargs: Keyword arguments forwarded to the component constructor.

    Returns:
        Instantiated component.
    """
    cls = get_component(stage, name)
    return cls(*args, **kwargs)


def list_components(stage: str) -> list[str]:
    """Return names of all registered components for a stage."""
    return list(_REGISTRY.get(stage, {}).keys())


def list_stages() -> list[str]:
    """Return names of all registered stages."""
    return list(_REGISTRY.keys())
