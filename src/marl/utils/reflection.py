"""
General-purpose reflection utilities for type hints and class hierarchies.

These helpers are used by both :mod:`marl.utils.serialization` (subclass
lookup for discriminator-based deserialization) and :mod:`marl.utils.tuning`
(abstract-type detection for Optuna search-space inference).

Public API
----------
unwrap_optional        Strip ``None`` from ``X | None`` → ``X``.
is_abstract            Detect abstract classes (handles non-ABCMeta pattern).
get_concrete_subclasses  Collect every non-abstract subclass recursively.
get_subclass_from_name   Find one subclass by ``__name__`` (depth-first).
get_subclass_map         Build a ``{name: cls}`` dict for a whole hierarchy.
"""

import inspect
from types import NoneType, UnionType
from typing import Any, Literal, Type, Union, get_args, get_origin


def unwrap_optional(hint: Any) -> Any:
    """
    Strip ``None`` from an optional type annotation and return the inner type.

    Handles both the modern ``X | None`` (:class:`types.UnionType`, Python
    3.10+) and the legacy ``Optional[X]`` / ``Union[X, None]`` spelling from
    the :mod:`typing` module.

    Returns *hint* unchanged when:

    * The annotation is not a union at all (e.g. a plain ``float``).
    * The union has more than one non-``None`` member (e.g. ``X | Y | None``).

    Examples::

        >>> unwrap_optional(int | None)      # → int
        >>> unwrap_optional(Optional[str])   # → str
        >>> unwrap_optional(float)           # → float  (unchanged)
        >>> unwrap_optional(X | Y | None)    # → X | Y | None  (unchanged)
    """
    origin = get_origin(hint)
    # typing.Union[X, None] / typing.Optional[X] / X | None
    # Guard on origin is Union so that arbitrary generics like List[int] or
    # EnvConfig[E] (which have a non-Union, non-None origin) are never
    # mistakenly unwrapped.
    if origin is Union:
        args = [a for a in get_args(hint) if a is not NoneType]
        if len(args) == 1:
            return args[0]
    # Fallback for native X | None (types.UnionType) on older Python builds
    # where get_origin may return UnionType instead of Union.
    if isinstance(hint, UnionType):
        args = [a for a in get_args(hint) if a is not NoneType]
        if len(args) == 1:
            return args[0]
    return hint


def is_abstract(cls: type) -> bool:
    """
    Return ``True`` if *cls* has any unimplemented abstract methods.

    Two distinct patterns are handled:

    **Standard ABCMeta path** — classes that inherit from :class:`abc.ABC`
    or use :class:`abc.ABCMeta` as their metaclass.  Python populates
    ``cls.__abstractmethods__`` automatically, so
    :func:`inspect.isabstract` already returns ``True`` for them.

    **Bare ``@abstractmethod`` without ABCMeta** — a pattern common in this
    codebase.  In that case ``__abstractmethods__`` is *never* populated by
    the metaclass (because ``ABCMeta.__new__`` is not involved), so
    :func:`inspect.isabstract` incorrectly returns ``False``.  This function
    additionally scans :func:`dir` for any attribute whose
    ``__isabstractmethod__`` flag is ``True``, catching these classes.
    """
    if inspect.isabstract(cls):
        return True
    return any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))


def get_concrete_subclasses(cls: type) -> list[type]:
    """
    Recursively collect every non-abstract subclass of *cls*.

    Uses :func:`is_abstract` to filter candidates, so both the standard
    :class:`abc.ABC` pattern and bare ``@abstractmethod`` usage are handled
    correctly.  Only subclasses that have already been *imported* (and thus
    registered in Python's internal ``__subclasses__`` tracking) will appear
    in the result.

    *cls* itself is never included even when it is concrete.

    Args:
        cls: Root class to start the search from.

    Returns:
        Flat list of concrete subclasses in depth-first order.
    """
    result: list[type] = []
    for sub in cls.__subclasses__():
        if not is_abstract(sub):
            result.append(sub)
        result.extend(get_concrete_subclasses(sub))
    return result


def get_subclass_from_name(base_class: Type, class_name: str) -> Type | None:
    """
    Return the first subclass of *base_class* whose ``__name__`` matches
    *class_name*, searching the whole subclass tree depth-first.

    Returns ``None`` if no match is found.

    .. note::
        *base_class* itself is **not** considered a candidate, even when its
        name matches *class_name*.
    """
    for subclass in base_class.__subclasses__():
        if subclass.__name__ == class_name:
            return subclass
        result = get_subclass_from_name(subclass, class_name)
        if result is not None:
            return result
    return None


def get_subclass_map(base_class: Type) -> dict[str, Type]:
    """
    Build a ``{class_name: class}`` mapping for *base_class* and every
    subclass in its hierarchy, recursively.

    Unlike :func:`get_subclass_from_name`, *base_class* **is** included in
    the returned mapping under its own ``__name__``.
    """
    mapping: dict[str, Type] = {base_class.__name__: base_class}
    for subclass in base_class.__subclasses__():
        mapping[subclass.__name__] = subclass
        mapping.update(get_subclass_map(subclass))
    return mapping
