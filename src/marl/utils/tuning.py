"""
Optuna hyperparameter tuning for Serializable dataclasses.

The two public symbols are:

* :func:`tuning` — attaches search-space metadata to a dataclass field via
  ``dataclasses.field(metadata=tuning(...))``.
* :func:`suggest` — standalone function that recursively constructs a
  dataclass instance whose fields are suggested by an Optuna trial.

Typical usage::

    from dataclasses import dataclass, field
    from marl.utils import suggest, tuning

    @dataclass
    class MyTrainer(Trainer):
        lr: float = field(default=1e-4, metadata=tuning(1e-5, 1e-2, log=True))
        batch_size: int = field(default=64, metadata=tuning(16, 256))
        # Literal and bool fields are auto-detected — no annotation needed:
        optimiser: Literal["adam", "rmsprop"] = "adam"
        double_q: bool = True

    def objective(trial: optuna.Trial) -> float:
        trainer = suggest(
            MyTrainer, trial,
            qnetwork=MLP.from_env(env),   # required override
        )
        ...
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import MISSING, dataclass, fields
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from optuna import Trial

TUNE_KEY = "tune"
T = TypeVar("T")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal spec
# ---------------------------------------------------------------------------


@dataclass
class _TuneSpec:
    """Tuning specification stored in a field's metadata by :func:`tuning`."""

    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | int | None = None
    choices: list | None = None


# ---------------------------------------------------------------------------
# Public annotation helper
# ---------------------------------------------------------------------------


def tuning(
    low: float | None = None,
    high: float | None = None,
    *,
    log: bool = False,
    step: float | int | None = None,
    choices: list | None = None,
) -> dict[str, Any]:
    """
    Define the search space for parameter tuning with Optuna. The output of this functions
    is meant to be stored in the `metadata` argument of a dataclass field:
    ```py
    lr: float = field(metadata=tuning(1e-5, 1e-2, log=True))
    batch_size: int = field(default=64, metadata=tuning(16, 256))
    ```

    Two modes:

    **Numeric range** — ``tuning(low, high, log=False, step=None)``
        Both ``low`` and ``high`` are required.  The field's resolved type
        annotation (``int`` or ``float``) determines which Optuna suggest
        method is called.  Use ``log=True`` for parameters that vary over
        orders of magnitude (learning rates, etc.).  ``step`` is optional and
        mutually exclusive with ``log=True``.

    **Categorical / polymorphic** — ``tuning(choices=[...])``
        If the list elements are **types** (classes), :func:`suggest` will
        suggest the class name categorically and then recursively suggest the
        chosen class's own fields.  Use this to restrict which concrete
        subclasses are searched when the field is typed as an abstract base
        class (by default :func:`suggest` auto-collects *all* concrete
        subclasses, which may be too broad).

        If the elements are **plain values**, ``trial.suggest_categorical`` is
        used directly.

    Args:
        low: Lower bound (inclusive) for numeric suggestions.
        high: Upper bound (inclusive) for numeric suggestions.
        log: Sample on a log scale.  Requires ``low > 0``.
        step: Discretisation step.  Mutually exclusive with ``log=True``.
        choices: List of allowed values or types.  Mutually exclusive with
            ``low`` / ``high``.

    Raises:
        ValueError: If neither ``choices`` nor both ``low`` and ``high`` are
            provided, or if both are provided simultaneously.
    """
    if choices is not None and (low is not None or high is not None):
        raise ValueError("tuning(): cannot specify both choices and low/high bounds.")
    if choices is None and (low is None or high is None):
        raise ValueError(
            "tuning(): must specify either choices=[...] or both low and high bounds. "
            f"Got: low={low!r}, high={high!r}, choices={choices!r}."
        )
    return {TUNE_KEY: _TuneSpec(low=low, high=high, log=log, step=step, choices=choices)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_abstract(cls: type) -> bool:
    """
    Return True if *cls* has any unimplemented abstract methods.

    Covers two cases:
    * Classes that inherit from ``abc.ABC`` / use ``ABCMeta`` (the standard
      path checked by ``inspect.isabstract``).
    * Classes that use ``@abstractmethod`` *without* ``ABCMeta`` — common in
      this codebase.  In that case ``__abstractmethods__`` is never populated
      by the metaclass, but the decorator still sets
      ``method.__isabstractmethod__ = True`` on the function object, which we
      can detect via ``dir``.
    """
    if inspect.isabstract(cls):
        return True
    return any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))


def _get_concrete_subclasses(cls: type) -> list[type]:
    """Recursively collect every non-abstract subclass of *cls*."""
    result: list[type] = []
    for sub in cls.__subclasses__():
        if not _is_abstract(sub):
            result.append(sub)
        result.extend(_get_concrete_subclasses(sub))
    return result


def _unwrap_optional(hint: Any) -> Any:
    """Unwrap ``X | None`` → ``X``.  Returns *hint* unchanged otherwise."""
    origin = get_origin(hint)
    # typing.Union[X, None] / typing.Optional[X]
    if origin is not None and origin is not Literal:
        args = [a for a in get_args(hint) if a is not NoneType]
        if len(args) == 1:
            return args[0]
    # X | None  (types.UnionType, Python 3.10+)
    if isinstance(hint, UnionType):
        args = [a for a in get_args(hint) if a is not NoneType]
        if len(args) == 1:
            return args[0]
    return hint


# ---------------------------------------------------------------------------
# Public suggest function
# ---------------------------------------------------------------------------


def suggest(cls: type[T], trial: "Trial", *, prefix: str = "", **overrides: Any) -> T:
    """
    Suggest hyperparameters for a dataclass using an Optuna trial.

    Works on any :class:`~marl.utils.Serializable` dataclass.  Each
    ``init=True`` field is handled according to the following rules, applied
    in strict priority order:

    1. **Override** — if the field name is present in ``**overrides``, use
       that value verbatim and skip all further rules.
    2. **tuning(choices=[TypeA, TypeB])** *(elements are types)* — suggest the
       class name categorically, then recurse into the winner via
       ``suggest(chosen_cls, trial, prefix=...)``.
    3. **tuning(choices=["a", "b"])** *(elements are plain values)* —
       ``trial.suggest_categorical``.
    4. **tuning(low, high)** on a ``float`` field — ``trial.suggest_float``.
    5. **tuning(low, high)** on an ``int`` field — ``trial.suggest_int``.
    6. **bool** field *(no annotation)* —
       ``trial.suggest_categorical([True, False])``.
    7. **Literal["a", "b"]** field *(no annotation)* —
       ``trial.suggest_categorical`` with the literal values.
    8. **Concrete, non-** ``torch.nn.Module`` **Serializable** field *(no
       annotation)* — recurse: ``suggest(field_type, trial, prefix=...)``.
    9. **Abstract, non-** ``torch.nn.Module`` **Serializable** field *(no
       annotation)* — auto-collect all concrete subclasses (recursively via
       ``__subclasses__``), suggest the class name categorically, recurse
       into the winner.  Use ``tuning(choices=[...])`` on the field to
       restrict which subclasses are searched.
    10. **float / int field with a default** *(no annotation, not handled
        above)* — use the default value and emit a ``WARNING`` so the user
        knows the field is not being tuned.
    11. **Any other field with a default** *(not handled above)* — use the
        default silently.
    12. **Required field with no default** — raise ``ValueError``.

    Parameter names registered in the trial are dot-separated so that nested
    suggestions remain readable in the Optuna dashboard.  For example, a
    field ``tau`` inside a field ``target_updater`` is registered as
    ``target_updater.tau``.

    .. note::
        ``torch.nn.Module`` subclasses (including all ``NN`` subclasses) are
        excluded from rules 8 and 9.  Their fields typically depend on
        environment shape and cannot be suggested without external information.
        Pass them explicitly via ``**overrides``.

    Args:
        cls: The dataclass type to instantiate.
        trial: The current Optuna trial.
        prefix: Dot-separated prefix prepended to every parameter name.
            Used internally during recursion; do not set this manually.
        **overrides: Field values to use verbatim, bypassing suggestion.
            Required for any field whose construction depends on information
            not available from the dataclass definition alone (e.g. a
            Q-network that needs environment shape).

    Returns:
        A fully-constructed instance of *cls*.

    Raises:
        ValueError: If a required field has no default and is not covered by
            a ``tuning()`` annotation, an auto-detection rule, or an override.

    Example::
    ```py
    from marl.utils import suggest, tuning

    def objective(trial: optuna.Trial) -> float:
        trainer = suggest(
            DQN, trial,
            qnetwork=MLP.from_env(env),
            memory=suggest(TransitionMemory, trial),
            mixer=None,
            train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
        )
        run_experiment(trainer)
        return trainer.best_return
    ```
    """
    import torch

    from marl.utils.serialization import Serializable

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    init_kwargs: dict[str, Any] = {}

    for f in fields(cls):  # type: ignore[arg-type]
        if not f.init:
            continue

        name = f.name
        full_name = f"{prefix}.{name}" if prefix else name
        hint = hints.get(name)
        spec: _TuneSpec | None = f.metadata.get(TUNE_KEY)

        # Resolve default (call factory eagerly if present)
        if f.default is not MISSING:
            default = f.default
            has_default = True
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
            has_default = True
        else:
            default = MISSING
            has_default = False

        # ------------------------------------------------------------------
        # Rule 1 — explicit override
        # ------------------------------------------------------------------
        if name in overrides:
            init_kwargs[name] = overrides[name]
            continue

        # Unwrap Optional[X] / X | None for all subsequent checks
        resolved = _unwrap_optional(hint) if hint is not None else None

        # ------------------------------------------------------------------
        # Rules 2 & 3 — tuning(choices=[...])
        # ------------------------------------------------------------------
        if spec is not None and spec.choices is not None:
            if all(isinstance(c, type) for c in spec.choices):
                # Rule 2: choices are types → categorical class selection + recurse
                type_name = trial.suggest_categorical(
                    f"{full_name}.__type__",
                    [c.__name__ for c in spec.choices],
                )
                chosen_cls = next(c for c in spec.choices if c.__name__ == type_name)
                init_kwargs[name] = suggest(chosen_cls, trial, prefix=full_name)
            else:
                # Rule 3: choices are plain values → direct categorical
                init_kwargs[name] = trial.suggest_categorical(full_name, spec.choices)
            continue

        # ------------------------------------------------------------------
        # Rules 4 & 5 — tuning(low, high)
        # ------------------------------------------------------------------
        if spec is not None and spec.low is not None and spec.high is not None:
            if resolved is int:
                # Rule 5
                step = int(spec.step) if spec.step is not None else 1
                init_kwargs[name] = trial.suggest_int(full_name, int(spec.low), int(spec.high), log=spec.log, step=step)
            else:
                # Rule 4 (float, or any numeric type not explicitly int)
                init_kwargs[name] = trial.suggest_float(full_name, spec.low, spec.high, log=spec.log, step=spec.step)
            continue

        # ------------------------------------------------------------------
        # Rule 6 — bool
        # ------------------------------------------------------------------
        if resolved is bool:
            init_kwargs[name] = trial.suggest_categorical(full_name, [True, False])
            continue

        # ------------------------------------------------------------------
        # Rule 7 — Literal[...]
        # ------------------------------------------------------------------
        if get_origin(resolved) is Literal:
            init_kwargs[name] = trial.suggest_categorical(full_name, list(get_args(resolved)))
            continue

        # ------------------------------------------------------------------
        # Rules 8 & 9 — Serializable subclass (concrete or abstract)
        # ------------------------------------------------------------------
        if (
            resolved is not None
            and isinstance(resolved, type)
            and issubclass(resolved, Serializable)
            and not issubclass(resolved, torch.nn.Module)
        ):
            if not _is_abstract(resolved):
                # Rule 8: concrete Serializable → recurse directly
                init_kwargs[name] = suggest(resolved, trial, prefix=full_name)
                continue
            else:
                # Rule 9: abstract Serializable → auto-collect concrete subclasses
                candidates = _get_concrete_subclasses(resolved)
                if candidates:
                    type_name = trial.suggest_categorical(
                        f"{full_name}.__type__",
                        [c.__name__ for c in candidates],
                    )
                    chosen_cls = next(c for c in candidates if c.__name__ == type_name)
                    init_kwargs[name] = suggest(chosen_cls, trial, prefix=full_name)
                    continue
                # No concrete subclasses found — fall through to default / raise

        # ------------------------------------------------------------------
        # Rule 10 — float/int with default: use it, but warn
        # ------------------------------------------------------------------
        if has_default and default is not None and resolved in (int, float):
            logger.warning(
                "Field '%s' (%s) in %s has no tuning() annotation — "
                "using default value %r. Add tuning(low, high) to include it in the search.",
                full_name,
                "float" if resolved is float else "int",
                cls.__name__,
                default,
            )
            init_kwargs[name] = default
            continue

        # ------------------------------------------------------------------
        # Rule 11 — any other field with a default: use it silently
        # ------------------------------------------------------------------
        if has_default:
            init_kwargs[name] = default
            continue

        # ------------------------------------------------------------------
        # Rule 12 — required field with no default: raise
        # ------------------------------------------------------------------
        raise ValueError(
            f"Cannot suggest a value for required field '{full_name}' "
            f"(type: {hint!r}) in {cls.__name__}. "
            "Either add tuning() metadata to the field, provide a default value, "
            "or pass it as a keyword-argument override to suggest()."
        )

    return cls(**init_kwargs)
