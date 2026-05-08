"""
Tests for the tuning() annotation helper and its supporting utilities.

Covers:
  - tuning()         public API — spec creation, validation
  - _TuneSpec        internal dataclass — field values
  - _is_abstract()   helper — detects @abstractmethod with and without ABC
  - _get_concrete_subclasses()  helper — recursive subclass discovery
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pytest

from marl.utils.serialization import Serializable
from marl.utils.tuning import TUNE_KEY, _get_concrete_subclasses, _is_abstract, _TuneSpec, tuning

# ===========================================================================
# tuning()
# ===========================================================================


class TestTuningNumericRange:
    def test_creates_spec_with_correct_bounds(self):
        meta = tuning(1e-5, 1e-2)
        spec = meta[TUNE_KEY]
        assert spec.low == 1e-5
        assert spec.high == 1e-2

    def test_log_defaults_to_false(self):
        spec = tuning(1e-5, 1e-2)[TUNE_KEY]
        assert spec.log is False

    def test_log_flag_is_stored(self):
        spec = tuning(1e-5, 1e-2, log=True)[TUNE_KEY]
        assert spec.log is True

    def test_step_defaults_to_none(self):
        spec = tuning(16, 256)[TUNE_KEY]
        assert spec.step is None

    def test_step_is_stored(self):
        spec = tuning(16, 256, step=16)[TUNE_KEY]
        assert spec.step == 16

    def test_choices_is_none_for_numeric_range(self):
        spec = tuning(0.0, 1.0)[TUNE_KEY]
        assert spec.choices is None

    def test_returns_dict_with_tune_key(self):
        meta = tuning(0.0, 1.0)
        assert isinstance(meta, dict)
        assert TUNE_KEY in meta
        assert isinstance(meta[TUNE_KEY], _TuneSpec)

    def test_integer_bounds_are_stored_as_given(self):
        spec = tuning(50, 2000)[TUNE_KEY]
        assert spec.low == 50
        assert spec.high == 2000


class TestTuningChoices:
    def test_value_choices_stored(self):
        spec = tuning(choices=["adam", "rmsprop"])[TUNE_KEY]
        assert spec.choices == ["adam", "rmsprop"]

    def test_type_choices_stored(self):
        @dataclass
        class A(Serializable):
            pass

        @dataclass
        class B(Serializable):
            pass

        spec = tuning(choices=[A, B])[TUNE_KEY]
        assert spec.choices == [A, B]

    def test_bounds_are_none_for_choices(self):
        spec = tuning(choices=["a", "b"])[TUNE_KEY]
        assert spec.low is None
        assert spec.high is None


class TestTuningValidation:
    def test_raises_when_both_choices_and_low_provided(self):
        with pytest.raises(ValueError, match="cannot specify both"):
            tuning(1e-5, 1e-2, choices=["a"])

    def test_raises_when_both_choices_and_high_provided(self):
        with pytest.raises(ValueError, match="cannot specify both"):
            tuning(choices=["a"], high=1.0)

    def test_raises_when_only_low_provided(self):
        with pytest.raises(ValueError):
            tuning(1e-5)

    def test_raises_when_only_high_provided(self):
        with pytest.raises(ValueError):
            tuning(high=1e-2)

    def test_raises_when_no_args_provided(self):
        with pytest.raises(ValueError):
            tuning()

    def test_raises_on_none_choices(self):
        """Explicitly passing choices=None with no bounds should still raise."""
        with pytest.raises(ValueError):
            tuning(choices=None)


class TestTuningAsFieldMetadata:
    def test_metadata_survives_field_attachment(self):
        @dataclass
        class C:
            lr: float = field(default=1e-4, metadata=tuning(1e-5, 1e-2, log=True))

        import dataclasses

        f = next(f for f in dataclasses.fields(C) if f.name == "lr")
        spec = f.metadata[TUNE_KEY]
        assert spec.low == 1e-5
        assert spec.high == 1e-2
        assert spec.log is True

    def test_choices_metadata_survives_field_attachment(self):
        @dataclass
        class A(Serializable):
            pass

        @dataclass
        class B(Serializable):
            pass

        @dataclass
        class C:
            child: object = field(default_factory=A, metadata=tuning(choices=[A, B]))

        import dataclasses

        f = next(f for f in dataclasses.fields(C) if f.name == "child")
        spec = f.metadata[TUNE_KEY]
        assert A in spec.choices
        assert B in spec.choices


# ===========================================================================
# _is_abstract()
# ===========================================================================


class TestIsAbstract:
    # -- Without ABC (the common pattern in this codebase) --

    def test_class_with_abstractmethod_no_abc_is_abstract(self):
        class Base:
            @abstractmethod
            def do(self): ...

        assert _is_abstract(Base)

    def test_subclass_that_overrides_all_methods_is_not_abstract(self):
        class Base:
            @abstractmethod
            def do(self) -> int: ...

        class Concrete(Base):
            def do(self):
                return 42

        assert not _is_abstract(Concrete)

    def test_subclass_that_leaves_method_abstract_is_still_abstract(self):
        class Base:
            @abstractmethod
            def do(self): ...

        class StillAbstract(Base):
            pass  # doesn't override do()

        assert _is_abstract(StillAbstract)

    # -- With ABC --

    def test_abc_class_is_abstract(self):
        class Base(ABC):
            @abstractmethod
            def method(self): ...

        assert _is_abstract(Base)

    def test_abc_concrete_subclass_is_not_abstract(self):
        class Base(ABC):
            @abstractmethod
            def method(self) -> int: ...

        class Concrete(Base):
            def method(self):
                return 1

        assert not _is_abstract(Concrete)

    # -- Real codebase classes --

    def test_target_parameters_updater_is_abstract(self):
        from marl.training.qtarget_updater import TargetParametersUpdater

        assert _is_abstract(TargetParametersUpdater)

    def test_soft_update_is_not_abstract(self):
        from marl.training.qtarget_updater import SoftUpdate

        assert not _is_abstract(SoftUpdate)

    def test_hard_update_is_not_abstract(self):
        from marl.training.qtarget_updater import HardUpdate

        assert not _is_abstract(HardUpdate)

    def test_replay_memory_is_abstract(self):
        from marl.models.replay_memory import ReplayMemory

        assert _is_abstract(ReplayMemory)

    def test_transition_memory_is_not_abstract(self):
        from marl.models.replay_memory import TransitionMemory

        assert not _is_abstract(TransitionMemory)

    def test_policy_is_abstract(self):
        from marl.models import Policy

        assert _is_abstract(Policy)

    def test_arg_max_is_not_abstract(self):
        from marl.policy import ArgMax

        assert not _is_abstract(ArgMax)


# ===========================================================================
# _get_concrete_subclasses()
# ===========================================================================


class TestGetConcreteSubclasses:
    def test_finds_both_updater_subclasses(self):
        from marl.training.qtarget_updater import TargetParametersUpdater

        subs = _get_concrete_subclasses(TargetParametersUpdater)
        names = {c.__name__ for c in subs}
        assert "SoftUpdate" in names
        assert "HardUpdate" in names

    def test_does_not_include_abstract_base(self):
        from marl.training.qtarget_updater import TargetParametersUpdater

        subs = _get_concrete_subclasses(TargetParametersUpdater)
        assert TargetParametersUpdater not in subs

    def test_all_returned_classes_are_not_abstract(self):
        from marl.models.replay_memory import ReplayMemory

        subs = _get_concrete_subclasses(ReplayMemory)
        assert len(subs) > 0
        for cls in subs:
            assert not _is_abstract(cls), f"{cls.__name__} should be concrete"

    def test_recurses_into_intermediate_abstract_classes(self):
        """Concrete subclasses of abstract intermediate classes are found."""

        @dataclass
        class Grandparent(Serializable):
            @abstractmethod
            def act(self) -> int: ...

        @dataclass
        class Parent(Grandparent):
            @abstractmethod
            def act(self) -> int: ...

        @dataclass
        class Child(Parent):
            def act(self):
                return 1

        subs = _get_concrete_subclasses(Grandparent)
        assert Child in subs
        assert Grandparent not in subs
        assert Parent not in subs

    def test_returns_empty_list_for_leaf_class(self):
        @dataclass
        class Leaf(Serializable):
            x: int = 0

        subs = _get_concrete_subclasses(Leaf)
        assert subs == []
