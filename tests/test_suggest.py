"""
Tests for the suggest() function.

The file is organised in sections that mirror the 12-rule decision table:

  Rule  1  Override
  Rules 2-3  tuning(choices=[...])
  Rules 4-5  tuning(low, high)
  Rule  6  bool auto-detection
  Rule  7  Literal auto-detection
  Rule  8  Concrete Serializable recursion
  Rule  9  Abstract Serializable auto-collection
  Rule 10  float/int default → use + warn
  Rule 11  other default → use silently
  Rule 12  required field, no default → raise

  Naming    Dot-separated parameter names
  Real      Tests against actual annotated codebase classes
  DQN       End-to-end integration test
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sequence

import pytest
from optuna import Trial

from marl.utils import suggest, tuning
from marl.utils.serialization import Serializable

# ===========================================================================
# MockTrial
#
# Records every suggest_* call so tests can assert on which parameters were
# registered and with which arguments.  Returns values from a pre-set
# response dict; falls back to the midpoint for numeric ranges and the first
# element for categoricals.
# ===========================================================================


class MockTrial(Trial):
    """Lightweight stand-in for optuna.Trial that records all suggest calls."""

    def __init__(self, responses: dict | None = None):
        self.responses: dict = responses or {}
        self.calls: list[dict] = []

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False, step=None) -> float:
        self.calls.append({"method": "float", "name": name, "low": low, "high": high, "log": log, "step": step})
        return float(self.responses.get(name, (low + high) / 2))

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False, step: int = 1) -> int:
        self.calls.append({"method": "int", "name": name, "low": low, "high": high, "log": log, "step": step})
        return int(self.responses.get(name, (low + high) // 2))

    def suggest_categorical(self, name: str, choices: Sequence):
        self.calls.append({"method": "categorical", "name": name, "choices": list(choices)})
        return self.responses.get(name, choices[0])

    # -- Convenience query helpers --

    def calls_for(self, name: str) -> list[dict]:
        return [c for c in self.calls if c["name"] == name]

    def single_call(self, name: str) -> dict:
        """Return the one call for *name*, asserting exactly one exists."""
        matching = self.calls_for(name)
        assert len(matching) == 1, f"Expected 1 call for '{name}', got {len(matching)}: {matching}"
        return matching[0]

    def registered_names(self) -> set[str]:
        return {c["name"] for c in self.calls}


# ===========================================================================
# Shared test dataclasses
# ===========================================================================


@dataclass
class _FloatField(Serializable):
    x: float = field(default=0.5, metadata=tuning(0.1, 1.0))


@dataclass
class _IntField(Serializable):
    n: int = field(default=64, metadata=tuning(16, 256))


@dataclass
class _BoolField(Serializable):
    flag: bool = True


@dataclass
class _LiteralField(Serializable):
    mode: Literal["fast", "slow"] = "fast"


# ===========================================================================
# Rule 1 — Override
# ===========================================================================


class TestRuleOverride:
    def test_override_value_is_used_verbatim(self):
        trial = MockTrial()
        result = suggest(_FloatField, trial, x=99.0)
        assert result.x == 99.0

    def test_no_suggest_call_made_for_overridden_field(self):
        trial = MockTrial()
        suggest(_FloatField, trial, x=99.0)
        assert "x" not in trial.registered_names()

    def test_override_works_for_required_field(self):
        @dataclass
        class C(Serializable):
            required: float  # no default

        result = suggest(C, MockTrial(), required=3.14)
        assert result.required == 3.14

    def test_override_takes_priority_over_tuning_annotation(self):
        """Even a fully annotated field is bypassed when overridden."""
        trial = MockTrial(responses={"x": 0.55})
        result = suggest(_FloatField, trial, x=999.0)
        assert result.x == 999.0


# ===========================================================================
# Rules 2 & 3 — tuning(choices=[...])
# ===========================================================================


class TestRuleChoices:
    def test_type_choices_produces_categorical_call_with_class_names(self):
        """Class names (strings) are passed to suggest_categorical, not type objects."""

        @dataclass
        class Sub1(Serializable):
            v: float = field(default=1.0, metadata=tuning(0.5, 2.0))

        @dataclass
        class Sub2(Serializable):
            n: int = field(default=10, metadata=tuning(5, 20))

        @dataclass
        class Parent(Serializable):
            child: object = field(default_factory=Sub1, metadata=tuning(choices=[Sub1, Sub2]))

        trial = MockTrial()
        suggest(Parent, trial)

        call = trial.single_call("child.__type__")
        assert call["method"] == "categorical"
        assert set(call["choices"]) == {"Sub1", "Sub2"}

    def test_type_choices_recurses_into_chosen_subclass(self):
        @dataclass
        class A(Serializable):
            a: float = field(default=1.0, metadata=tuning(0.5, 2.0))

        @dataclass
        class B(Serializable):
            b: int = field(default=10, metadata=tuning(5, 20))

        @dataclass
        class Container(Serializable):
            item: object = field(default_factory=A, metadata=tuning(choices=[A, B]))

        trial = MockTrial(responses={"item.__type__": "B", "item.b": 15})
        result = suggest(Container, trial)

        assert isinstance(result.item, B)
        assert result.item.b == 15

    def test_type_choices_uses_prefix_for_subclass_fields(self):
        @dataclass
        class Sub(Serializable):
            val: float = field(default=1.0, metadata=tuning(0.5, 2.0))

        @dataclass
        class Outer(Serializable):
            inner: object = field(default_factory=Sub, metadata=tuning(choices=[Sub]))

        trial = MockTrial(responses={"inner.__type__": "Sub"})
        suggest(Outer, trial)

        assert "inner.val" in trial.registered_names()

    def test_value_choices_uses_suggest_categorical_directly(self):
        @dataclass
        class C(Serializable):
            opt: str = field(default="adam", metadata=tuning(choices=["adam", "rmsprop"]))

        trial = MockTrial(responses={"opt": "rmsprop"})
        result = suggest(C, trial)

        assert result.opt == "rmsprop"
        call = trial.single_call("opt")
        assert call["method"] == "categorical"
        assert set(call["choices"]) == {"adam", "rmsprop"}

    def test_type_choices_restricts_abstract_subclass_search(self):
        """tuning(choices=[SubA]) on an abstract-typed field uses only SubA, not all subclasses."""
        from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater

        @dataclass
        class Wrapper(Serializable):
            updater: TargetParametersUpdater = field(
                default_factory=SoftUpdate,
                metadata=tuning(choices=[SoftUpdate]),  # only SoftUpdate allowed
            )

        trial = MockTrial(responses={"updater.__type__": "SoftUpdate"})
        result = suggest(Wrapper, trial)

        assert isinstance(result.updater, SoftUpdate)
        call = trial.single_call("updater.__type__")
        # HardUpdate must NOT appear — the user restricted the search
        assert "HardUpdate" not in call["choices"]


# ===========================================================================
# Rules 4 & 5 — tuning(low, high)
# ===========================================================================


class TestRuleNumericRange:
    def test_float_field_calls_suggest_float(self):
        trial = MockTrial()
        suggest(_FloatField, trial)
        call = trial.single_call("x")
        assert call["method"] == "float"

    def test_float_field_passes_correct_bounds(self):
        trial = MockTrial()
        suggest(_FloatField, trial)
        call = trial.single_call("x")
        assert call["low"] == pytest.approx(0.1)
        assert call["high"] == pytest.approx(1.0)

    def test_float_field_passes_log_flag(self):
        @dataclass
        class C(Serializable):
            lr: float = field(default=1e-4, metadata=tuning(1e-5, 1e-2, log=True))

        trial = MockTrial()
        suggest(C, trial)
        assert trial.single_call("lr")["log"] is True

    def test_float_field_passes_step(self):
        @dataclass
        class C(Serializable):
            val: float = field(default=0.5, metadata=tuning(0.0, 1.0, step=0.1))

        trial = MockTrial()
        suggest(C, trial)
        assert trial.single_call("val")["step"] == pytest.approx(0.1)

    def test_int_field_calls_suggest_int(self):
        trial = MockTrial()
        suggest(_IntField, trial)
        call = trial.single_call("n")
        assert call["method"] == "int"

    def test_int_field_passes_correct_bounds(self):
        trial = MockTrial()
        suggest(_IntField, trial)
        call = trial.single_call("n")
        assert call["low"] == 16
        assert call["high"] == 256

    def test_int_field_returns_integer(self):
        trial = MockTrial(responses={"n": 128})
        result = suggest(_IntField, trial)
        assert result.n == 128
        assert isinstance(result.n, int)

    def test_float_suggested_value_is_used(self):
        trial = MockTrial(responses={"x": 0.7})
        result = suggest(_FloatField, trial)
        assert result.x == pytest.approx(0.7)

    def test_bounds_are_satisfied_with_real_optuna(self):
        """Values from a real Optuna sampler must respect the declared bounds."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        @dataclass
        class C(Serializable):
            tau: float = field(default=0.01, metadata=tuning(1e-3, 0.5, log=True))
            period: int = field(default=200, metadata=tuning(50, 2000))

        study = optuna.create_study()
        for _ in range(5):
            trial = study.ask()
            result = suggest(C, trial)
            study.tell(trial, 0.0)
            assert 1e-3 <= result.tau <= 0.5
            assert 50 <= result.period <= 2000


# ===========================================================================
# Rule 6 — bool auto-detection
# ===========================================================================


class TestRuleBool:
    def test_bool_field_calls_suggest_categorical(self):
        trial = MockTrial()
        suggest(_BoolField, trial)
        call = trial.single_call("flag")
        assert call["method"] == "categorical"

    def test_bool_choices_are_true_and_false(self):
        trial = MockTrial()
        suggest(_BoolField, trial)
        assert set(trial.single_call("flag")["choices"]) == {True, False}

    def test_bool_false_can_be_returned(self):
        trial = MockTrial(responses={"flag": False})
        result = suggest(_BoolField, trial)
        assert result.flag is False

    def test_bool_true_can_be_returned(self):
        trial = MockTrial(responses={"flag": True})
        result = suggest(_BoolField, trial)
        assert result.flag is True


# ===========================================================================
# Rule 7 — Literal auto-detection
# ===========================================================================


class TestRuleLiteral:
    def test_literal_field_calls_suggest_categorical(self):
        trial = MockTrial()
        suggest(_LiteralField, trial)
        call = trial.single_call("mode")
        assert call["method"] == "categorical"

    def test_literal_choices_match_type_args(self):
        trial = MockTrial()
        suggest(_LiteralField, trial)
        assert set(trial.single_call("mode")["choices"]) == {"fast", "slow"}

    def test_literal_suggested_value_is_used(self):
        trial = MockTrial(responses={"mode": "slow"})
        result = suggest(_LiteralField, trial)
        assert result.mode == "slow"

    def test_literal_with_more_choices(self):
        @dataclass
        class C(Serializable):
            opt: Literal["adam", "rmsprop", "sgd"] = "adam"

        trial = MockTrial()
        suggest(C, trial)
        assert set(trial.single_call("opt")["choices"]) == {"adam", "rmsprop", "sgd"}


# ===========================================================================
# Rule 8 — Concrete Serializable recursion
# ===========================================================================


class TestRuleConcreteSerializable:
    def test_concrete_field_is_recursed_into(self):
        @dataclass
        class Inner(Serializable):
            val: float = field(default=0.5, metadata=tuning(0.1, 1.0))

        @dataclass
        class Outer(Serializable):
            inner: Inner = field(default_factory=lambda: Inner(0.5))

        trial = MockTrial(responses={"inner.val": 0.7})
        result = suggest(Outer, trial)

        assert isinstance(result.inner, Inner)
        assert result.inner.val == pytest.approx(0.7)

    def test_concrete_field_uses_dot_separated_name(self):
        @dataclass
        class Inner(Serializable):
            val: float = field(default=0.5, metadata=tuning(0.1, 1.0))

        @dataclass
        class Outer(Serializable):
            inner: Inner = field(default_factory=lambda: Inner(0.5))

        trial = MockTrial()
        suggest(Outer, trial)

        assert "inner.val" in trial.registered_names()
        assert "val" not in trial.registered_names()

    def test_concrete_field_produces_correct_instance_type(self):
        from marl.training.qtarget_updater import SoftUpdate

        @dataclass
        class Wrapper(Serializable):
            updater: SoftUpdate = field(default_factory=SoftUpdate)

        trial = MockTrial()
        result = suggest(Wrapper, trial)
        assert isinstance(result.updater, SoftUpdate)


# ===========================================================================
# Rule 9 — Abstract Serializable auto-collection
# ===========================================================================


class TestRuleAbstractSerializable:
    def test_abstract_field_registers_type_choice(self):
        @dataclass
        class Base(Serializable):
            @abstractmethod
            def act(self) -> str: ...

        @dataclass
        class ConcreteA(Base):
            def act(self):
                return "a"

        @dataclass
        class ConcreteB(Base):
            def act(self):
                return "b"

        @dataclass
        class Container(Serializable):
            item: Base = field(default_factory=ConcreteA)

        trial = MockTrial()
        suggest(Container, trial)

        assert "item.__type__" in trial.registered_names()

    def test_abstract_field_categorical_choices_are_concrete_class_names(self):
        @dataclass
        class Base(Serializable):
            @abstractmethod
            def act(self) -> str: ...

        @dataclass
        class ConcreteA(Base):
            def act(self):
                return "a"

        @dataclass
        class ConcreteB(Base):
            def act(self):
                return "b"

        @dataclass
        class Container(Serializable):
            item: Base = field(default_factory=ConcreteA)

        trial = MockTrial()
        suggest(Container, trial)

        type_choices = set(trial.single_call("item.__type__")["choices"])
        assert "ConcreteA" in type_choices
        assert "ConcreteB" in type_choices

    def test_abstract_field_recurses_into_chosen_class(self):
        @dataclass
        class Base(Serializable):
            @abstractmethod
            def act(self) -> float: ...

        @dataclass
        class ConcreteA(Base):
            a: float = field(default=1.0, metadata=tuning(0.5, 2.0))

            def act(self):
                return self.a

        @dataclass
        class ConcreteB(Base):
            b: int = field(default=10, metadata=tuning(5, 20))

            def act(self):
                return self.b

        @dataclass
        class Container(Serializable):
            item: Base = field(default_factory=ConcreteA)

        trial = MockTrial(responses={"item.__type__": "ConcreteB", "item.b": 15})
        result = suggest(Container, trial)

        assert isinstance(result.item, ConcreteB)
        assert result.item.b == 15

    def test_abstract_target_updater_auto_collects_soft_and_hard(self):
        """Integration: TargetParametersUpdater is abstract, so HardUpdate and SoftUpdate are found."""
        from marl.training.qtarget_updater import HardUpdate, SoftUpdate, TargetParametersUpdater

        @dataclass
        class Wrapper(Serializable):
            updater: TargetParametersUpdater = field(default_factory=SoftUpdate)

        trial = MockTrial(responses={"updater.__type__": "HardUpdate", "updater.update_period": 300})
        result = suggest(Wrapper, trial)

        assert isinstance(result.updater, HardUpdate)
        assert result.updater.update_period == 300

    def test_abstract_updater_tau_is_suggested_when_soft_chosen(self):
        from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater

        @dataclass
        class Wrapper(Serializable):
            updater: TargetParametersUpdater = field(default_factory=SoftUpdate)

        trial = MockTrial(responses={"updater.__type__": "SoftUpdate", "updater.tau": 0.05})
        result = suggest(Wrapper, trial)

        assert isinstance(result.updater, SoftUpdate)
        assert result.updater.tau == pytest.approx(0.05)
        call = trial.single_call("updater.tau")
        assert call["method"] == "float"
        assert call["log"] is True  # SoftUpdate.tau has log=True


# ===========================================================================
# Rule 10 — float/int with default → use + warn
# ===========================================================================


class TestRuleDefaultWithWarning:
    def test_unannotated_float_uses_default(self, caplog):
        @dataclass
        class C(Serializable):
            gamma: float = 0.99

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.gamma == pytest.approx(0.99)

    def test_unannotated_float_emits_warning(self, caplog):
        @dataclass
        class C(Serializable):
            gamma: float = 0.99

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            suggest(C, MockTrial())

        assert "gamma" in caplog.text
        assert "no tuning()" in caplog.text

    def test_unannotated_int_uses_default(self, caplog):
        @dataclass
        class C(Serializable):
            n_steps: int = 1000

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.n_steps == 1000

    def test_unannotated_int_emits_warning(self, caplog):
        @dataclass
        class C(Serializable):
            n_steps: int = 1000

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            suggest(C, MockTrial())

        assert "n_steps" in caplog.text

    def test_warning_names_the_class(self, caplog):
        @dataclass
        class MySpecificClass(Serializable):
            rate: float = 0.5

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            suggest(MySpecificClass, MockTrial())

        assert "MySpecificClass" in caplog.text

    def test_warning_names_the_field(self, caplog):
        @dataclass
        class C(Serializable):
            my_particular_rate: float = 0.5

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            suggest(C, MockTrial())

        assert "my_particular_rate" in caplog.text

    def test_no_suggest_call_made_for_untuned_field(self, caplog):
        @dataclass
        class C(Serializable):
            gamma: float = 0.99

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            trial = MockTrial()
            suggest(C, trial)

        assert "gamma" not in trial.registered_names()

    def test_optional_float_warns_and_uses_non_none_default(self, caplog):
        """float | None with a non-None default still triggers a warning."""

        @dataclass
        class C(Serializable):
            clip: float | None = 10.0

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.clip == pytest.approx(10.0)
        assert "clip" in caplog.text


# ===========================================================================
# Rule 11 — other default → use silently
# ===========================================================================


class TestRuleDefaultSilent:
    def test_none_default_is_used_without_warning(self, caplog):
        @dataclass
        class C(Serializable):
            thing: object = None

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.thing is None
        assert caplog.text == ""

    def test_string_default_is_used_without_warning(self, caplog):
        @dataclass
        class C(Serializable):
            name: str = "default"

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.name == "default"
        assert caplog.text == ""

    def test_tuple_default_is_used_without_warning(self, caplog):
        @dataclass
        class C(Serializable):
            interval: tuple = (5, "step")

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.interval == (5, "step")
        assert caplog.text == ""

    def test_none_default_for_optional_field_is_silent(self, caplog):
        @dataclass
        class C(Serializable):
            maybe: float | None = None

        with caplog.at_level(logging.WARNING, logger="marl.utils.tuning"):
            result = suggest(C, MockTrial())

        assert result.maybe is None
        assert caplog.text == ""


# ===========================================================================
# Rule 12 — required field, no default → raise
# ===========================================================================


class TestRuleRequired:
    def test_required_field_raises_value_error(self):
        @dataclass
        class C(Serializable):
            required: float

        with pytest.raises(ValueError):
            suggest(C, MockTrial())

    def test_error_message_names_the_field(self):
        @dataclass
        class C(Serializable):
            my_special_field: int

        with pytest.raises(ValueError, match="my_special_field"):
            suggest(C, MockTrial())

    def test_error_message_names_the_class(self):
        @dataclass
        class MySpecialClass(Serializable):
            required: float

        with pytest.raises(ValueError, match="MySpecialClass"):
            suggest(MySpecialClass, MockTrial())

    def test_override_satisfies_required_field_no_raise(self):
        @dataclass
        class C(Serializable):
            required: float

        result = suggest(C, MockTrial(), required=3.14)
        assert result.required == pytest.approx(3.14)


# ===========================================================================
# Dot-separated parameter naming
# ===========================================================================


class TestParameterNaming:
    def test_top_level_field_has_plain_name(self):
        trial = MockTrial()
        suggest(_FloatField, trial)
        assert "x" in trial.registered_names()

    def test_nested_field_has_dot_separated_name(self):
        @dataclass
        class Inner(Serializable):
            tau: float = field(default=0.01, metadata=tuning(1e-3, 0.5))

        @dataclass
        class Outer(Serializable):
            target_updater: Inner = field(default_factory=Inner)

        trial = MockTrial()
        suggest(Outer, trial)

        assert "target_updater.tau" in trial.registered_names()
        assert "tau" not in trial.registered_names()

    def test_three_levels_deep(self):
        @dataclass
        class L3(Serializable):
            z: float = field(default=1.0, metadata=tuning(0.5, 2.0))

        @dataclass
        class L2(Serializable):
            l3: L3 = field(default_factory=L3)

        @dataclass
        class L1(Serializable):
            l2: L2 = field(default_factory=L2)

        trial = MockTrial()
        suggest(L1, trial)

        assert "l2.l3.z" in trial.registered_names()

    def test_abstract_type_uses_dot_type_suffix(self):
        from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater

        @dataclass
        class Wrapper(Serializable):
            updater: TargetParametersUpdater = field(default_factory=SoftUpdate)

        trial = MockTrial()
        suggest(Wrapper, trial)

        assert "updater.__type__" in trial.registered_names()

    def test_custom_prefix_is_prepended(self):
        trial = MockTrial()
        suggest(_FloatField, trial, prefix="parent")

        assert "parent.x" in trial.registered_names()
        assert "x" not in trial.registered_names()

    def test_abstract_subclass_fields_use_parent_field_as_prefix(self):
        """After choosing 'SoftUpdate' for 'updater', its 'tau' becomes 'updater.tau'."""
        from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater

        @dataclass
        class Wrapper(Serializable):
            updater: TargetParametersUpdater = field(default_factory=SoftUpdate)

        trial = MockTrial(responses={"updater.__type__": "SoftUpdate"})
        suggest(Wrapper, trial)

        assert "updater.tau" in trial.registered_names()


# ===========================================================================
# Real annotated codebase classes
# ===========================================================================


class TestRealClasses:
    def test_suggest_soft_update(self):
        from marl.training.qtarget_updater import SoftUpdate

        trial = MockTrial()
        result = suggest(SoftUpdate, trial)

        assert isinstance(result, SoftUpdate)
        call = trial.single_call("tau")
        assert call["method"] == "float"
        assert call["log"] is True
        assert call["low"] == pytest.approx(1e-3)
        assert call["high"] == pytest.approx(0.5)
        assert 1e-3 <= result.tau <= 0.5

    def test_suggest_hard_update(self):
        from marl.training.qtarget_updater import HardUpdate

        trial = MockTrial()
        result = suggest(HardUpdate, trial)

        assert isinstance(result, HardUpdate)
        call = trial.single_call("update_period")
        assert call["method"] == "int"
        assert call["low"] == 50
        assert call["high"] == 2000
        assert 50 <= result.update_period <= 2000

    def test_suggest_transition_memory(self):
        from marl.models.replay_memory import TransitionMemory

        trial = MockTrial()
        result = suggest(TransitionMemory, trial)

        assert isinstance(result, TransitionMemory)
        call = trial.single_call("max_size")
        assert call["method"] == "int"
        assert call["low"] == 1_000
        assert call["high"] == 200_000
        assert 1_000 <= result.max_size <= 200_000

    def test_soft_update_tau_bounds_with_real_optuna(self):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from marl.training.qtarget_updater import SoftUpdate

        study = optuna.create_study()
        for _ in range(5):
            trial = study.ask()
            result = suggest(SoftUpdate, trial)
            study.tell(trial, result.tau)
            assert 1e-3 <= result.tau <= 0.5

    def test_hard_update_period_bounds_with_real_optuna(self):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from marl.training.qtarget_updater import HardUpdate

        study = optuna.create_study()
        for _ in range(5):
            trial = study.ask()
            result = suggest(HardUpdate, trial)
            study.tell(trial, result.update_period)
            assert 50 <= result.update_period <= 2000


# ===========================================================================
# DQN end-to-end integration
# ===========================================================================


class TestDQNIntegration:
    """
    Full integration test: suggest() on DQN with real Optuna trials.

    env-dependent fields (qnetwork, memory, mixer, train_policy, test_policy)
    are passed as overrides.  All other fields are auto-suggested.
    """

    @pytest.fixture(scope="class")
    def env(self):
        from marlenv.catalog import DiscreteMockEnv

        return DiscreteMockEnv()

    @pytest.fixture(scope="class")
    def dqn_trial_result(self, env):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from marl import policy
        from marl.models.replay_memory import TransitionMemory
        from marl.nn.model_bank import qnetworks
        from marl.training.dqn import DQN

        study = optuna.create_study()
        trial = study.ask()
        trainer = suggest(
            DQN,
            trial,
            qnetwork=qnetworks.from_env(env),
            memory=suggest(TransitionMemory, trial),
            mixer=None,
            train_policy=policy.EpsilonGreedy.linear(1.0, 0.05, 50_000),
            # test_policy is overridden to avoid random failure from Policy
            # auto-collection (some concrete Policy subclasses such as
            # EpsilonGreedy require constructor args not available here).
            test_policy=policy.ArgMax(),
        )
        return trainer, trial

    def test_result_is_dqn_instance(self, dqn_trial_result):
        from marl.training.dqn import DQN

        trainer, _ = dqn_trial_result
        assert isinstance(trainer, DQN)

    def test_lr_is_within_declared_bounds(self, dqn_trial_result):
        trainer, _ = dqn_trial_result
        assert 1e-5 <= trainer.lr <= 1e-2

    def test_batch_size_is_within_declared_bounds(self, dqn_trial_result):
        trainer, _ = dqn_trial_result
        assert 16 <= trainer.batch_size <= 256

    def test_memory_max_size_is_within_declared_bounds(self, dqn_trial_result):
        trainer, _ = dqn_trial_result
        assert 1_000 <= trainer.memory.max_size <= 200_000

    def test_double_qlearning_is_bool(self, dqn_trial_result):
        trainer, _ = dqn_trial_result
        assert isinstance(trainer.double_qlearning, bool)

    def test_optimiser_type_is_valid(self, dqn_trial_result):
        trainer, _ = dqn_trial_result
        assert trainer.optimiser_type in ("adam", "rmsprop")

    def test_target_updater_is_soft_or_hard(self, dqn_trial_result):
        from marl.training.qtarget_updater import HardUpdate, SoftUpdate

        trainer, _ = dqn_trial_result
        assert isinstance(trainer.target_updater, (SoftUpdate, HardUpdate))

    def test_target_updater_type_is_registered_in_trial(self, dqn_trial_result):
        _, trial = dqn_trial_result
        assert "target_updater.__type__" in trial.params

    def test_lr_is_registered_in_trial(self, dqn_trial_result):
        _, trial = dqn_trial_result
        assert "lr" in trial.params

    def test_batch_size_is_registered_in_trial(self, dqn_trial_result):
        _, trial = dqn_trial_result
        assert "batch_size" in trial.params
