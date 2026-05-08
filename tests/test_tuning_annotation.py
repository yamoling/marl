"""
Tests for the tuning() annotation helper.

Covers:
  - tuning()   public API — spec creation, validation
  - _TuneSpec  internal dataclass — field values

Note: is_abstract and get_concrete_subclasses were previously private helpers
in this module. They are now in marl.utils.reflection and tested in
test_reflection.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from marl.utils.serialization import Serializable
from marl.utils.tuning import TUNE_KEY, _TuneSpec, tuning

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
