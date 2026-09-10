"""
Tests for marl.utils.reflection.

Covers every public function in the module:

  unwrap_optional          — strip None from X | None → X
  is_abstract              — detect abstract classes (ABCMeta and bare @abstractmethod)
  get_concrete_subclasses  — recursive non-abstract subclass collection
  get_subclass_from_name   — depth-first subclass lookup by __name__
  get_subclass_map         — full {name: cls} hierarchy dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from marl.utils.reflection import (
    get_concrete_subclasses,
    get_subclass_from_name,
    get_subclass_map,
    is_abstract,
    unwrap_optional,
)
from marl.utils.serialization import Serializable

# ===========================================================================
# unwrap_optional
# ===========================================================================


class TestUnwrapOptional:
    # -- Modern PEP 604 syntax (X | None) --

    def test_unwraps_int_or_none(self):
        assert unwrap_optional(int | None) is int

    def test_unwraps_str_or_none(self):
        assert unwrap_optional(str | None) is str

    def test_unwraps_float_or_none(self):
        assert unwrap_optional(float | None) is float

    # -- Legacy typing.Optional / typing.Union --

    def test_unwraps_optional_int(self):
        assert unwrap_optional(Optional[int]) is int

    def test_unwraps_union_str_none(self):
        assert unwrap_optional(Union[str, None]) is str

    # -- Plain (non-optional) types are returned unchanged --

    def test_plain_int_unchanged(self):
        assert unwrap_optional(int) is int

    def test_plain_float_unchanged(self):
        assert unwrap_optional(float) is float

    def test_plain_str_unchanged(self):
        assert unwrap_optional(str) is str

    def test_plain_class_unchanged(self):
        class Foo:
            pass

        assert unwrap_optional(Foo) is Foo

    # -- Multi-member unions (not strictly X | None) are returned unchanged --

    def test_two_non_none_types_unchanged(self):
        hint = int | str
        assert unwrap_optional(hint) is hint

    def test_three_member_union_unchanged(self):
        hint = int | str | None
        # Two non-None members → cannot unwrap to a single type
        assert unwrap_optional(hint) is hint

    # -- Parameterized generics must NOT be unwrapped --

    def test_list_of_int_is_unchanged(self):
        """List[int] has a single type arg but is not a Union — must not be unwrapped."""
        from typing import List

        hint = List[int]
        assert unwrap_optional(hint) is hint

    def test_parameterized_serializable_is_unchanged(self):
        """EnvConfig[E] has a single TypeVar arg but is not Optional — must not be unwrapped.

        This is the exact case that triggered the serialization regression:
        resolve_type(EnvConfig[E]) was incorrectly returning the TypeVar E
        (which resolved further to its MARLEnv bound) instead of EnvConfig.
        """
        import typing

        from marl.env import EnvConfig

        E = typing.TypeVar("E")
        hint = EnvConfig[E]
        assert unwrap_optional(hint) is hint

    # -- Identity: unwrapping twice is idempotent --

    def test_idempotent_on_optional(self):
        unwrapped = unwrap_optional(int | None)
        assert unwrap_optional(unwrapped) is int

    # -- Unwrapping preserves the inner type object --

    def test_returns_exact_inner_class_for_serializable(self):
        @dataclass
        class MyClass(Serializable):
            pass

        assert unwrap_optional(MyClass | None) is MyClass


# ===========================================================================
# is_abstract
# ===========================================================================


class TestIsAbstract:
    # -- Without ABC (the common pattern in this codebase) --

    def test_class_with_abstractmethod_no_abc_is_abstract(self):
        class Base:
            @abstractmethod
            def do(self): ...

        assert is_abstract(Base)

    def test_subclass_that_overrides_all_methods_is_not_abstract(self):
        class Base:
            @abstractmethod
            def do(self): ...

        class Concrete(Base):
            def do(self) -> None:
                pass

        assert not is_abstract(Concrete)

    def test_subclass_that_leaves_method_abstract_is_still_abstract(self):
        class Base:
            @abstractmethod
            def do(self): ...

        class StillAbstract(Base):
            pass  # doesn't implement do()

        assert is_abstract(StillAbstract)

    # -- With ABC --

    def test_abc_class_is_abstract(self):
        class Base(ABC):
            @abstractmethod
            def method(self): ...

        assert is_abstract(Base)

    def test_abc_concrete_subclass_is_not_abstract(self):
        class Base(ABC):
            @abstractmethod
            def method(self): ...

        class Concrete(Base):
            def method(self) -> None:
                pass

        assert not is_abstract(Concrete)

    # -- Real codebase classes --

    def test_target_parameters_updater_is_abstract(self):
        from marl.algos.qtarget_updater import TargetParametersUpdater

        assert is_abstract(TargetParametersUpdater)

    def test_soft_update_is_not_abstract(self):
        from marl.algos.qtarget_updater import SoftUpdate

        assert not is_abstract(SoftUpdate)

    def test_hard_update_is_not_abstract(self):
        from marl.algos.qtarget_updater import HardUpdate

        assert not is_abstract(HardUpdate)

    def test_replay_memory_is_abstract(self):
        from marl.models.replay_memory import ReplayMemory

        assert is_abstract(ReplayMemory)

    def test_transition_memory_is_not_abstract(self):
        from marl.models.replay_memory import TransitionMemory

        assert not is_abstract(TransitionMemory)

    def test_policy_is_abstract(self):
        from marl.models import Policy

        assert is_abstract(Policy)

    def test_arg_max_is_not_abstract(self):
        from marl.policy import ArgMax

        assert not is_abstract(ArgMax)


# ===========================================================================
# get_concrete_subclasses
# ===========================================================================


class TestGetConcreteSubclasses:
    def test_finds_both_updater_subclasses(self):
        from marl.algos.qtarget_updater import TargetParametersUpdater

        subs = get_concrete_subclasses(TargetParametersUpdater)
        names = {c.__name__ for c in subs}
        assert "SoftUpdate" in names
        assert "HardUpdate" in names

    def test_does_not_include_abstract_base(self):
        from marl.algos.qtarget_updater import TargetParametersUpdater

        subs = get_concrete_subclasses(TargetParametersUpdater)
        assert TargetParametersUpdater not in subs

    def test_all_returned_classes_are_concrete(self):
        from marl.models.replay_memory import ReplayMemory

        subs = get_concrete_subclasses(ReplayMemory)
        assert len(subs) > 0
        for cls in subs:
            assert not is_abstract(cls), f"{cls.__name__} should be concrete"

    def test_recurses_into_intermediate_abstract_classes(self):
        @dataclass
        class Grandparent(Serializable):
            @abstractmethod
            def act(self): ...

        @dataclass
        class Parent(Grandparent):
            @abstractmethod
            def act(self): ...

        @dataclass
        class Child(Parent):
            def act(self) -> None:
                pass

        subs = get_concrete_subclasses(Grandparent)
        assert Child in subs
        assert Grandparent not in subs
        assert Parent not in subs

    def test_returns_empty_list_for_leaf_class(self):
        @dataclass
        class Leaf(Serializable):
            x: int = 0

        assert get_concrete_subclasses(Leaf) == []

    def test_concrete_base_is_not_included_in_result(self):
        """get_concrete_subclasses never includes the root even when it is concrete."""

        @dataclass
        class Root(Serializable):
            pass  # concrete, no abstract methods

        @dataclass
        class Child(Root):
            pass

        subs = get_concrete_subclasses(Root)
        assert Root not in subs
        assert Child in subs


# ===========================================================================
# get_subclass_from_name
# ===========================================================================


class TestGetSubclassFromName:
    @dataclass
    class Animal(Serializable):
        pass

    @dataclass
    class Dog(Animal):
        pass

    @dataclass
    class Cat(Animal):
        pass

    @dataclass
    class Kitten(Cat):
        pass

    def test_finds_direct_subclass(self):
        result = get_subclass_from_name(self.Animal, "Dog")
        assert result is self.Dog

    def test_finds_other_direct_subclass(self):
        result = get_subclass_from_name(self.Animal, "Cat")
        assert result is self.Cat

    def test_finds_deeply_nested_subclass(self):
        """Kitten is a subclass of Cat, which is a subclass of Animal."""
        result = get_subclass_from_name(self.Animal, "Kitten")
        assert result is self.Kitten

    def test_returns_none_for_unknown_name(self):
        assert get_subclass_from_name(self.Animal, "NonExistent") is None

    def test_base_class_itself_is_not_returned(self):
        """Base class name is never a match, even if it equals the requested name."""
        assert get_subclass_from_name(self.Animal, "Animal") is None

    def test_returns_first_match_depth_first(self):
        """When multiple classes share a name (unusual), depth-first wins."""

        @dataclass
        class Root(Serializable):
            pass

        @dataclass
        class Branch(Root):
            pass

        @dataclass
        class Leaf(Branch):  # name "Leaf" at depth 2
            pass

        result = get_subclass_from_name(Root, "Leaf")
        assert result is Leaf

    def test_works_on_real_codebase_classes(self):
        from marl.algos.qtarget_updater import SoftUpdate, TargetParametersUpdater

        result = get_subclass_from_name(TargetParametersUpdater, "SoftUpdate")
        assert result is SoftUpdate

    def test_returns_none_for_sibling_class(self):
        """SoftUpdate is not a subclass of HardUpdate."""
        from marl.algos.qtarget_updater import HardUpdate

        assert get_subclass_from_name(HardUpdate, "SoftUpdate") is None


# ===========================================================================
# get_subclass_map
# ===========================================================================


class TestGetSubclassMap:
    @dataclass
    class Shape(Serializable):
        pass

    @dataclass
    class Circle(Shape):
        pass

    @dataclass
    class Square(Shape):
        pass

    @dataclass
    class SmallSquare(Square):
        pass

    def test_includes_base_class_itself(self):
        m = get_subclass_map(self.Shape)
        assert "Shape" in m
        assert m["Shape"] is self.Shape

    def test_includes_direct_subclasses(self):
        m = get_subclass_map(self.Shape)
        assert "Circle" in m
        assert m["Circle"] is self.Circle
        assert "Square" in m

    def test_includes_deeply_nested_subclasses(self):
        m = get_subclass_map(self.Shape)
        assert "SmallSquare" in m
        assert m["SmallSquare"] is self.SmallSquare

    def test_values_are_class_objects(self):
        m = get_subclass_map(self.Shape)
        for name, cls in m.items():
            assert isinstance(cls, type)
            assert cls.__name__ == name

    def test_difference_from_get_subclass_from_name(self):
        """get_subclass_map includes the base; get_subclass_from_name does not."""
        assert get_subclass_from_name(self.Shape, "Shape") is None
        assert get_subclass_map(self.Shape).get("Shape") is self.Shape

    def test_works_on_real_codebase_hierarchy(self):
        from marl.algos.qtarget_updater import TargetParametersUpdater

        m = get_subclass_map(TargetParametersUpdater)
        assert "TargetParametersUpdater" in m
        assert "SoftUpdate" in m
        assert "HardUpdate" in m
