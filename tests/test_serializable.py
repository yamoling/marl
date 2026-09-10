"""
Tests for the Serializable base class.

Covers:
  - to_dict / from_dict  (flat, nested, polymorphic, optional, defaults)
  - to_json  / from_json  (round-trip, beautify flag)
  - to_file  / from_file  (round-trip, path handling)
  - the discriminator key ("class-name") mechanics
  - DQN trainer  — full round-trip is possible (all fields are proper dataclasses)
  - PPO trainer  — serialization only (SimpleActorCritic cannot be round-tripped
                   because it uses a custom __init__ with parameters that are
                   not exposed as dataclass fields)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from marlenv.catalog import DiscreteMockEnv

from marl import Experiment, algos
from marl.env import EnvConfig
from marl.nn.model_bank import qnetworks
from marl.utils.serialization import DISCRIMINATOR_KEY, Serializable

# ===========================================================================
# Mock dataclasses used throughout this module
# ===========================================================================


@dataclass
class Atom(Serializable):
    """Flat leaf — all primitive scalar fields."""

    x: float
    y: float
    label: str = "default"


@dataclass
class Molecule(Serializable):
    """Wraps one Atom — exercises nested Serializable recursion."""

    atom: Atom
    weight: float = 1.0


@dataclass
class Vehicle(Serializable):
    """Abstract-ish base used for polymorphic dispatch tests."""

    speed: int = 100


@dataclass
class Car(Vehicle):
    doors: int = 4


@dataclass
class Bike(Vehicle):
    has_motor: bool = False


@dataclass
class Wrapper(Serializable):
    """Has an optional Serializable field."""

    value: int
    tag: Atom | None = None


# ===========================================================================
# 1. Flat object — to_dict / from_dict
# ===========================================================================


class TestFlatDict:
    @pytest.fixture
    def atom(self) -> Atom:
        return Atom(x=3.0, y=4.0, label="pt")

    def test_to_dict_includes_all_init_fields(self, atom):
        d = atom.to_dict()
        assert "x" in d
        assert "y" in d
        assert "label" in d

    def test_to_dict_excludes_non_init_state(self, atom):
        """Only fields with init=True (plus the discriminator and the `name` property) should appear."""
        d = atom.to_dict()
        keys = set(d.keys()) - {DISCRIMINATOR_KEY, "name"}
        assert keys == {"x", "y", "label"}

    def test_to_dict_values_are_correct(self, atom):
        d = atom.to_dict()
        assert d["x"] == pytest.approx(3.0)
        assert d["y"] == pytest.approx(4.0)
        assert d["label"] == "pt"

    def test_to_dict_contains_discriminator_with_class_name(self, atom):
        d = atom.to_dict()
        assert DISCRIMINATOR_KEY in d
        assert d[DISCRIMINATOR_KEY] == "Atom"

    def test_from_dict_reconstructs_all_fields(self, atom):
        restored = Atom.from_dict(atom.to_dict())
        assert restored.x == pytest.approx(atom.x)
        assert restored.y == pytest.approx(atom.y)
        assert restored.label == atom.label

    def test_round_trip_produces_equal_object(self, atom):
        assert Atom.from_dict(atom.to_dict()) == atom

    def test_from_dict_applies_default_when_field_absent(self):
        d = {"x": 1.0, "y": 2.0, DISCRIMINATOR_KEY: "Atom"}  # label absent
        restored = Atom.from_dict(d)
        assert restored.label == "default"

    def test_from_dict_raises_key_error_for_missing_required_field(self):
        d = {DISCRIMINATOR_KEY: "Atom", "y": 1.0}  # x has no default
        with pytest.raises(KeyError):
            Atom.from_dict(d)

    def test_from_dict_pops_discriminator_from_input_dict(self, atom):
        """from_dict modifies its argument — discriminator is consumed."""
        d = atom.to_dict()
        assert DISCRIMINATOR_KEY in d
        Atom.from_dict(d)
        assert DISCRIMINATOR_KEY not in d

    def test_each_to_dict_call_returns_a_fresh_dict(self, atom):
        d1 = atom.to_dict()
        d2 = atom.to_dict()
        assert d1 is not d2  # distinct objects


# ===========================================================================
# 2. Nested Serializable objects
# ===========================================================================


class TestNestedDict:
    @pytest.fixture
    def molecule(self) -> Molecule:
        return Molecule(atom=Atom(x=1.0, y=2.0, label="inner"), weight=3.5)

    def test_to_dict_serializes_child_as_nested_dict(self, molecule):
        d = molecule.to_dict()
        assert isinstance(d["atom"], dict)

    def test_to_dict_nested_dict_has_discriminator(self, molecule):
        d = molecule.to_dict()
        assert d["atom"][DISCRIMINATOR_KEY] == "Atom"

    def test_to_dict_nested_dict_contains_child_values(self, molecule):
        d = molecule.to_dict()
        assert d["atom"]["x"] == pytest.approx(1.0)
        assert d["atom"]["label"] == "inner"

    def test_from_dict_reconstructs_child_as_serializable(self, molecule):
        restored = Molecule.from_dict(molecule.to_dict())
        assert isinstance(restored.atom, Atom)

    def test_from_dict_preserves_nested_values(self, molecule):
        restored = Molecule.from_dict(molecule.to_dict())
        assert restored.atom.x == pytest.approx(molecule.atom.x)
        assert restored.atom.y == pytest.approx(molecule.atom.y)
        assert restored.atom.label == molecule.atom.label
        assert restored.weight == pytest.approx(molecule.weight)

    def test_json_round_trip_reconstructs_nested_type(self, molecule):
        restored = Molecule.from_json(molecule.to_json())
        assert isinstance(restored.atom, Atom)

    def test_json_round_trip_preserves_nested_values(self, molecule):
        restored = Molecule.from_json(molecule.to_json())
        assert restored.atom.x == pytest.approx(molecule.atom.x)
        assert restored.weight == pytest.approx(molecule.weight)


# ===========================================================================
# 3. Polymorphic dispatch via the discriminator
# ===========================================================================


class TestPolymorphicDispatch:
    def test_base_class_from_dict_dispatches_to_correct_subclass(self):
        d = {DISCRIMINATOR_KEY: "Car", "speed": 120, "doors": 2}
        obj = Vehicle.from_dict(d)
        assert type(obj) is Car
        assert obj.doors == 2

    def test_from_dict_dispatches_to_correct_of_two_subclasses(self):
        car_dict = {"speed": 80, "doors": 4, DISCRIMINATOR_KEY: "Car"}
        bike_dict = {"speed": 30, "has_motor": True, DISCRIMINATOR_KEY: "Bike"}
        car = Vehicle.from_dict(car_dict)
        bike = Vehicle.from_dict(bike_dict)
        assert type(car) is Car
        assert type(bike) is Bike

    def test_subclass_to_dict_contains_own_class_name(self):
        car = Car(speed=90, doors=3)
        d = car.to_dict()
        assert d[DISCRIMINATOR_KEY] == "Car"

    def test_round_trip_via_base_class_preserves_subclass_type(self):
        car = Car(speed=90, doors=3)
        d = car.to_dict()
        restored = Vehicle.from_dict(d)
        assert type(restored) is Car
        assert restored.doors == 3

    def test_calling_from_dict_directly_on_subclass_also_works(self):
        car = Car(speed=110, doors=4)
        d = car.to_dict()
        restored = Car.from_dict(d)
        assert isinstance(restored, Car)
        assert restored.speed == 110

    def test_from_dict_raises_key_error_for_unknown_subclass(self):
        d = {DISCRIMINATOR_KEY: "Hovercraft", "speed": 200}
        with pytest.raises(KeyError):
            Vehicle.from_dict(d)

    def test_json_round_trip_preserves_subclass_type(self):
        bike = Bike(speed=25, has_motor=True)
        restored = Vehicle.from_json(bike.to_json())
        assert type(restored) is Bike
        assert restored.has_motor is True


# ===========================================================================
# 4. Optional (X | None) fields
# ===========================================================================


class TestOptionalFields:
    def test_none_field_serializes_as_none_in_dict(self):
        w = Wrapper(value=5, tag=None)
        assert w.to_dict()["tag"] is None

    def test_none_field_deserializes_back_to_none(self):
        w = Wrapper(value=5, tag=None)
        restored = Wrapper.from_dict(w.to_dict())
        assert restored.tag is None

    def test_non_none_field_serializes_as_nested_dict(self):
        w = Wrapper(value=5, tag=Atom(x=1.0, y=2.0))
        d = w.to_dict()
        assert isinstance(d["tag"], dict)
        assert d["tag"][DISCRIMINATOR_KEY] == "Atom"

    def test_non_none_field_deserializes_as_serializable_instance(self):
        w = Wrapper(value=5, tag=Atom(x=1.0, y=2.0, label="tagged"))
        restored = Wrapper.from_dict(w.to_dict())
        assert isinstance(restored.tag, Atom)
        assert restored.tag.label == "tagged"

    def test_json_round_trip_with_none(self):
        w = Wrapper(value=7, tag=None)
        restored = Wrapper.from_json(w.to_json())
        assert restored.value == 7
        assert restored.tag is None

    def test_json_round_trip_with_non_none(self):
        w = Wrapper(value=3, tag=Atom(x=9.0, y=-1.0))
        restored = Wrapper.from_json(w.to_json())
        assert isinstance(restored.tag, Atom)
        assert restored.tag.x == pytest.approx(9.0)


# ===========================================================================
# 5. JSON (to_json / from_json)
# ===========================================================================


class TestJSON:
    def test_to_json_returns_bytes(self):
        assert isinstance(Atom(1.0, 2.0).to_json(), bytes)

    def test_to_json_is_valid_json(self):
        data = json.loads(Atom(1.0, 2.0).to_json())
        assert isinstance(data, dict)

    def test_to_json_beautify_produces_indented_output(self):
        raw = Atom(1.0, 2.0).to_json(beautify=False)
        pretty = Atom(1.0, 2.0).to_json(beautify=True)
        assert b"\n" in pretty
        assert b"\n" not in raw  # compact form has no newlines

    def test_from_json_round_trip(self):
        atom = Atom(x=-5.5, y=0.0, label="json")
        restored = Atom.from_json(atom.to_json())
        assert restored == atom

    def test_from_json_reconstructs_correct_type(self):
        car = Car(speed=60, doors=2)
        restored = Vehicle.from_json(car.to_json())
        assert type(restored) is Car


# ===========================================================================
# 6. File I/O (to_file / from_file)
# ===========================================================================


class TestFileIO:
    def test_to_file_creates_the_file(self, tmp_path):
        path = tmp_path / "atom.json"
        Atom(1.0, 2.0).to_file(path)
        assert path.exists()

    def test_to_file_writes_valid_json(self, tmp_path):
        path = tmp_path / "atom.json"
        Atom(1.0, 2.0).to_file(path)
        with open(path) as f:
            parsed = json.load(f)
        assert parsed[DISCRIMINATOR_KEY] == "Atom"

    def test_to_file_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "atom.json")
        Atom(1.0, 2.0).to_file(path)
        assert Path(path).exists()

    def test_to_file_creates_one_level_of_missing_parent(self, tmp_path):
        """to_file calls path.parent.mkdir(exist_ok=True) — one missing level only."""
        path = tmp_path / "subdir" / "atom.json"
        Atom(1.0, 2.0).to_file(path)
        assert path.exists()

    def test_from_file_round_trip(self, tmp_path):
        atom = Atom(x=7.0, y=-3.0, label="file")
        path = tmp_path / "atom.json"
        atom.to_file(path)
        restored = Atom.from_file(path)
        assert restored == atom

    def test_from_file_accepts_string_path(self, tmp_path):
        atom = Atom(x=1.0, y=2.0)
        path = tmp_path / "atom.json"
        atom.to_file(str(path))
        restored = Atom.from_file(str(path))
        assert restored == atom

    def test_from_file_restores_subclass_type(self, tmp_path):
        bike = Bike(speed=15, has_motor=False)
        path = tmp_path / "vehicle.json"
        bike.to_file(path)
        restored = Vehicle.from_file(path)
        assert type(restored) is Bike


# ===========================================================================
# 7. DQN trainer — full round-trip
#
# QMLP, ArgMax, TransitionMemory, SoftUpdate and HardUpdate are all proper
# @dataclass Serializables, so both to_dict→from_dict and to_json→from_json
# are expected to succeed completely.
# ===========================================================================


def _make_dqn(*, tau=0.02, memory_size=5_000, batch_size=128, lr=3e-4, optimiser: Literal["rmsprop", "adam"] = "rmsprop"):
    """Utility that builds a concrete DQN instance for testing."""
    from marlenv.catalog import DiscreteMockEnv

    from marl import policy
    from marl.algos.dqn import DQN
    from marl.algos.qtarget_updater import SoftUpdate
    from marl.nn.model_bank import qnetworks

    env = DiscreteMockEnv()
    return DQN(
        qnetwork=qnetworks.from_env(env),
        train_policy=policy.ArgMax(),
        memory_size=memory_size,
        mixer=None,
        lr=lr,
        batch_size=batch_size,
        double_qlearning=False,
        target_updater=SoftUpdate(tau=tau),
        optimiser_type=optimiser,
        gamma=0.95,
    )


class TestDQNToDict:
    """Verify the structure produced by DQN.to_dict()."""

    @pytest.fixture(scope="class")
    def dqn(self):
        return _make_dqn()

    def test_discriminator_is_dqn(self, dqn):
        assert dqn.to_dict()[DISCRIMINATOR_KEY] == "DQN"

    def test_scalar_hyperparameters_are_present(self, dqn):
        d = dqn.to_dict()
        assert d["lr"] == pytest.approx(3e-4)
        assert d["batch_size"] == 128
        assert d["double_qlearning"] is False
        assert d["optimiser_type"] == "rmsprop"
        assert d["gamma"] == pytest.approx(0.95)

    def test_mixer_is_none(self, dqn):
        assert dqn.to_dict()["mixer"] is None

    def test_qnetwork_serialized_with_own_discriminator(self, dqn):
        d = dqn.to_dict()
        assert isinstance(d["qnetwork"], dict)
        assert d["qnetwork"][DISCRIMINATOR_KEY] == "QMLP"

    def test_qnetwork_contains_architecture_fields(self, dqn):
        qd = dqn.to_dict()["qnetwork"]
        assert "n_actions" in qd
        assert "obs_shape" in qd
        assert "hidden_sizes" in qd

    def test_memory_size_is_present(self, dqn):
        assert dqn.to_dict()["memory_size"] == 5_000

    def test_target_updater_serialized_as_soft_update(self, dqn):
        d = dqn.to_dict()
        assert isinstance(d["target_updater"], dict)
        assert d["target_updater"][DISCRIMINATOR_KEY] == "SoftUpdate"

    def test_target_updater_contains_tau(self, dqn):
        d = dqn.to_dict()["target_updater"]
        assert d["tau"] == pytest.approx(0.02)

    def test_train_policy_serialized_as_arg_max(self, dqn):
        d = dqn.to_dict()
        assert isinstance(d["train_policy"], dict)
        assert d["train_policy"][DISCRIMINATOR_KEY] == "ArgMax"


class TestDQNDictRoundTrip:
    """DQN.to_dict() → DQN.from_dict() should restore the full config."""

    @pytest.fixture(scope="class")
    def pair(self):
        from marl.algos.dqn import DQN

        original = _make_dqn()
        restored = DQN.from_dict(original.to_dict())
        return original, restored

    def test_lr_is_preserved(self, pair):
        original, restored = pair
        assert restored.lr == pytest.approx(original.lr)

    def test_batch_size_is_preserved(self, pair):
        original, restored = pair
        assert restored.batch_size == original.batch_size

    def test_double_qlearning_is_preserved(self, pair):
        original, restored = pair
        assert restored.double_qlearning == original.double_qlearning

    def test_optimiser_type_is_preserved(self, pair):
        original, restored = pair
        assert restored.optimiser_type == original.optimiser_type

    def test_gamma_is_preserved(self, pair):
        original, restored = pair
        assert restored.gamma == pytest.approx(original.gamma)

    def test_qnetwork_class_is_preserved(self, pair):
        original, restored = pair
        assert type(restored.qnetwork).__name__ == type(original.qnetwork).__name__

    def test_qnetwork_n_actions_is_preserved(self, pair):
        original, restored = pair
        assert restored.qnetwork.n_actions == original.qnetwork.n_actions

    def test_qnetwork_obs_size_is_preserved(self, pair):
        """obs_size = math.prod(obs_shape) — robust to tuple/list conversion."""
        original, restored = pair
        assert restored.qnetwork.obs_size == original.qnetwork.obs_size

    def test_memory_class_is_preserved(self, pair):
        from marl.models import TransitionMemory

        _, restored = pair
        assert isinstance(restored.memory, TransitionMemory)

    def test_memory_max_size_is_preserved(self, pair):
        original, restored = pair
        assert restored.memory.max_size == original.memory.max_size

    def test_target_updater_class_is_preserved(self, pair):
        from marl.algos.qtarget_updater import SoftUpdate

        _, restored = pair
        assert isinstance(restored.target_updater, SoftUpdate)

    def test_target_updater_tau_is_preserved(self, pair):
        original, restored = pair
        assert restored.target_updater.tau == pytest.approx(original.target_updater.tau)

    def test_train_policy_class_is_preserved(self, pair):
        from marl.policy import ArgMax

        _, restored = pair
        assert isinstance(restored.train_policy, ArgMax)

    def test_trainer_base_class_from_dict_dispatches_to_dqn(self):
        """Trainer.from_dict can reconstruct a DQN via the discriminator."""
        from marl.models import Trainer

        dqn = _make_dqn()
        d = dqn.to_dict()
        restored = Trainer.from_dict(d)
        assert type(restored).__name__ == "DQN"

    def test_hard_update_round_trips_correctly(self):
        from marlenv.catalog import DiscreteMockEnv

        from marl import policy
        from marl.algos.dqn import DQN
        from marl.algos.qtarget_updater import HardUpdate
        from marl.nn.model_bank import qnetworks

        env = DiscreteMockEnv()
        dqn = DQN(
            qnetwork=qnetworks.from_env(env),
            train_policy=policy.ArgMax(),
            memory_size=1_000,
            mixer=None,
            target_updater=HardUpdate(update_period=500),
        )
        restored = DQN.from_dict(dqn.to_dict())
        assert isinstance(restored.target_updater, HardUpdate)
        assert restored.target_updater.update_period == 500


class TestDQNJSONRoundTrip:
    """DQN.to_json() → DQN.from_json() — every field must survive JSON encoding."""

    @pytest.fixture(scope="class")
    def pair(self):
        from marl.algos.dqn import DQN

        original = _make_dqn()
        restored = DQN.from_json(original.to_json())
        return original, restored

    def test_to_json_returns_bytes(self, pair):
        original, _ = pair
        assert isinstance(original.to_json(), bytes)

    def test_to_json_is_valid_json(self, pair):
        original, _ = pair
        parsed = json.loads(original.to_json())
        assert parsed[DISCRIMINATOR_KEY] == "DQN"

    def test_lr_survives_json(self, pair):
        original, restored = pair
        assert restored.lr == pytest.approx(original.lr)

    def test_batch_size_survives_json(self, pair):
        original, restored = pair
        assert restored.batch_size == original.batch_size

    def test_gamma_survives_json(self, pair):
        original, restored = pair
        assert restored.gamma == pytest.approx(original.gamma)

    def test_train_interval_survives_json(self, pair):
        """tuple becomes list after JSON encoding; DQN.__post_init__ handles both."""
        original, restored = pair
        assert list(restored.train_interval) == list(original.train_interval)

    def test_qnetwork_class_survives_json(self, pair):
        original, restored = pair
        assert type(restored.qnetwork).__name__ == type(original.qnetwork).__name__

    def test_qnetwork_architecture_survives_json(self, pair):
        original, restored = pair
        assert restored.qnetwork.n_actions == original.qnetwork.n_actions
        assert restored.qnetwork.obs_size == original.qnetwork.obs_size

    def test_memory_class_survives_json(self, pair):
        from marl.models import TransitionMemory

        _, restored = pair
        assert isinstance(restored.memory, TransitionMemory)

    def test_target_updater_class_survives_json(self, pair):
        from marl.algos.qtarget_updater import SoftUpdate

        _, restored = pair
        assert isinstance(restored.target_updater, SoftUpdate)

    def test_target_updater_tau_survives_json(self, pair):
        original, restored = pair
        assert restored.target_updater.tau == pytest.approx(original.target_updater.tau)


class TestDQNFileRoundTrip:
    @pytest.fixture(scope="class")
    def saved_path(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("dqn") / "dqn_config.json"
        _make_dqn().to_file(path)
        return path

    def test_file_is_created(self, saved_path):
        assert saved_path.exists()

    def test_file_contains_valid_json(self, saved_path):
        with open(saved_path) as f:
            parsed = json.load(f)
        assert parsed[DISCRIMINATOR_KEY] == "DQN"

    def test_from_file_restores_lr(self, saved_path):
        from marl.algos.dqn import DQN

        restored = DQN.from_file(saved_path)
        assert restored.lr == pytest.approx(3e-4)

    def test_from_file_restores_batch_size(self, saved_path):
        from marl.algos.dqn import DQN

        restored = DQN.from_file(saved_path)
        assert restored.batch_size == 128

    def test_from_file_restores_nested_types(self, saved_path):
        from marl.algos.dqn import DQN
        from marl.algos.qtarget_updater import SoftUpdate
        from marl.models import TransitionMemory

        restored = DQN.from_file(saved_path)
        assert isinstance(restored.memory, TransitionMemory)
        assert isinstance(restored.target_updater, SoftUpdate)


def test_experiment_serialization():
    from datetime import datetime

    env = EnvConfig.from_any(DiscreteMockEnv())
    exp = Experiment(
        n_steps=1_000_000,
        logdir="logs/test-experiment-serialization",
        loggers=("csv",),
        creation_timestamp=datetime.now(),
        trainer=algos.DQN(qnetworks.from_env(env), memory_size=50000, mixer=None),
        env=env,
        test_env=env,
    )
    json = exp.to_json()
    restored = Experiment.from_json(json)
    assert isinstance(restored, Experiment)
    assert restored.n_steps == exp.n_steps
    assert type(restored.trainer).__name__ == "DQN"
