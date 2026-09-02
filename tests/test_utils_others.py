"""Tests for marl.utils.others: defaults_to, alpha_num_order, hash_ndarray, obs_to_hashes, seed."""

import random

import numpy as np
import torch
from marlenv.catalog import DiscreteMockEnv

from marl.utils.others import alpha_num_order, defaults_to, hash_ndarray, obs_to_hashes, seed


class TestDefaultsTo:
    def test_returns_the_value_when_not_none(self):
        assert defaults_to(5, lambda: 10) == 5

    def test_returns_the_default_when_none(self):
        assert defaults_to(None, lambda: 10) == 10

    def test_does_not_call_the_default_factory_when_value_is_given(self):
        calls = []

        def factory():
            calls.append(1)
            return 0

        defaults_to(3, factory)
        assert calls == []


class TestAlphaNumOrder:
    def test_pads_numbers_to_a_fixed_width(self):
        assert alpha_num_order("a6b12.125") == "a" + "0" * 7 + "6" + "b" + "0" * 6 + "12" + "." + "0" * 5 + "125"

    def test_sorting_strings_with_numbers_is_numeric_not_lexicographic(self):
        strings = ["item2", "item10", "item1"]
        sorted_strings = sorted(strings, key=alpha_num_order)
        assert sorted_strings == ["item1", "item2", "item10"]

    def test_string_without_digits_is_unchanged(self):
        assert alpha_num_order("hello") == "hello"


class TestHashNdarray:
    def test_same_content_gives_same_hash(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        assert hash_ndarray(a) == hash_ndarray(b)

    def test_different_content_gives_different_hash(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([1, 2, 4], dtype=np.float32)
        assert hash_ndarray(a) != hash_ndarray(b)

    def test_returns_an_int(self):
        assert isinstance(hash_ndarray(np.zeros(3)), int)


class TestObsToHashes:
    def test_returns_one_hash_per_agent(self):
        env = DiscreteMockEnv()
        obs, _ = env.reset()
        hashes = obs_to_hashes(obs)
        assert len(hashes) == obs.data.shape[0]

    def test_identical_observations_give_identical_hashes(self):
        env = DiscreteMockEnv()
        obs, _ = env.reset()
        assert obs_to_hashes(obs) == obs_to_hashes(obs)


class TestSeed:
    def test_seeding_makes_random_reproducible(self):
        seed(42)
        a = random.random()
        seed(42)
        b = random.random()
        assert a == b

    def test_seeding_makes_numpy_reproducible(self):
        seed(42)
        a = np.random.rand(5)
        seed(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_seeding_makes_torch_reproducible(self):
        seed(42)
        a = torch.rand(5)
        seed(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_seeds_provided_environments(self):
        env = DiscreteMockEnv()
        calls = []
        env.seed = lambda s: calls.append(s)
        seed(7, env)
        assert calls == [7]
