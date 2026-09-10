"""Tests for marl.models.dataset.Dataset.nice_label."""

import numpy as np

from marl.models.dataset import Dataset


def _make_dataset(label: str) -> Dataset:
    arr = np.zeros(3, dtype=np.float32)
    return Dataset(logdir="logs/x", ticks=[0.0, 1.0, 2.0], label=label, category="cat", mean=arr, min=arr, max=arr, std=arr, ci95=arr)


def test_nice_label_replaces_underscores_with_spaces_and_capitalizes():
    assert _make_dataset("exit_rate").nice_label == "Exit rate"


def test_nice_label_replaces_hyphens_with_spaces():
    assert _make_dataset("score-0").nice_label == "Score 0"


def test_nice_label_capitalizes_single_word():
    assert _make_dataset("reward").nice_label == "Reward"


def test_nice_label_lowercases_remaining_letters():
    """`str.capitalize()` also lowercases the rest of the string."""
    assert _make_dataset("MEAN_REWARD").nice_label == "Mean reward"
