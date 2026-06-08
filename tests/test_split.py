"""Tests for the patient-independent data split (split_dataset).

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 15.3
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.config import Config
from src.data.preprocess import split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = [
    "patient_id",
    "recording_id",
    "segment_id",
    "spectrogram_path",
    "disease_class",
]


def _make_df(patient_ids: list[str]) -> pd.DataFrame:
    """Build a minimal DataFrame with the 5 required columns for the given IDs."""
    rows = []
    for i, pid in enumerate(patient_ids):
        rows.append(
            {
                "patient_id": pid,
                "recording_id": f"rec_{pid}_{i}",
                "segment_id": f"seg_{pid}_{i}",
                "spectrogram_path": f"/fake/path/{pid}_{i}.npy",
                "disease_class": "COPD",
            }
        )
    return pd.DataFrame(rows, columns=_REQUIRED_COLS)


def _write_split_def(tmp_dir: Path, train_ids: list[str], test_ids: list[str]) -> Path:
    """Serialise a split-definition JSON file and return its path."""
    split_def = tmp_dir / "split_def.json"
    split_def.write_text(json.dumps({"train": train_ids, "test": test_ids}))
    return split_def


def _make_config(tmp_dir: Path) -> Config:
    """Return a Config whose splits_dir points inside *tmp_dir*."""
    cfg = Config()
    cfg.splits_dir = tmp_dir / "splits"
    return cfg


# ---------------------------------------------------------------------------
# Property 8: Patient-Independent Split No-Overlap
# Validates: Requirements 4.1, 4.2, 15.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    train_ids=st.sets(
        st.text(
            min_size=1,
            max_size=5,
            alphabet=st.characters(whitelist_categories=("Nd",)),
        ),
        min_size=1,
        max_size=10,
    ),
    test_ids=st.sets(
        st.text(
            min_size=1,
            max_size=5,
            alphabet=st.characters(whitelist_categories=("Nd",)),
        ),
        min_size=1,
        max_size=10,
    ),
)
def test_patient_independent_split_no_overlap(
    train_ids: set[str], test_ids: set[str]
) -> None:
    """Property 8: Patient-Independent Split No-Overlap

    For any disjoint pair of train/test patient-ID sets, split_dataset must
    produce two result DataFrames whose patient_id columns share no elements.

    Validates: Requirements 4.1, 4.2, 15.3
    """
    assume(train_ids.isdisjoint(test_ids))

    all_ids = list(train_ids | test_ids)
    df = _make_df(all_ids)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        split_def = _write_split_def(tmp_dir, list(train_ids), list(test_ids))
        config = _make_config(tmp_dir)

        train_df, test_df = split_dataset(df, split_def, config)

    train_pids = set(train_df["patient_id"].astype(str))
    test_pids = set(test_df["patient_id"].astype(str))

    assert train_pids.isdisjoint(test_pids), (
        f"Overlap detected between train and test patient IDs: "
        f"{train_pids & test_pids}"
    )


def test_split_raises_assertion_error_on_overlap(tmp_path: Path) -> None:
    """split_dataset must raise AssertionError when train/test IDs overlap.

    Validates: Requirements 4.2, 15.3
    """
    overlapping_id = "101"
    train_ids = [overlapping_id, "102", "103"]
    test_ids = [overlapping_id, "104", "105"]

    df = _make_df(train_ids + [pid for pid in test_ids if pid != overlapping_id])
    split_def = _write_split_def(tmp_path, train_ids, test_ids)
    config = _make_config(tmp_path)

    with pytest.raises(AssertionError) as exc_info:
        split_dataset(df, split_def, config)

    error_message = str(exc_info.value)
    assert overlapping_id in error_message, (
        f"Expected overlapping ID '{overlapping_id}' in error message, "
        f"got: {error_message!r}"
    )


# ---------------------------------------------------------------------------
# Property 9: Split CSV Column Schema
# Validates: Requirement 4.5
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_train=st.integers(min_value=1, max_value=5),
    n_test=st.integers(min_value=1, max_value=5),
)
def test_split_csv_column_schema(n_train: int, n_test: int) -> None:
    """Property 9: Split CSV Column Schema

    Both train.csv and test.csv must contain exactly the five required columns:
    patient_id, recording_id, segment_id, spectrogram_path, disease_class.

    Validates: Requirement 4.5
    """
    # Build disjoint patient IDs: train uses 1..n_train, test uses 1001..1000+n_test
    train_ids = [str(i) for i in range(1, n_train + 1)]
    test_ids = [str(1000 + i) for i in range(1, n_test + 1)]

    all_ids = train_ids + test_ids
    df = _make_df(all_ids)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        split_def = _write_split_def(tmp_dir, train_ids, test_ids)
        config = _make_config(tmp_dir)

        split_dataset(df, split_def, config)

        train_csv_path = config.splits_dir / "train.csv"
        test_csv_path = config.splits_dir / "test.csv"

        assert train_csv_path.exists(), "train.csv was not written"
        assert test_csv_path.exists(), "test.csv was not written"

        train_read = pd.read_csv(train_csv_path)
        test_read = pd.read_csv(test_csv_path)

    assert list(train_read.columns) == _REQUIRED_COLS, (
        f"train.csv columns mismatch. Expected {_REQUIRED_COLS}, "
        f"got {list(train_read.columns)}"
    )
    assert list(test_read.columns) == _REQUIRED_COLS, (
        f"test.csv columns mismatch. Expected {_REQUIRED_COLS}, "
        f"got {list(test_read.columns)}"
    )
