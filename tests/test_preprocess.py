"""Tests for src/data/preprocess.py — annotation parsing and label mapping.

Covers:
  - Property 1: Annotation Parsing Round-Trip Integrity (Task 3.4, Req 2.1)
  - Property 2: Label Mapping Completeness (Task 3.5, Req 2.3, 2.4)
  - Property 3: Null Metadata Substitution Completeness (Task 3.6, Req 2.5)
  - Unit tests for unmatched patient_id exclusion (Task 3.7, Req 2.2)
  - Unit tests for malformed row skipping (Task 3.7, Req 2.6)
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.preprocess import map_label, parse_icbhi_annotations, save_metadata
from src.models.types import DiseaseClass

# ---------------------------------------------------------------------------
# Known ICBHI label vocabulary
# ---------------------------------------------------------------------------

KNOWN_LABELS = [
    "COPD",
    "Healthy",
    "URTI",
    "Bronchiectasis",
    "Pneumonia",
    "Bronchiolitis",
    "Asthma",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_annotation_file(path: Path, rows: list[tuple]) -> None:
    """Write an ICBHI-style tab-separated annotation file at *path*.

    Args:
        path: Destination file path.
        rows: Each tuple is (start_time, end_time, crackle_label, wheeze_label).
    """
    lines = [f"{st}\t{et}\t{cl}\t{wl}" for st, et, cl, wl in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_diagnosis_file(path: Path, mapping: dict[str, str]) -> None:
    """Write an ICBHI_diagnosis.txt file at *path*.

    Args:
        path: Destination file path.
        mapping: patient_id → diagnosis string.
    """
    lines = [f"{pid}\t{diag}" for pid, diag in mapping.items()]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Property 1: Annotation Parsing Round-Trip Integrity
# Feature: lung-disease-management, Property 1: Annotation Parsing Round-Trip Integrity
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    rows=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=1),
            st.integers(min_value=0, max_value=1),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_annotation_parsing_round_trip(
    rows: list[tuple],
) -> None:
    """Parsed DataFrame values must exactly match source annotation values.

    **Validates: Requirements 2.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        patient_id = "101"
        diagnosis = "COPD"

        # Write the annotation file using the expected ICBHI naming convention
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_file, rows)

        # Write the diagnosis cross-reference file
        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: diagnosis})

        result = parse_icbhi_annotations(tmp_path)

        assert len(result) == len(rows), (
            f"Expected {len(rows)} rows, got {len(result)}"
        )

        for i, (start_time, end_time, crackle_label, wheeze_label) in enumerate(rows):
            row = result.iloc[i]
            # Float comparison — parsing round-trip must be exact for finite values
            assert math.isclose(row["start_time"], start_time, rel_tol=1e-9, abs_tol=1e-12), (
                f"Row {i}: start_time mismatch: {row['start_time']} vs {start_time}"
            )
            assert math.isclose(row["end_time"], end_time, rel_tol=1e-9, abs_tol=1e-12), (
                f"Row {i}: end_time mismatch: {row['end_time']} vs {end_time}"
            )
            assert row["crackle_label"] == crackle_label, (
                f"Row {i}: crackle_label mismatch"
            )
            assert row["wheeze_label"] == wheeze_label, (
                f"Row {i}: wheeze_label mismatch"
            )
            assert row["patient_id"] == patient_id
            assert row["diagnosis"] == diagnosis


# ---------------------------------------------------------------------------
# Property 2: Label Mapping Completeness
# Feature: lung-disease-management, Property 2: Label Mapping Completeness
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(st.sampled_from(KNOWN_LABELS))
def test_known_labels_return_disease_class(label: str) -> None:
    """Every official ICBHI label must map to a DiseaseClass instance.

    **Validates: Requirements 2.3, 2.4**
    """
    result = map_label(label)
    assert isinstance(result, DiseaseClass), (
        f"Expected DiseaseClass for known label '{label}', got {result!r}"
    )


@settings(max_examples=100)
@given(st.text().filter(lambda s: s not in KNOWN_LABELS))
def test_unknown_labels_return_none(label: str) -> None:
    """Any string not in the official ICBHI vocabulary must return None.

    **Validates: Requirements 2.3, 2.4**
    """
    result = map_label(label)
    assert result is None, (
        f"Expected None for unknown label '{label}', got {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: Null Metadata Substitution Completeness
# Feature: lung-disease-management, Property 3: Null Metadata Substitution Completeness
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    record=st.fixed_dictionaries(
        {
            "patient_id": st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cc", "Cs"))),
            "age": st.one_of(
                st.none(),
                st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False),
            ),
            "sex": st.one_of(st.none(), st.sampled_from(["M", "F"])),
            "bmi": st.one_of(
                st.none(),
                st.floats(min_value=10, max_value=60, allow_nan=False, allow_infinity=False),
            ),
            "smoking_status": st.one_of(
                st.none(), st.sampled_from(["never", "former", "current"])
            ),
            "recording_location": st.one_of(st.none(), st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cc", "Cs")))),
        }
    )
)
def test_null_metadata_substitution(record: dict) -> None:
    """Missing metadata fields must be stored as NaN/null in the CSV, not as strings.

    **Validates: Requirements 2.5**
    """
    df = pd.DataFrame([record])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)

    # String-typed columns — must be reloaded as str to preserve zero-padded
    # strings (e.g. "00") that CSV type inference would otherwise coerce to int.
    STRING_COLS = {"patient_id", "sex", "smoking_status", "recording_location"}
    reload_dtype = {col: str for col in STRING_COLS}

    try:
        save_metadata(df, path)
        loaded = pd.read_csv(path, dtype=reload_dtype)

        for col in ["age", "sex", "bmi", "smoking_status", "recording_location"]:
            if record[col] is None:
                # Null values must be stored as NaN (empty cell), NOT as "null"/"None"
                assert pd.isna(loaded[col].iloc[0]), (
                    f"Column '{col}': expected NaN for None value, "
                    f"got {loaded[col].iloc[0]!r}"
                )
            else:
                # Present values must be preserved accurately.
                # Floats may lose insignificant trailing digits in CSV round-trip
                # so we compare with a relative tolerance for numeric fields.
                actual = loaded[col].iloc[0]
                expected = record[col]
                if isinstance(expected, float):
                    assert math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-12), (
                        f"Column '{col}': expected {expected!r}, got {actual!r}"
                    )
                else:
                    assert actual == expected or str(actual) == str(expected), (
                        f"Column '{col}': expected {expected!r}, got {actual!r}"
                    )
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Unit tests — annotation parsing (Task 3.7)
# ---------------------------------------------------------------------------


class TestUnmatchedPatientIdExcludesSegments:
    """Verifies unmatched patient_id warning and segment exclusion (Req 2.2)."""

    def test_segments_excluded_for_unmatched_patient(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Segments for a patient_id not in ICBHI_diagnosis.txt must be absent from result."""
        # Patient 999 is NOT in the diagnosis file
        ann_file = tmp_path / "999_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_file, [(0.0, 1.0, 0, 0), (1.0, 2.0, 1, 0)])

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {"101": "COPD"})  # patient 999 not here

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            result = parse_icbhi_annotations(tmp_path)

        # No rows for patient 999 should appear
        assert len(result) == 0 or (result["patient_id"] != "999").all(), (
            "Segments for unmatched patient 999 must be excluded from the result."
        )

    def test_warning_logged_for_unmatched_patient(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning must be logged that identifies the unmatched patient_id."""
        ann_file = tmp_path / "999_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_file, [(0.0, 1.0, 0, 0)])

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {"101": "COPD"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            parse_icbhi_annotations(tmp_path)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("999" in str(m) for m in warning_messages), (
            "Expected a warning message identifying patient_id '999'."
        )

    def test_matched_patients_still_included(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Segments for valid patients must appear in the result even when other patients fail."""
        # Patient 999 is invalid, patient 101 is valid
        ann_invalid = tmp_path / "999_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_invalid, [(0.0, 1.0, 0, 0)])

        ann_valid = tmp_path / "101_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_valid, [(0.0, 2.0, 1, 0), (2.0, 4.0, 0, 1)])

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {"101": "Healthy"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            result = parse_icbhi_annotations(tmp_path)

        assert len(result) == 2
        assert (result["patient_id"] == "101").all()

    def test_empty_dataframe_returned_when_all_patients_unmatched(
        self, tmp_path: Path
    ) -> None:
        """Empty DataFrame with correct columns must be returned when no patient matches."""
        ann_file = tmp_path / "999_1_al_sc_Meditron.txt"
        _write_annotation_file(ann_file, [(0.0, 1.0, 0, 0)])

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {"200": "COPD"})

        result = parse_icbhi_annotations(tmp_path)

        assert len(result) == 0
        expected_cols = {
            "start_time", "end_time", "crackle_label", "wheeze_label",
            "patient_id", "recording_id", "diagnosis",
        }
        assert set(result.columns) == expected_cols


class TestMalformedRowSkipped:
    """Verifies malformed row skipping with correct file/row info in warning (Req 2.6)."""

    def test_malformed_row_excluded_valid_row_present(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Result must contain only the valid row; the malformed row is skipped."""
        patient_id = "101"
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"

        # Row 1: valid; Row 2: malformed (only 2 columns)
        content = "0.0\t1.0\t0\t1\nBAD_ROW only_two\n"
        ann_file.write_text(content, encoding="utf-8")

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: "COPD"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            result = parse_icbhi_annotations(tmp_path)

        assert len(result) == 1, f"Expected 1 valid row, got {len(result)}"
        row = result.iloc[0]
        assert row["start_time"] == 0.0
        assert row["end_time"] == 1.0
        assert row["crackle_label"] == 0
        assert row["wheeze_label"] == 1

    def test_warning_logged_with_file_name(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning message must include the annotation file name."""
        patient_id = "101"
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"
        content = "0.0\t1.0\t0\t1\nBAD\n"
        ann_file.write_text(content, encoding="utf-8")

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: "COPD"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            parse_icbhi_annotations(tmp_path)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        file_name = ann_file.name
        assert any(file_name in str(m) for m in warning_messages), (
            f"Expected warning containing file name '{file_name}' in: {warning_messages}"
        )

    def test_warning_logged_with_row_number(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning message must include the 1-indexed row number of the malformed row."""
        patient_id = "101"
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"
        # Row 1 valid, row 2 malformed — warning should mention row 2
        content = "0.0\t1.0\t0\t1\nBAD_ROW_DATA\n"
        ann_file.write_text(content, encoding="utf-8")

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: "COPD"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            parse_icbhi_annotations(tmp_path)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        # Row 2 (1-indexed) should appear in a warning
        assert any("2" in str(m) for m in warning_messages), (
            f"Expected warning with row number '2', got: {warning_messages}"
        )

    def test_unparseable_values_trigger_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rows with non-numeric values in numeric columns must trigger a warning."""
        patient_id = "101"
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"
        # Row 1 valid; Row 2 has valid column count but non-numeric start_time
        content = "0.0\t1.0\t0\t1\nABC\t1.0\t0\t1\n"
        ann_file.write_text(content, encoding="utf-8")

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: "COPD"})

        with caplog.at_level(logging.WARNING, logger="src.data.preprocess"):
            result = parse_icbhi_annotations(tmp_path)

        # Only the valid row should be in the result
        assert len(result) == 1
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) >= 1

    def test_all_malformed_rows_skipped_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """When all rows are malformed, an empty DataFrame with correct columns is returned."""
        patient_id = "101"
        ann_file = tmp_path / f"{patient_id}_1_al_sc_Meditron.txt"
        content = "BAD\nALSO_BAD\n"
        ann_file.write_text(content, encoding="utf-8")

        diag_file = tmp_path / "ICBHI_diagnosis.txt"
        _write_diagnosis_file(diag_file, {patient_id: "COPD"})

        result = parse_icbhi_annotations(tmp_path)

        assert len(result) == 0
        expected_cols = {
            "start_time", "end_time", "crackle_label", "wheeze_label",
            "patient_id", "recording_id", "diagnosis",
        }
        assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# Audio Signal Processing Property Tests — Tasks 4.5–4.8
# ---------------------------------------------------------------------------

import numpy as np
import scipy.io.wavfile

from src.config import Config
from src.data.preprocess import (
    apply_bandpass_filter,
    preprocess_audio,
    resample_audio,
    segment_audio,
)

_CONFIG = Config()


# ---------------------------------------------------------------------------
# Property 4: Resampling Invariant
# Feature: lung-disease-management, Property 4: Resampling Invariant
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=4000, max_value=48000))
def test_resampling_invariant(orig_sr: int) -> None:
    """Output length must equal config.target_sample_rate for any 1-second input.

    **Validates: Requirements 3.1**
    """
    config = _CONFIG
    # Generate 1 second of audio at orig_sr
    audio = np.random.default_rng(42).uniform(-1.0, 1.0, orig_sr).astype(np.float32)
    resampled = resample_audio(audio, orig_sr, config)
    assert len(resampled) == config.target_sample_rate, (
        f"Expected {config.target_sample_rate} samples, got {len(resampled)} "
        f"(orig_sr={orig_sr})"
    )


# ---------------------------------------------------------------------------
# Property 5: Bandpass Filter Frequency Invariant
# Feature: lung-disease-management, Property 5: Bandpass Filter Frequency Invariant
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    freq=st.floats(
        min_value=100.0,
        max_value=1500.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_bandpass_passband_signal_preserved(freq: float) -> None:
    """Passband sinusoids (100–1500 Hz) must retain > 50% of input power.

    **Validates: Requirements 3.2**
    """
    config = _CONFIG
    sr = config.target_sample_rate  # 4000 Hz
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    filtered = apply_bandpass_filter(audio, config)

    input_power = float(np.mean(audio ** 2))
    output_power = float(np.mean(filtered ** 2))

    assert output_power > input_power * 0.5, (
        f"Passband signal at {freq:.1f} Hz attenuated too much: "
        f"input_power={input_power:.6f}, output_power={output_power:.6f}"
    )


@settings(max_examples=100)
@given(
    freq=st.floats(
        min_value=1.0,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_bandpass_stopband_signal_attenuated(freq: float) -> None:
    """Stopband sinusoids (1–20 Hz, well below 50 Hz cutoff) must be attenuated to < 10% input power.

    **Validates: Requirements 3.2**
    """
    config = _CONFIG
    sr = config.target_sample_rate  # 4000 Hz
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    filtered = apply_bandpass_filter(audio, config)

    input_power = float(np.mean(audio ** 2))
    output_power = float(np.mean(filtered ** 2))

    assert output_power < input_power * 0.1, (
        f"Stopband signal at {freq:.1f} Hz not attenuated enough: "
        f"input_power={input_power:.6f}, output_power={output_power:.6f}"
    )


# ---------------------------------------------------------------------------
# Property 6: Segmentation Uniformity
# Feature: lung-disease-management, Property 6: Segmentation Uniformity
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=80000))
def test_segmentation_uniformity(num_samples: int) -> None:
    """Every segment must be exactly window_size (20000) samples.

    For recordings shorter than 5 seconds (< 20000 samples), exactly one
    zero-padded segment is produced.

    **Validates: Requirements 3.4**
    """
    config = _CONFIG
    window_size = int(config.segment_duration * config.target_sample_rate)  # 20000

    audio = np.zeros(num_samples, dtype=np.float32)
    segments = segment_audio(audio, config)

    # Every segment must have exactly window_size samples
    for i, seg in enumerate(segments):
        assert len(seg) == window_size, (
            f"Segment {i} has {len(seg)} samples, expected {window_size} "
            f"(num_samples={num_samples})"
        )

    # Recordings shorter than window_size must produce exactly 1 segment
    if num_samples < window_size:
        assert len(segments) == 1, (
            f"Expected exactly 1 segment for short recording ({num_samples} samples), "
            f"got {len(segments)}"
        )


# ---------------------------------------------------------------------------
# Property 7: Spectrogram Output Shape Invariant
# Feature: lung-disease-management, Property 7: Spectrogram Output Shape Invariant
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(st.just(None))
def test_spectrogram_output_shape(dummy: None) -> None:
    """preprocess_audio output must have shape (3, 128, 216) and dtype float32.

    **Validates: Requirements 3.5, 3.6**
    """
    config = _CONFIG
    sr = config.target_sample_rate  # 4000 Hz
    window_size = int(config.segment_duration * sr)  # 20000 samples = 5 s

    # Create a 5-second audio segment with a known signal
    audio = np.sin(
        2 * np.pi * 440.0 * np.arange(window_size, dtype=np.float32) / sr
    ).astype(np.float32)

    # Write to a temporary WAV file (int16 for compatibility)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)

    try:
        scipy.io.wavfile.write(str(wav_path), sr, audio_int16)
        result = preprocess_audio(wav_path, config)
    finally:
        wav_path.unlink(missing_ok=True)

    expected_shape = (3, config.n_mels, config.n_time_frames)  # (3, 128, 216)
    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result.shape}"
    )
    assert result.dtype == np.float32, (
        f"Expected dtype float32, got {result.dtype}"
    )
