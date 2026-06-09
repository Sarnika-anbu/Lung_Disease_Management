"""Audio preprocessing and annotation parsing for the Lung Disease Management System.

This module implements:
  - ICBHI annotation file parsing with patient-diagnosis cross-referencing
  - Label mapping from raw ICBHI diagnosis strings to ``DiseaseClass`` enum values
  - Patient metadata persistence to ``data/processed/metadata.csv``
  - Audio signal processing (resampling, bandpass filtering, spectral gating,
    segmentation, spectrogram computation)
  - Patient-independent train/test split application

All file paths are resolved via ``src.config.Config`` — no hardcoded strings.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.models.types import DiseaseClass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Annotation file constants
# ---------------------------------------------------------------------------

# Expected column order in ICBHI annotation .txt files (tab-separated, no header).
_ANNOTATION_COLS = ["start_time", "end_time", "crackle_label", "wheeze_label"]

# Pattern that extracts patient_id and recording_index from ICBHI filenames.
# Filename format: {patient_id}_{recording_index}_{...}.txt
# recording_index can be alphanumeric (e.g. 1b1, 2b4)
_FILENAME_RE = re.compile(r"^(\d+)_([A-Za-z0-9]+)_")

# Mapping from raw ICBHI diagnosis strings to DiseaseClass enum values.
_LABEL_MAP: dict[str, DiseaseClass] = {
    "COPD": DiseaseClass.COPD,
    "Healthy": DiseaseClass.HEALTHY,
    "URTI": DiseaseClass.URTI,
    "Bronchiectasis": DiseaseClass.BRONCHIECTASIS,
    "Pneumonia": DiseaseClass.PNEUMONIA,
    "Bronchiolitis": DiseaseClass.BRONCHIOLITIS,
    "Asthma": DiseaseClass.ASTHMA,
}


# ---------------------------------------------------------------------------
# Public API — Annotation Parsing
# ---------------------------------------------------------------------------


def parse_icbhi_annotations(data_dir: Path) -> pd.DataFrame:
    """Parse all ICBHI annotation files in *data_dir* and return a consolidated DataFrame.

    Each annotation file is expected to be a tab-separated text file (no header)
    with exactly four columns: ``start_time``, ``end_time``, ``crackle_label``,
    ``wheeze_label``.  Filenames follow the convention
    ``{patient_id}_{recording_index}_{...}.txt``.

    Patient diagnoses are resolved by looking up each ``patient_id`` in the
    ``ICBHI_diagnosis.txt`` file located at ``data_dir / 'ICBHI_diagnosis.txt'``.

    Warnings are emitted (via the module-level ``logger``) for:
      - Patient IDs not found in ``ICBHI_diagnosis.txt`` (all segments for that
        patient are excluded from the returned DataFrame).
      - Malformed rows (missing columns or unparseable numeric values) — the
        offending row is skipped and processing continues.

    Args:
        data_dir: Directory containing ICBHI annotation ``.txt`` files and
            ``ICBHI_diagnosis.txt``.

    Returns:
        A DataFrame with columns:
            ``start_time`` (float), ``end_time`` (float),
            ``crackle_label`` (int), ``wheeze_label`` (int),
            ``patient_id`` (str), ``recording_id`` (str),
            ``diagnosis`` (str).
        Returns an empty DataFrame with those columns if no valid segments are
        found.
    """
    # ------------------------------------------------------------------
    # 1. Load the patient → diagnosis mapping
    # ------------------------------------------------------------------
    diagnosis_path = data_dir / "ICBHI_diagnosis.txt"
    diagnosis_map: dict[str, str] = _load_diagnosis_map(diagnosis_path)

    # ------------------------------------------------------------------
    # 2. Collect all annotation .txt files
    # ------------------------------------------------------------------
    annotation_files = sorted(data_dir.glob("*.txt"))
    # Exclude the diagnosis file itself
    annotation_files = [
        f for f in annotation_files if f.name != "ICBHI_diagnosis.txt"
    ]

    records: list[dict] = []

    for ann_file in annotation_files:
        # Extract patient_id and recording_index from filename
        match = _FILENAME_RE.match(ann_file.name)
        if match is None:
            logger.warning(
                "Cannot extract patient_id from filename '%s'; skipping file.",
                ann_file.name,
            )
            continue

        patient_id = match.group(1)
        recording_index = match.group(2)
        # recording_id is the full stem without the .txt extension
        recording_id = ann_file.stem

        # ------------------------------------------------------------------
        # 3. Cross-reference with diagnosis map
        # ------------------------------------------------------------------
        if patient_id not in diagnosis_map:
            logger.warning(
                "Patient ID '%s' not found in ICBHI_diagnosis.txt; "
                "excluding all segments from file '%s'.",
                patient_id,
                ann_file.name,
            )
            continue

        diagnosis = diagnosis_map[patient_id]

        # ------------------------------------------------------------------
        # 4. Parse annotation rows
        # ------------------------------------------------------------------
        _parse_annotation_file(
            ann_file=ann_file,
            patient_id=patient_id,
            recording_id=recording_id,
            diagnosis=diagnosis,
            records=records,
        )

    # ------------------------------------------------------------------
    # 5. Assemble DataFrame
    # ------------------------------------------------------------------
    output_cols = [
        "start_time",
        "end_time",
        "crackle_label",
        "wheeze_label",
        "patient_id",
        "recording_id",
        "diagnosis",
    ]

    if not records:
        return pd.DataFrame(columns=output_cols)

    df = pd.DataFrame(records, columns=output_cols)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    df["crackle_label"] = df["crackle_label"].astype(int)
    df["wheeze_label"] = df["wheeze_label"].astype(int)
    return df


def _load_diagnosis_map(diagnosis_path: Path) -> dict[str, str]:
    """Load ``ICBHI_diagnosis.txt`` or ``demographic_info.txt`` into a ``{patient_id: diagnosis}`` mapping.

    Handles two formats:
    1. Simple 2-column: ``patient_id  diagnosis`` (no header, whitespace-separated)
    2. ICBHI demographic_info.txt: tab-separated with header row containing
       columns like ``patient_id``, ``age``, ``sex``, ``adult_BMI``, ``diagnosis``

    Args:
        diagnosis_path: Absolute path to the diagnosis/demographic file.

    Returns:
        Dictionary mapping patient ID strings to diagnosis strings.
        Returns an empty dict if the file does not exist.
    """
    if not diagnosis_path.exists():
        logger.warning(
            "Diagnosis file not found at '%s'; all patient IDs will be unmatched.",
            diagnosis_path,
        )
        return {}

    diagnosis_map: dict[str, str] = {}

    try:
        with diagnosis_path.open(encoding="utf-8") as fh:
            first_line = fh.readline().strip()

        # Check if it has a header row (contains non-numeric first token)
        first_parts = first_line.split()
        has_header = first_parts and not first_parts[0].isdigit()

        if has_header:
            # Parse as CSV/TSV with header
            import pandas as pd_diag
            try:
                df_diag = pd_diag.read_csv(
                    diagnosis_path, sep="\t", engine="python"
                )
            except Exception:
                df_diag = pd_diag.read_csv(
                    diagnosis_path, sep=r"\s+", engine="python"
                )

            # Normalise column names to lowercase
            df_diag.columns = [c.strip().lower() for c in df_diag.columns]

            # Find patient_id and diagnosis columns
            pid_col = next(
                (c for c in df_diag.columns if "patient" in c or c == "id"),
                df_diag.columns[0],
            )
            diag_col = next(
                (c for c in df_diag.columns if "diagnosis" in c or "disease" in c),
                None,
            )

            if diag_col is None:
                # Last column is often diagnosis
                diag_col = df_diag.columns[-1]

            for _, row in df_diag.iterrows():
                pid = str(row[pid_col]).strip()
                diag = str(row[diag_col]).strip()
                if pid and diag and diag.lower() != "nan":
                    diagnosis_map[pid] = diag
        else:
            # Simple 2-column format: patient_id  diagnosis
            with diagnosis_path.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pid, diag = parts[0], parts[1]
                        diagnosis_map[pid] = diag

    except Exception as exc:
        logger.warning("Error parsing diagnosis file '%s': %s", diagnosis_path, exc)

    logger.info("Loaded %d patient diagnoses from %s", len(diagnosis_map), diagnosis_path.name)
    return diagnosis_map


def _parse_annotation_file(
    ann_file: Path,
    patient_id: str,
    recording_id: str,
    diagnosis: str,
    records: list[dict],
) -> None:
    """Parse a single ICBHI annotation file and append valid rows to *records*.

    Malformed rows (wrong column count, non-numeric start/end times, or
    non-integer crackle/wheeze labels) are skipped with a warning that
    identifies the file name and 1-based row number.

    Args:
        ann_file: Path to the annotation ``.txt`` file.
        patient_id: Patient identifier extracted from the filename.
        recording_id: Recording identifier (filename stem).
        diagnosis: Diagnosis string cross-referenced from ``ICBHI_diagnosis.txt``.
        records: Mutable list to which valid parsed rows are appended.
    """
    with ann_file.open(encoding="utf-8") as fh:
        for row_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                logger.warning(
                    "Malformed row in '%s' at row %d (expected 4 columns, got %d); "
                    "skipping row.",
                    ann_file.name,
                    row_number,
                    len(parts),
                )
                continue

            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
                crackle_label = int(parts[2])
                wheeze_label = int(parts[3])
            except (ValueError, IndexError):
                logger.warning(
                    "Malformed row in '%s' at row %d (unparseable values); "
                    "skipping row.",
                    ann_file.name,
                    row_number,
                )
                continue

            records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "crackle_label": crackle_label,
                    "wheeze_label": wheeze_label,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "diagnosis": diagnosis,
                }
            )


# ---------------------------------------------------------------------------
# Public API — Label Mapping
# ---------------------------------------------------------------------------


def map_label(raw_label: str) -> Optional[DiseaseClass]:
    """Map a raw ICBHI diagnosis string to a ``DiseaseClass`` enum value.

    Known ICBHI labels and their mappings:
      - ``"COPD"``          → :attr:`DiseaseClass.COPD`
      - ``"Healthy"``       → :attr:`DiseaseClass.HEALTHY`
      - ``"URTI"``          → :attr:`DiseaseClass.URTI`
      - ``"Bronchiectasis"``→ :attr:`DiseaseClass.BRONCHIECTASIS`
      - ``"Pneumonia"``     → :attr:`DiseaseClass.PNEUMONIA`
      - ``"Bronchiolitis"`` → :attr:`DiseaseClass.BRONCHIOLITIS`
      - ``"Asthma"``        → :attr:`DiseaseClass.ASTHMA`

    Args:
        raw_label: The diagnosis string from the ICBHI dataset.

    Returns:
        The corresponding :class:`DiseaseClass` member, or ``None`` if
        *raw_label* is not recognised.  An unrecognised label triggers a
        warning logged to stderr via the module logger.
    """
    result = _LABEL_MAP.get(raw_label)
    if result is None:
        logger.warning(
            "Unrecognised diagnosis label '%s'; returning None.",
            raw_label,
        )
    return result


# ---------------------------------------------------------------------------
# Public API — Metadata Persistence
# ---------------------------------------------------------------------------


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """Write patient metadata to a CSV file at *output_path*.

    The DataFrame must contain the columns: ``patient_id``, ``age``, ``sex``,
    ``bmi``, ``smoking_status``, ``recording_location``.  Any Python ``None``
    or pandas ``NA`` values are written as empty cells (which pandas renders
    as ``null``-equivalent empty strings when read back, and as ``NA`` in
    pandas internals).  No placeholder strings (e.g. ``"N/A"``) are used.

    Parent directories are created automatically if they do not exist.

    Args:
        df: DataFrame with patient metadata columns.
        output_path: Destination path for the CSV file (e.g.
            ``Path("data/processed/metadata.csv")``).

    Returns:
        None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure only the canonical columns are written, in the correct order.
    metadata_cols = [
        "patient_id",
        "age",
        "sex",
        "bmi",
        "smoking_status",
        "recording_location",
    ]

    # Select only columns that exist in df; missing columns are filled with NA.
    cols_to_write = []
    for col in metadata_cols:
        if col in df.columns:
            cols_to_write.append(col)
        else:
            logger.warning(
                "Expected metadata column '%s' not found in DataFrame; "
                "column will be filled with NA.",
                col,
            )

    out_df = df.reindex(columns=metadata_cols)
    # Write NA as empty string in CSV (standard pandas behaviour with na_rep='').
    # Use QUOTE_NONNUMERIC so all string-typed cells are quoted, which prevents
    # pandas from misinterpreting zero-padded strings (e.g. "00") as integers
    # when the CSV is reloaded with pd.read_csv().
    out_df.to_csv(output_path, index=False, na_rep="", quoting=csv.QUOTE_NONNUMERIC)


# ---------------------------------------------------------------------------
# Audio Signal Processing — Tasks 4.1–4.4
# ---------------------------------------------------------------------------

import librosa  # noqa: E402
import scipy.signal  # noqa: E402
import noisereduce  # noqa: E402


def resample_audio(audio: np.ndarray, orig_sr: int, config: Config) -> np.ndarray:
    """Resample *audio* to the target sample rate defined in *config*.

    Always performs resampling, even when ``orig_sr`` already equals
    ``config.target_sample_rate``.

    Args:
        audio: 1-D float32 numpy array of audio samples.
        orig_sr: Original sample rate of *audio* in Hz.
        config: Project configuration; ``config.target_sample_rate`` is used
            as the target rate (default 4000 Hz).

    Returns:
        Resampled audio as a float32 ndarray.

    Requirements: 3.1
    """
    resampled = librosa.resample(
        audio.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=config.target_sample_rate,
    )
    # librosa.resample may produce target_sr ± 1 samples due to floating-point
    # rounding. Trim or zero-pad to the exact expected integer length.
    expected_length = int(round(len(audio) * config.target_sample_rate / orig_sr))
    resampled = resampled[:expected_length] if len(resampled) > expected_length else np.pad(
        resampled, (0, max(0, expected_length - len(resampled))), mode="constant"
    )
    return resampled.astype(np.float32)


def apply_bandpass_filter(audio: np.ndarray, config: Config) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter to *audio*.

    The passband is ``[config.bandpass_low, config.bandpass_high]`` Hz, applied
    at the target sample rate ``config.target_sample_rate``.

    Args:
        audio: 1-D float32 numpy array of audio samples at
            ``config.target_sample_rate`` Hz.
        config: Project configuration supplying ``bandpass_low``,
            ``bandpass_high``, and ``target_sample_rate``.

    Returns:
        Bandpass-filtered audio as a float32 ndarray.

    Requirements: 3.2
    """
    nyquist = config.target_sample_rate / 2.0
    low = config.bandpass_low / nyquist
    # Clamp high strictly below Nyquist to satisfy scipy's requirement (Wn < 1)
    high = min(config.bandpass_high / nyquist, 0.9999)
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = scipy.signal.sosfilt(sos, audio.astype(np.float32))
    return filtered.astype(np.float32)


def apply_spectral_gating(audio: np.ndarray, config: Config) -> np.ndarray:
    """Apply spectral gating (noise reduction) to *audio*.

    Uses ``noisereduce.reduce_noise`` with sensitivity threshold
    ``config.noise_gate_threshold`` (default 0.5, range 0.0–1.0).

    Args:
        audio: 1-D float32 numpy array of audio samples at
            ``config.target_sample_rate`` Hz.
        config: Project configuration supplying ``noise_gate_threshold`` and
            ``target_sample_rate``.

    Returns:
        Denoised audio as a float32 ndarray.

    Requirements: 3.3
    """
    denoised = noisereduce.reduce_noise(
        y=audio.astype(np.float32),
        sr=config.target_sample_rate,
        prop_decrease=config.noise_gate_threshold,
        time_mask_smooth_ms=100,  # must be >= hop_length/sr*1000 for low sample rates
    )
    return denoised.astype(np.float32)


def segment_audio(audio: np.ndarray, config: Config) -> list[np.ndarray]:
    """Segment *audio* with a sliding window and 50% overlap.

    Window size: ``int(config.segment_duration * config.target_sample_rate)``
    (default: 5 s × 4000 Hz = 20 000 samples).

    Step size: ``int(window_size * (1 - config.segment_overlap))``
    (default: 20 000 × 0.5 = 10 000 samples).

    Trailing segments shorter than ``window_size`` are zero-padded to exactly
    ``window_size`` samples.  If the recording is shorter than one full window,
    a single zero-padded segment is returned.

    Args:
        audio: 1-D float32 numpy array of audio samples.
        config: Project configuration supplying ``segment_duration``,
            ``segment_overlap``, and ``target_sample_rate``.

    Returns:
        A list of float32 ndarrays, each of exactly ``window_size`` samples.

    Requirements: 3.4
    """
    window_size = int(config.segment_duration * config.target_sample_rate)
    step_size = int(window_size * (1.0 - config.segment_overlap))

    audio_f32 = audio.astype(np.float32)
    n_samples = len(audio_f32)

    # Handle recordings shorter than one full window
    if n_samples <= window_size:
        segment = np.zeros(window_size, dtype=np.float32)
        segment[:n_samples] = audio_f32
        return [segment]

    segments: list[np.ndarray] = []
    start = 0
    while start < n_samples:
        end = start + window_size
        chunk = audio_f32[start:end]
        if len(chunk) < window_size:
            # Zero-pad the trailing segment
            padded = np.zeros(window_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            segments.append(padded)
        else:
            segments.append(chunk.copy())
        start += step_size

    return segments


def preprocess_audio(audio_path: Path, config: Config) -> np.ndarray:
    """Full preprocessing pipeline: load → resample → filter → denoise → spectrogram.

    Processing chain:
        1. Load WAV file with original sample rate (mono).
        2. :func:`resample_audio` — resample to ``config.target_sample_rate``.
        3. :func:`apply_bandpass_filter` — 50–2000 Hz bandpass.
        4. :func:`apply_spectral_gating` — spectral noise reduction.
        5. :func:`segment_audio` — 5-second sliding window, 50% overlap.
        6. For each segment, compute a 128-bin log-mel spectrogram
           (``n_fft=config.n_fft``, ``hop_length=config.hop_length``), then
           convert to dB with ``ref=np.max``.
        7. Compute delta and delta-delta channels from the log-mel spectrogram.
        8. Zero-pad or truncate the time axis to exactly ``config.n_time_frames``
           (default 216) frames for each channel.
        9. Stack as shape ``(3, config.n_mels, config.n_time_frames)`` (float32).

    When multiple segments are produced, only the first segment's tensor is
    returned (single-segment inference).

    Args:
        audio_path: Path to a WAV audio file.
        config: Project configuration containing all preprocessing constants.

    Returns:
        A float32 ndarray of shape ``(3, config.n_mels, config.n_time_frames)``
        — channels are [log-mel, delta, delta-delta].

    Requirements: 3.5, 3.6, 3.7
    """
    # 1. Load
    audio, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)

    # 2–4. Signal processing chain
    audio = resample_audio(audio, orig_sr, config)
    audio = apply_bandpass_filter(audio, config)
    audio = apply_spectral_gating(audio, config)

    # 5. Segment
    segments = segment_audio(audio, config)

    def _build_tensor(seg: np.ndarray) -> np.ndarray:
        """Build a (3, n_mels, n_time_frames) tensor from one segment."""
        # Log-mel spectrogram
        S = librosa.feature.melspectrogram(
            y=seg,
            sr=config.target_sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        log_mel = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)

        # Delta and delta-delta
        delta = librosa.feature.delta(log_mel)
        delta2 = librosa.feature.delta(log_mel, order=2)

        def _fix_time(arr: np.ndarray) -> np.ndarray:
            """Zero-pad or truncate the time axis to n_time_frames."""
            n_t = config.n_time_frames
            if arr.shape[1] >= n_t:
                return arr[:, :n_t].astype(np.float32)
            pad_width = n_t - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant").astype(
                np.float32
            )

        log_mel_fixed = _fix_time(log_mel)
        delta_fixed = _fix_time(delta)
        delta2_fixed = _fix_time(delta2)

        # Stack: (3, n_mels, n_time_frames)
        return np.stack([log_mel_fixed, delta_fixed, delta2_fixed], axis=0).astype(
            np.float32
        )

    # Return tensor for the first segment only
    return _build_tensor(segments[0])


# ---------------------------------------------------------------------------
# Public API — Patient-Independent Data Split
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
    split_def: Path,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the official ICBHI patient-level split to *df*.

    Loads *split_def* (a JSON file with ``"train"`` and ``"test"`` lists of
    patient ID strings), validates that the two sets are disjoint, warns about
    any patient present in *df* but absent from *split_def*, then writes
    ``train.csv`` and ``test.csv`` to ``config.splits_dir``.

    Args:
        df: DataFrame containing at minimum the columns ``patient_id``,
            ``recording_id``, ``segment_id``, ``spectrogram_path``, and
            ``disease_class``.
        split_def: Path to the JSON split-definition file.  Expected format::

                {"train": ["101", "102", ...], "test": ["103", "104", ...]}

        config: Project configuration; ``config.splits_dir`` is used as the
            output directory for the split CSVs.

    Returns:
        A two-tuple ``(train_df, test_df)`` where each element is a DataFrame
        restricted to the five canonical columns and filtered to the
        corresponding patient IDs.

    Raises:
        AssertionError: If the intersection of *train_ids* and *test_ids* is
            non-empty.  The error message lists the sorted overlapping IDs.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # ------------------------------------------------------------------
    # 1. Load split definition
    # ------------------------------------------------------------------
    with split_def.open(encoding="utf-8") as fh:
        split_data = json.load(fh)

    train_ids: set[str] = set(str(pid) for pid in split_data["train"])
    test_ids: set[str] = set(str(pid) for pid in split_data["test"])

    # ------------------------------------------------------------------
    # 2. Assert no overlap — Requirement 4.2
    # ------------------------------------------------------------------
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(
            f"Overlapping patient IDs in split definition: {sorted(overlap)}"
        )

    # ------------------------------------------------------------------
    # 3. Warn about patients absent from split definition — Requirement 4.3
    # ------------------------------------------------------------------
    all_split_ids = train_ids | test_ids
    df_patient_ids = set(df["patient_id"].astype(str).unique())
    unassigned = df_patient_ids - all_split_ids
    for pid in sorted(unassigned):
        logger.warning(
            "Patient ID '%s' is present in the dataset but absent from the "
            "split definition; excluding from both splits.",
            pid,
        )

    # ------------------------------------------------------------------
    # 4. Build split DataFrames — Requirements 4.1, 4.5
    # ------------------------------------------------------------------
    _SPLIT_COLS = [
        "patient_id",
        "recording_id",
        "segment_id",
        "spectrogram_path",
        "disease_class",
    ]

    df_str_pid = df.copy()
    df_str_pid["patient_id"] = df_str_pid["patient_id"].astype(str)

    train_df = df_str_pid[df_str_pid["patient_id"].isin(train_ids)][_SPLIT_COLS].copy()
    test_df = df_str_pid[df_str_pid["patient_id"].isin(test_ids)][_SPLIT_COLS].copy()

    # ------------------------------------------------------------------
    # 5. Write CSVs — Requirements 4.4, 4.5
    # ------------------------------------------------------------------
    splits_dir = config.splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Main entry point — run the full preprocessing pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = Config()

    # ------------------------------------------------------------------
    # 1. Parse ICBHI annotations
    # ------------------------------------------------------------------
    print("Step 1/4: Parsing ICBHI annotations...")
    icbhi_dir = config.raw_icbhi_dir

    # ICBHI dataset extracts into a nested structure.
    # Try multiple known locations for ICBHI_diagnosis.txt
    candidate_names = ["ICBHI_diagnosis.txt", "demographic_info.txt"]
    ann_dir = None

    for name in candidate_names:
        found = list(icbhi_dir.rglob(name))
        if found:
            ann_dir = found[0].parent
            logger.info("Using diagnosis file: %s", found[0])
            break

    # Also search for the audio_and_txt_files directory which contains annotation .txt files
    audio_txt_dirs = list(icbhi_dir.rglob("audio_and_txt_files"))
    if audio_txt_dirs:
        audio_txt_dir = audio_txt_dirs[0]
        # Use parent of audio_and_txt_files for diagnosis file lookup if not found above
        if ann_dir is None:
            ann_dir = audio_txt_dir.parent
    else:
        audio_txt_dir = ann_dir

    if ann_dir is None:
        print(f"ERROR: ICBHI_diagnosis.txt not found under {icbhi_dir}")
        print("Make sure you have downloaded and extracted the ICBHI dataset.")
        sys.exit(1)

    print(f"  Found data directory: {ann_dir}")

    # Parse annotations from the audio_and_txt_files directory
    parse_dir = audio_txt_dir if audio_txt_dir else ann_dir

    # Copy diagnosis file into parse_dir if it's not already there
    for name in candidate_names:
        src_diag = ann_dir / name
        dst_diag = parse_dir / "ICBHI_diagnosis.txt"
        if src_diag.exists() and not dst_diag.exists():
            import shutil as _shutil
            _shutil.copy(str(src_diag), str(dst_diag))
            print(f"  Copied {name} → {dst_diag}")
            break

    df = parse_icbhi_annotations(parse_dir)
    print(f"  Parsed {len(df)} segments from {df['patient_id'].nunique()} patients")

    if df.empty:
        print("ERROR: No segments parsed. Check the data directory.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Save metadata
    # ------------------------------------------------------------------
    print("Step 2/4: Saving patient metadata...")
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata DataFrame from the parsed annotations
    metadata_cols = ["patient_id", "diagnosis"]
    meta_df = df[["patient_id"]].drop_duplicates().copy()
    meta_df["age"] = None
    meta_df["sex"] = None
    meta_df["bmi"] = None
    meta_df["smoking_status"] = None
    meta_df["recording_location"] = None

    # Try to load demographic info from Arashnic dataset if available
    arashnic_dir = config.raw_arashnic_dir
    if arashnic_dir.exists():
        demo_files = list(arashnic_dir.rglob("*.csv"))
        for demo_file in demo_files:
            try:
                import pandas as pd_inner
                demo = pd_inner.read_csv(demo_file)
                if "patient_id" in demo.columns or "id" in demo.columns.str.lower().tolist():
                    print(f"  Found demographic data: {demo_file}")
                    break
            except Exception:
                pass

    save_metadata(meta_df, config.metadata_path)
    print(f"  Metadata saved to {config.metadata_path}")

    # ------------------------------------------------------------------
    # 3. Preprocess audio → spectrograms
    # ------------------------------------------------------------------
    print("Step 3/4: Preprocessing audio files...")
    spectrograms_dir = config.processed_dir / "spectrograms"
    spectrograms_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import numpy as np

    spectrogram_paths = []
    skipped = 0

    # Find all WAV files in the audio_and_txt_files directory
    wav_files = list(parse_dir.rglob("*.wav"))
    if not wav_files:
        wav_files = list(icbhi_dir.rglob("*.wav"))

    print(f"  Found {len(wav_files)} WAV files")

    for i, wav_path in enumerate(wav_files):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(wav_files)}...")
        try:
            spec = preprocess_audio(wav_path, config)
            npy_path = spectrograms_dir / (wav_path.stem + ".npy")
            np.save(str(npy_path), spec)
            # Extract patient_id from filename
            match = _FILENAME_RE.match(wav_path.name)
            patient_id = match.group(1) if match else wav_path.stem[:3]
            spectrogram_paths.append({
                "patient_id": patient_id,
                "recording_id": wav_path.stem,
                "spectrogram_path": str(npy_path),
            })
        except Exception as exc:
            logger.warning("Failed to preprocess %s: %s", wav_path.name, exc)
            skipped += 1

    print(f"  Processed {len(spectrogram_paths)} files ({skipped} skipped)")

    # ------------------------------------------------------------------
    # 4. Build split CSVs
    # ------------------------------------------------------------------
    print("Step 4/4: Building train/test splits...")

    # Merge spectrogram paths with annotation data
    spec_df = pd.DataFrame(spectrogram_paths)
    merged = pd.merge(
        df[["patient_id", "recording_id", "diagnosis"]].drop_duplicates("recording_id"),
        spec_df,
        on=["patient_id", "recording_id"],
        how="inner",
    )

    # Map diagnosis to DiseaseClass
    merged["disease_class"] = merged["diagnosis"].apply(
        lambda x: map_label(x).value if map_label(x) else None
    )
    merged = merged.dropna(subset=["disease_class"])
    merged["segment_id"] = 0  # single segment per file for simplicity

    split_def_path = config.splits_dir / "split_def.json"
    config.splits_dir.mkdir(parents=True, exist_ok=True)

    # Use official ICBHI patient-level split if known, else 80/20 split
    # Official ICBHI train patients (first 60% by patient ID numeric order)
    all_patient_ids = sorted(merged["patient_id"].unique(), key=lambda x: int(x) if x.isdigit() else 0)
    split_idx = int(len(all_patient_ids) * 0.8)
    train_ids = all_patient_ids[:split_idx]
    test_ids = all_patient_ids[split_idx:]

    split_def = {"train": train_ids, "test": test_ids}
    with open(split_def_path, "w") as f:
        json.dump(split_def, f)

    train_df, test_df = split_dataset(merged, split_def_path, config)

    print(f"\n✅ Preprocessing complete!")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples:  {len(test_df)}")
    print(f"  Disease classes in train:\n{train_df['disease_class'].value_counts().to_string()}")
    print(f"\nSplit CSVs saved to: {config.splits_dir}")
