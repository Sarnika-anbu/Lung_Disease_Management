"""Dataset download pipeline for the Lung Disease Management System.

Downloads the ICBHI 2017 Respiratory Sound Database and the Arashnic Lung
Sounds dataset from Kaggle to their respective raw data directories.

Usage::

    python src/data/download.py

Credentials are read from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment
variables, or from ``~/.kaggle/kaggle.json`` if the environment variables are
absent.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

from src.config import Config


# ---------------------------------------------------------------------------
# Kaggle dataset identifiers
# ---------------------------------------------------------------------------

_ICBHI_DATASET = "vbookshelf/respiratory-sound-database"
_ARASHNIC_DATASET = "arashnic/lung-sound-dataset"

_KAGGLE_JSON_PATH = Path.home() / ".kaggle" / "kaggle.json"


def _load_credentials() -> tuple[str, str]:
    """Load Kaggle credentials from environment variables or kaggle.json.

    Attempts to read ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` from environment
    variables first.  If either is absent, falls back to
    ``~/.kaggle/kaggle.json``.

    Returns:
        A ``(username, key)`` tuple of Kaggle API credentials.

    Raises:
        SystemExit: If credentials cannot be found or are invalid, prints a
            descriptive message to stderr and exits with status code 1.
    """
    username = os.environ.get("sarnikaga")
    key = os.environ.get("KGAT_c10a0caaa784bfb8452210d66440edde")

    if username and key:
        return username, key

    # Fall back to ~/.kaggle/kaggle.json
    if _KAGGLE_JSON_PATH.exists():
        try:
            creds = json.loads(_KAGGLE_JSON_PATH.read_text(encoding="utf-8"))
            username = creds.get("username")
            key = creds.get("key")
            if username and key:
                return username, key
            missing = []
            if not username:
                missing.append('"username"')
            if not key:
                missing.append('"key"')
            print(
                f"ERROR: {_KAGGLE_JSON_PATH} exists but is missing the "
                f"required field(s): {', '.join(missing)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(
                f"ERROR: Failed to parse {_KAGGLE_JSON_PATH}: {exc}. "
                "Ensure the file contains valid JSON with 'username' and 'key' fields.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Neither env vars nor kaggle.json found — print a descriptive error
    missing_vars: list[str] = []
    if not os.environ.get("KAGGLE_USERNAME"):
        missing_vars.append("KAGGLE_USERNAME")
    if not os.environ.get("KAGGLE_KEY"):
        missing_vars.append("KAGGLE_KEY")

    print(
        "ERROR: Kaggle credentials not found. "
        "Please provide credentials using one of the following methods:\n"
        f"  1. Set the environment variable(s): {', '.join(missing_vars)}\n"
        f"  2. Place a valid kaggle.json file at: {_KAGGLE_JSON_PATH}\n"
        "     The file must contain JSON with 'username' and 'key' fields, "
        "e.g.:\n"
        '     {"username": "<your_username>", "key": "<your_api_key>"}',
        file=sys.stderr,
    )
    sys.exit(1)


def _configure_kaggle_env(username: str, key: str) -> None:
    """Set Kaggle credentials as environment variables for the kaggle package.

    The ``kaggle`` library reads credentials from the ``KAGGLE_USERNAME`` and
    ``KAGGLE_KEY`` environment variables (or from ``~/.kaggle/kaggle.json``).
    This helper ensures the environment variables are populated before any
    ``kaggle`` API calls are made.

    Args:
        username: Kaggle account username.
        key: Kaggle API key.
    """
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def _validate_archive(dest_dir: Path, dataset_label: str) -> Path:
    """Locate and validate the downloaded Kaggle archive in *dest_dir*.

    Kaggle's ``dataset_download_files`` call produces a single ``.zip`` file
    inside the destination directory.  This helper searches for that file,
    checks it is at least 1 byte in size, and returns its path so the caller
    can pass it to the extraction step.

    Args:
        dest_dir: Directory where the archive was downloaded.
        dataset_label: Human-readable dataset name used in error messages.

    Returns:
        The :class:`~pathlib.Path` of the validated ``.zip`` archive.

    Raises:
        SystemExit: If no ``.zip`` file is found in *dest_dir*, or if the
            found archive is 0 bytes, prints a descriptive error to stderr and
            exits with status code 1 without attempting extraction.
    """
    zip_files = list(dest_dir.glob("*.zip"))

    if not zip_files:
        print(
            f"ERROR: Archive validation failed for '{dataset_label}': "
            f"no .zip file found in {dest_dir} after download.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use the first (and typically only) zip found.
    archive = zip_files[0]

    if archive.stat().st_size == 0:
        print(
            f"ERROR: Archive validation failed for '{dataset_label}': "
            f"downloaded file {archive} is 0 bytes.",
            file=sys.stderr,
        )
        sys.exit(1)

    return archive


def _extract_archive(archive: Path, dest_dir: Path, dataset_label: str) -> None:
    """Extract a validated ``.zip`` archive to *dest_dir*.

    Uses :mod:`zipfile` from the Python standard library to extract all
    members of *archive* into *dest_dir*.  If extraction fails for any reason
    (corrupt archive, I/O error, etc.), prints a descriptive error to stderr
    and exits with status code 1.

    Args:
        archive: Path to the validated ``.zip`` file to extract.
        dest_dir: Directory into which all archive members are extracted.
        dataset_label: Human-readable dataset name used in error messages.

    Raises:
        SystemExit: On :exc:`zipfile.BadZipFile` or any other extraction
            error, prints a descriptive message to stderr and exits with
            status code 1.
    """
    print(f"Extracting {dataset_label} …")
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
        print(f"Extraction complete: {dataset_label}")
    except zipfile.BadZipFile as exc:
        print(
            f"ERROR: Extraction failed for '{dataset_label}': "
            f"{archive} is not a valid zip file. Reason: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: Extraction failed for '{dataset_label}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _is_non_empty_dir(path: Path) -> bool:
    """Return True when *path* exists as a directory and contains at least one file.

    Args:
        path: Filesystem path to check.

    Returns:
        ``True`` if *path* is an existing directory with at least one entry,
        ``False`` otherwise.
    """
    return path.is_dir() and any(path.iterdir())


def _download_single_dataset(
    dataset: str,
    dest_dir: Path,
    dataset_label: str,
) -> None:
    """Download a single Kaggle dataset to a local directory.

    Before attempting the download, checks whether *dest_dir* already exists
    and is non-empty.  If so, prints a skip message to stdout and returns
    immediately without downloading, leaving any other datasets unaffected.

    Args:
        dataset: Kaggle dataset identifier in the form ``"owner/name"``.
        dest_dir: Destination directory for the downloaded files.
        dataset_label: Human-readable dataset name used in log messages.

    Raises:
        SystemExit: On any download failure (network error, non-200 response,
            or API authentication error), prints an error to stderr and exits
            with status code 1.
    """
    # Per-dataset skip logic (Requirement 1.5): if the target directory already
    # exists and is non-empty, skip this dataset independently.
    if _is_non_empty_dir(dest_dir):
        print(
            f"Skipping {dataset_label}: {dest_dir} already exists and is non-empty."
        )
        return

    # Import here so the module can be imported without kaggle installed
    try:
        import kaggle  # noqa: PLC0415
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: The 'kaggle' Python package is not installed. "
            "Install it with: pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        api = KaggleApiExtended()
        api.authenticate()
        print(f"Downloading {dataset_label} to {dest_dir} …")
        api.dataset_download_files(
            dataset,
            path=str(dest_dir),
            unzip=False,
            quiet=False,
            force=False,
        )
        print(f"Download complete: {dataset_label}")
        # Requirement 1.6 — validate archive before extraction (task 2.3)
        archive_path = _validate_archive(dest_dir, dataset_label)
        # Requirement 1.7 — extract the verified archive (task 2.4)
        _extract_archive(archive_path, dest_dir, dataset_label)
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        print(
            f"ERROR: Failed to download '{dataset_label}' "
            f"(dataset: {dataset}). Reason: {error_msg}",
            file=sys.stderr,
        )
        sys.exit(1)


def download_datasets(config: Config | None = None) -> None:
    """Download both Kaggle datasets to their respective raw data directories.

    Reads Kaggle credentials from the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``
    environment variables, or from ``~/.kaggle/kaggle.json`` as a fallback.
    If credentials are missing or invalid, prints a descriptive error message
    to stderr and exits with a non-zero status code.

    Each dataset is downloaded to the directory defined in *config*:

    * ICBHI 2017 Respiratory Sound Database → ``data/raw/icbhi/``
    * Arashnic Lung Sounds Dataset → ``data/raw/arashnic/``

    If a dataset's target directory already exists and is non-empty, that
    dataset is skipped and a message is printed to stdout; the other dataset
    is still checked and downloaded if needed.

    Args:
        config: Project configuration instance.  Defaults to a ``Config()``
            constructed with default path values when ``None`` is provided.
    """
    if config is None:
        config = Config()

    # Load and configure credentials
    username, key = _load_credentials()
    _configure_kaggle_env(username, key)

    # Download ICBHI dataset
    _download_single_dataset(
        dataset=_ICBHI_DATASET,
        dest_dir=config.raw_icbhi_dir,
        dataset_label="ICBHI 2017 Respiratory Sound Database",
    )

    # Download Arashnic dataset
    _download_single_dataset(
        dataset=_ARASHNIC_DATASET,
        dest_dir=config.raw_arashnic_dir,
        dataset_label="Arashnic Lung Sounds Dataset",
    )


if __name__ == "__main__":
    download_datasets()
