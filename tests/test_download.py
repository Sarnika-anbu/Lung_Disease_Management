"""Unit tests for src/data/download.py — archive validation and extraction.

Covers the ``_validate_archive`` helper (Requirement 1.6) and the
``_extract_archive`` helper (Requirements 1.7, 1.8):

_validate_archive:
- exits non-zero when no .zip file is present in dest_dir
- exits non-zero when the .zip file exists but is 0 bytes
- returns the Path to the archive when the file is valid (≥ 1 byte)

_extract_archive:
- successfully extracts a valid zip archive to dest_dir
- exits non-zero on a corrupt/bad zip file
- exits non-zero on a generic I/O error
- error messages identify the dataset label
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the private helpers directly for unit testing.
from src.data.download import _extract_archive, _validate_archive


class TestValidateArchive:
    """Tests for _validate_archive(dest_dir, dataset_label)."""

    def test_exits_when_no_zip_found(self, tmp_path: Path) -> None:
        """Should call sys.exit(1) when dest_dir contains no .zip file."""
        with pytest.raises(SystemExit) as exc_info:
            _validate_archive(tmp_path, "Test Dataset")
        assert exc_info.value.code == 1

    def test_exits_when_zip_is_zero_bytes(self, tmp_path: Path) -> None:
        """Should call sys.exit(1) when the .zip file exists but is empty."""
        empty_zip = tmp_path / "dataset.zip"
        empty_zip.write_bytes(b"")
        with pytest.raises(SystemExit) as exc_info:
            _validate_archive(tmp_path, "Test Dataset")
        assert exc_info.value.code == 1

    def test_returns_path_for_valid_archive(self, tmp_path: Path) -> None:
        """Should return the archive Path when the .zip file is at least 1 byte."""
        valid_zip = tmp_path / "dataset.zip"
        valid_zip.write_bytes(b"PK\x03\x04")  # minimal zip magic bytes
        result = _validate_archive(tmp_path, "Test Dataset")
        assert result == valid_zip

    def test_error_message_contains_dataset_label_when_absent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error output should identify the dataset name when no zip is found."""
        with pytest.raises(SystemExit):
            _validate_archive(tmp_path, "My Special Dataset")
        captured = capsys.readouterr()
        assert "My Special Dataset" in captured.err

    def test_error_message_contains_dataset_label_when_zero_bytes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error output should identify the dataset name when the archive is 0 bytes."""
        (tmp_path / "dataset.zip").write_bytes(b"")
        with pytest.raises(SystemExit):
            _validate_archive(tmp_path, "My Special Dataset")
        captured = capsys.readouterr()
        assert "My Special Dataset" in captured.err

    def test_picks_up_zip_regardless_of_filename(self, tmp_path: Path) -> None:
        """Any .zip filename in dest_dir should be recognised as the archive."""
        archive = tmp_path / "respiratory-sound-database.zip"
        archive.write_bytes(b"\xff" * 100)
        result = _validate_archive(tmp_path, "ICBHI Dataset")
        assert result == archive


def _make_zip(dest: Path, members: dict[str, bytes]) -> None:
    """Write a real zip archive at *dest* containing the given file members.

    Args:
        dest: Path where the zip file should be written.
        members: Mapping of archive member name → raw bytes content.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    dest.write_bytes(buf.getvalue())


class TestExtractArchive:
    """Tests for _extract_archive(archive, dest_dir, dataset_label)."""

    def test_extracts_files_to_dest_dir(self, tmp_path: Path) -> None:
        """Valid zip contents should appear in dest_dir after extraction."""
        archive = tmp_path / "dataset.zip"
        _make_zip(archive, {"audio.wav": b"RIFF", "meta.txt": b"patient"})

        dest = tmp_path / "extracted"
        dest.mkdir()
        _extract_archive(archive, dest, "Test Dataset")

        assert (dest / "audio.wav").read_bytes() == b"RIFF"
        assert (dest / "meta.txt").read_bytes() == b"patient"

    def test_exits_on_bad_zip_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit with status 1 when the archive is corrupt."""
        archive = tmp_path / "bad.zip"
        archive.write_bytes(b"not a zip at all")

        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            _extract_archive(archive, dest, "Corrupt Dataset")
        assert exc_info.value.code == 1

    def test_exits_on_bad_zip_error_message_contains_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error output should identify the dataset label on BadZipFile."""
        archive = tmp_path / "bad.zip"
        archive.write_bytes(b"\x00\x01\x02")

        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(SystemExit):
            _extract_archive(archive, dest, "My Special Dataset")
        captured = capsys.readouterr()
        assert "My Special Dataset" in captured.err

    def test_exits_on_generic_exception(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit with status 1 when ZipFile.extractall raises an unexpected error."""
        archive = tmp_path / "dataset.zip"
        _make_zip(archive, {"file.txt": b"data"})

        dest = tmp_path / "out"
        dest.mkdir()

        with patch("zipfile.ZipFile.extractall", side_effect=OSError("disk full")):
            with pytest.raises(SystemExit) as exc_info:
                _extract_archive(archive, dest, "Failing Dataset")
        assert exc_info.value.code == 1

    def test_error_message_contains_label_on_generic_exception(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error output should identify the dataset label on a generic exception."""
        archive = tmp_path / "dataset.zip"
        _make_zip(archive, {"file.txt": b"data"})

        dest = tmp_path / "out"
        dest.mkdir()

        with patch("zipfile.ZipFile.extractall", side_effect=OSError("disk full")):
            with pytest.raises(SystemExit):
                _extract_archive(archive, dest, "Labeled Dataset")
        captured = capsys.readouterr()
        assert "Labeled Dataset" in captured.err

    def test_prints_extraction_progress_messages(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should print start and completion messages to stdout."""
        archive = tmp_path / "dataset.zip"
        _make_zip(archive, {"x.txt": b"hello"})

        dest = tmp_path / "out"
        dest.mkdir()
        _extract_archive(archive, dest, "Progress Dataset")

        captured = capsys.readouterr()
        assert "Progress Dataset" in captured.out
        assert "Extracting" in captured.out
        assert "complete" in captured.out.lower()


# ---------------------------------------------------------------------------
# Additional tests added for Task 2.5
# Covers: credential loading (Req 1.3, 1.4), skip logic (Req 1.5),
#         0-byte archive via _download_single_dataset (Req 1.6),
#         network error (Req 1.8)
# ---------------------------------------------------------------------------

import json
from unittest.mock import MagicMock, patch

from src.config import Config
from src.data.download import (
    _download_single_dataset,
    _load_credentials,
    download_datasets,
)


class TestLoadCredentials:
    """Tests for _load_credentials() — Requirement 1.3, 1.4."""

    def test_loads_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return (username, key) when env vars are present."""
        monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
        monkeypatch.setenv("KAGGLE_KEY", "env_key_123")
        username, key = _load_credentials()
        assert username == "env_user"
        assert key == "env_key_123"

    def test_env_vars_take_priority_over_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars should be used even when kaggle.json also exists."""
        monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
        monkeypatch.setenv("KAGGLE_KEY", "env_key")

        # Place a kaggle.json with different credentials at a tmp location
        kaggle_json = tmp_path / "kaggle.json"
        kaggle_json.write_text(
            json.dumps({"username": "json_user", "key": "json_key"}),
            encoding="utf-8",
        )
        # Redirect the module-level path constant to our tmp file
        monkeypatch.setattr("src.data.download._KAGGLE_JSON_PATH", kaggle_json)

        username, key = _load_credentials()
        assert username == "env_user"
        assert key == "env_key"

    def test_falls_back_to_kaggle_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should read credentials from kaggle.json when env vars are absent."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        kaggle_json = tmp_path / "kaggle.json"
        kaggle_json.write_text(
            json.dumps({"username": "json_user", "key": "json_key_abc"}),
            encoding="utf-8",
        )
        monkeypatch.setattr("src.data.download._KAGGLE_JSON_PATH", kaggle_json)

        username, key = _load_credentials()
        assert username == "json_user"
        assert key == "json_key_abc"

    def test_exits_when_both_env_vars_and_json_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should exit with code 1 when no credentials are available."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        # Point the path to a non-existent file so the fallback also fails
        monkeypatch.setattr(
            "src.data.download._KAGGLE_JSON_PATH",
            tmp_path / "nonexistent.json",
        )

        with pytest.raises(SystemExit) as exc_info:
            _load_credentials()
        assert exc_info.value.code == 1

    def test_error_message_contains_json_path_and_var_names(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Error output should mention the env-var names and kaggle.json path."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        fake_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("src.data.download._KAGGLE_JSON_PATH", fake_path)

        with pytest.raises(SystemExit):
            _load_credentials()
        captured = capsys.readouterr()
        assert "KAGGLE_USERNAME" in captured.err
        assert "KAGGLE_KEY" in captured.err

    def test_exits_when_json_missing_username_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should exit with code 1 when kaggle.json is missing the 'username' key."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        kaggle_json = tmp_path / "kaggle.json"
        kaggle_json.write_text(json.dumps({"key": "only_key"}), encoding="utf-8")
        monkeypatch.setattr("src.data.download._KAGGLE_JSON_PATH", kaggle_json)

        with pytest.raises(SystemExit) as exc_info:
            _load_credentials()
        assert exc_info.value.code == 1

    def test_exits_when_json_is_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should exit with code 1 when kaggle.json contains malformed JSON."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        kaggle_json = tmp_path / "kaggle.json"
        kaggle_json.write_text("not valid json {{", encoding="utf-8")
        monkeypatch.setattr("src.data.download._KAGGLE_JSON_PATH", kaggle_json)

        with pytest.raises(SystemExit) as exc_info:
            _load_credentials()
        assert exc_info.value.code == 1


class TestDownloadDatasets:
    """Tests for download_datasets() — Requirements 1.3, 1.4, 1.5, 1.6, 1.8."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_credentials_patch(monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject valid credentials via env vars so _load_credentials succeeds."""
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key_xyz")

    # ------------------------------------------------------------------
    # Skip logic (Requirement 1.5)
    # ------------------------------------------------------------------

    def test_skips_both_datasets_when_dirs_non_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When both target dirs are non-empty the Kaggle API must never be called."""
        self._valid_credentials_patch(monkeypatch)

        icbhi_dir = tmp_path / "icbhi"
        icbhi_dir.mkdir()
        (icbhi_dir / "dummy.wav").write_bytes(b"\x00")

        arashnic_dir = tmp_path / "arashnic"
        arashnic_dir.mkdir()
        (arashnic_dir / "dummy.wav").write_bytes(b"\x00")

        config = Config(
            raw_icbhi_dir=icbhi_dir,
            raw_arashnic_dir=arashnic_dir,
        )

        # Patch KaggleApiExtended so any call would be detectable
        with patch("src.data.download._download_single_dataset") as mock_dl:
            # Re-implement only the skip check so we verify the real
            # _is_non_empty_dir logic propagates through download_datasets.
            # Restore original to test it properly:
            pass

        # Use real implementation but mock the Kaggle API layer
        mock_api = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "kaggle": MagicMock(),
                "kaggle.api": MagicMock(),
                "kaggle.api.kaggle_api_extended": MagicMock(
                    KaggleApiExtended=MagicMock(return_value=mock_api)
                ),
            },
        ):
            download_datasets(config=config)

        # dataset_download_files should NOT have been called for either dataset
        mock_api.dataset_download_files.assert_not_called()
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_calls_api_once_when_only_one_dir_non_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Kaggle API should be called exactly once when one dir is empty."""
        self._valid_credentials_patch(monkeypatch)

        # ICBHI dir is non-empty → should be skipped
        icbhi_dir = tmp_path / "icbhi"
        icbhi_dir.mkdir()
        (icbhi_dir / "dummy.wav").write_bytes(b"\x00")

        # Arashnic dir doesn't exist yet → should trigger download
        arashnic_dir = tmp_path / "arashnic"

        config = Config(
            raw_icbhi_dir=icbhi_dir,
            raw_arashnic_dir=arashnic_dir,
        )

        mock_api = MagicMock()
        # Make dataset_download_files create a valid zip so _validate_archive
        # and _extract_archive don't cause SystemExit
        def fake_download(*args, **kwargs):  # noqa: ANN001, ANN202
            dest = Path(kwargs.get("path", args[1] if len(args) > 1 else str(arashnic_dir)))
            import io as _io
            import zipfile as _zf

            buf = _io.BytesIO()
            with _zf.ZipFile(buf, "w") as zf:
                zf.writestr("file.txt", "data")
            (Path(dest) / "archive.zip").write_bytes(buf.getvalue())

        mock_api.dataset_download_files.side_effect = fake_download
        mock_api.authenticate.return_value = None

        mock_kaggle_api_module = MagicMock()
        mock_kaggle_api_module.KaggleApiExtended = MagicMock(return_value=mock_api)

        with patch.dict(
            "sys.modules",
            {
                "kaggle": MagicMock(),
                "kaggle.api": MagicMock(),
                "kaggle.api.kaggle_api_extended": mock_kaggle_api_module,
            },
        ):
            download_datasets(config=config)

        assert mock_api.dataset_download_files.call_count == 1

    # ------------------------------------------------------------------
    # Network error → exit non-zero (Requirement 1.8)
    # ------------------------------------------------------------------

    def test_exits_nonzero_on_network_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should exit with code 1 and print the dataset name when API raises."""
        self._valid_credentials_patch(monkeypatch)

        config = Config(
            raw_icbhi_dir=tmp_path / "icbhi",
            raw_arashnic_dir=tmp_path / "arashnic",
        )

        mock_api = MagicMock()
        mock_api.dataset_download_files.side_effect = ConnectionError(
            "Simulated network failure"
        )
        mock_api.authenticate.return_value = None

        mock_kaggle_api_module = MagicMock()
        mock_kaggle_api_module.KaggleApiExtended = MagicMock(return_value=mock_api)

        with patch.dict(
            "sys.modules",
            {
                "kaggle": MagicMock(),
                "kaggle.api": MagicMock(),
                "kaggle.api.kaggle_api_extended": mock_kaggle_api_module,
            },
        ):
            with pytest.raises(SystemExit) as exc_info:
                download_datasets(config=config)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Error message should identify the dataset and failure reason
        assert "ICBHI" in captured.err or "Simulated" in captured.err

    # ------------------------------------------------------------------
    # 0-byte archive → exit non-zero (Requirement 1.6 via _download_single_dataset)
    # ------------------------------------------------------------------

    def test_exits_nonzero_on_zero_byte_archive_via_download_single_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """After download, if the archive is 0 bytes, exit non-zero."""
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key")

        dest_dir = tmp_path / "dataset"

        mock_api = MagicMock()

        def fake_download_zero_bytes(*args, **kwargs):  # noqa: ANN001, ANN202
            """Create a 0-byte .zip file to simulate a failed download."""
            dest = kwargs.get("path", str(dest_dir))
            (Path(dest) / "archive.zip").write_bytes(b"")

        mock_api.dataset_download_files.side_effect = fake_download_zero_bytes
        mock_api.authenticate.return_value = None

        mock_kaggle_api_module = MagicMock()
        mock_kaggle_api_module.KaggleApiExtended = MagicMock(return_value=mock_api)

        with patch.dict(
            "sys.modules",
            {
                "kaggle": MagicMock(),
                "kaggle.api": MagicMock(),
                "kaggle.api.kaggle_api_extended": mock_kaggle_api_module,
            },
        ):
            with pytest.raises(SystemExit) as exc_info:
                _download_single_dataset(
                    dataset="owner/dataset",
                    dest_dir=dest_dir,
                    dataset_label="Test Dataset",
                )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Test Dataset" in captured.err

    # ------------------------------------------------------------------
    # Missing credentials → exit non-zero (Requirement 1.4)
    # ------------------------------------------------------------------

    def test_exits_nonzero_on_missing_credentials(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """download_datasets should exit 1 when no credentials are available."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        monkeypatch.setattr(
            "src.data.download._KAGGLE_JSON_PATH",
            tmp_path / "nonexistent.json",
        )

        config = Config(
            raw_icbhi_dir=tmp_path / "icbhi",
            raw_arashnic_dir=tmp_path / "arashnic",
        )

        with pytest.raises(SystemExit) as exc_info:
            download_datasets(config=config)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "KAGGLE" in captured.err
