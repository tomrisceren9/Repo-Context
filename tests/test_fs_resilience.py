from unittest.mock import patch

import pytest

from air_context_engine.core.fs import safe_resolve_path
from air_context_engine.analysis.scanner import RepositoryScanner


def test_safe_resolve_path_success(tmp_path):
    """Test that safe_resolve_path works normally when there are no permission errors."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    resolved = safe_resolve_path(test_dir, strict=True)
    assert resolved == test_dir.resolve()
    assert resolved.is_dir()


@patch("pathlib.Path.resolve")
def test_safe_resolve_path_permission_error_fallback(mock_resolve, tmp_path):
    """Test that safe_resolve_path falls back to absolute path on WinError 5 / PermissionError."""
    # Simulate a PermissionError (WinError 5) on network drive resolution
    mock_resolve.side_effect = PermissionError("[WinError 5] Access is denied")

    test_dir = tmp_path / "network_drive"
    test_dir.mkdir()

    resolved = safe_resolve_path(test_dir, strict=True)

    # Verify fallback to absolute path
    assert resolved == test_dir.absolute()
    mock_resolve.assert_called_once()


@patch("pathlib.Path.resolve")
def test_safe_resolve_path_filenotfound(mock_resolve, tmp_path):
    """Test strict=True raises FileNotFoundError even with fallback if file doesn't exist."""
    # Let resolve raise PermissionError, forcing fallback to absolute path
    mock_resolve.side_effect = PermissionError("[WinError 5] Access is denied")

    missing_path = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        safe_resolve_path(missing_path, strict=True)


@patch("pathlib.Path.resolve")
def test_scanner_init_resilience(mock_resolve, tmp_path):
    """Test RepositoryScanner can initialize properly without crashing on PermissionError."""
    mock_resolve.side_effect = PermissionError("Access denied simulated")

    test_repo = tmp_path / "repo"
    test_repo.mkdir()

    # Should not crash during initialization when resolving canonical_root
    scanner = RepositoryScanner(test_repo)

    assert scanner.canonical_root == test_repo.absolute()


def test_scanner_ignores_child_when_canonical_resolution_is_denied(tmp_path):
    test_repo = tmp_path / "repo"
    test_repo.mkdir()
    source = test_repo / "service.py"
    source.write_text("def run():\n    return True\n", encoding="utf-8")
    scanner = RepositoryScanner(test_repo)

    with patch("pathlib.Path.resolve", side_effect=PermissionError("denied")):
        assert scanner.is_ignored(source) is True
