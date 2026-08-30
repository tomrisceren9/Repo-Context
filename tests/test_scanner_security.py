import os
import tempfile
import unittest
from pathlib import Path

from air_context_engine.analysis.scanner import RepositoryScanner


class ScannerSecurityTests(unittest.TestCase):
    def test_credentials_hidden_files_and_external_symlinks_are_excluded(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            root = Path(repo_dir)
            outside = Path(outside_dir) / "outside.py"
            outside.write_text("secret = 'outside'\n", encoding="utf-8")
            (root / "public.py").write_text("value = 1\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=synthetic\n", encoding="utf-8")
            (root / "private.key").write_text("synthetic\n", encoding="utf-8")
            hidden = root / ".hidden"
            hidden.mkdir()
            (hidden / "module.py").write_text("value = 2\n", encoding="utf-8")
            link = root / "outside.py"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks are unavailable on this platform")

            scanned = {
                path.relative_to(root).as_posix()
                for path in RepositoryScanner(root).scan_files()
            }

        self.assertEqual(scanned, {"public.py"})

    def test_user_denylist_cannot_be_reopened_by_gitignore_negation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private = root / "private"
            private.mkdir()
            (private / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / ".gitignore").write_text("!private/module.py\n", encoding="utf-8")

            scanner = RepositoryScanner(root, deny_patterns=["private/"])
            scanned = {path.name for path in scanner.scan_files()}

        self.assertNotIn("module.py", scanned)

    @unittest.skipIf(os.name == "nt", "Unix permissions are required")
    def test_unreadable_child_resolution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "service.py"
            source.write_text("def run():\n    return True\n", encoding="utf-8")
            scanner = RepositoryScanner(root)

            from unittest.mock import patch

            with patch("pathlib.Path.resolve", side_effect=PermissionError("denied")):
                self.assertTrue(scanner.is_ignored(source))


if __name__ == "__main__":
    unittest.main()
