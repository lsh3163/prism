"""Exercise distribution path validation with generated, never-extracted archives."""

import importlib.util
import io
import tarfile
import tempfile
import unittest
import zipfile
from collections.abc import Callable
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_distribution.py"
SPEC = importlib.util.spec_from_file_location("prism_distribution_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

PACKAGE_ROOT = "prism_robot-0.1.0"
CONTENTS = b"source bytes shared by the wheel and source archive\n"


def write_bundle(directory: Path, source_path: Callable[[str], str]) -> None:
    """Write only the required files, with identical core bytes in both formats."""
    with zipfile.ZipFile(directory / f"{PACKAGE_ROOT}-py3-none-any.whl", "w") as wheel:
        for name in VALIDATOR.CORE_FILES:
            wheel.writestr(name, CONTENTS)
    with tarfile.open(directory / f"{PACKAGE_ROOT}.tar.gz", "w:gz") as source:
        for name in sorted(VALIDATOR.SOURCE_FILES):
            member = tarfile.TarInfo(source_path(name))
            member.size = len(CONTENTS)
            source.addfile(member, io.BytesIO(CONTENTS))


class DistributionValidationTest(unittest.TestCase):
    def test_minimal_required_content_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root, lambda name: f"{PACKAGE_ROOT}/{name}")
            VALIDATOR.check_distributions(root)

    def test_absolute_source_member_paths_are_rejected(self) -> None:
        # Stripping the first component before checking would hide the '/'.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root, lambda name: f"/{name}")
            with self.assertRaises(ValueError):
                VALIDATOR.check_distributions(root)

    def test_parent_traversal_source_root_is_rejected(self) -> None:
        # Each required file still appears present if '..' is stripped first.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root, lambda name: f"../{name}")
            with self.assertRaises(ValueError):
                VALIDATOR.check_distributions(root)

    def test_mixed_source_roots_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            misplaced = sorted(VALIDATOR.SOURCE_FILES)[0]
            write_bundle(
                root,
                lambda name: f"{'different-package' if name == misplaced else PACKAGE_ROOT}/{name}",
            )
            with self.assertRaises(ValueError):
                VALIDATOR.check_distributions(root)

    def test_source_root_must_match_distribution_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root, lambda name: f"different-package/{name}")
            with self.assertRaises(ValueError):
                VALIDATOR.check_distributions(root)


if __name__ == "__main__":
    unittest.main()
