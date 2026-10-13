"""Release validation must reject mismatched or malformed publication tags."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_release import check_release


class ReleaseTest(unittest.TestCase):
    def test_requires_exact_version_match(self):
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            with zipfile.ZipFile(dist / "visin_infer-0.1.0-py3-none-any.whl", "w") as wheel:
                wheel.writestr("visin_infer-0.1.0.dist-info/METADATA", "Name: visin-infer\nVersion: 0.1.0\n")
            self.assertEqual(check_release("v0.1.0", dist), "0.1.0")
            for tag in ("v0.2.0", "0.1.0", "v0.1.0rc1", "v0.1.0/extra"):
                with self.subTest(tag=tag), self.assertRaises(ValueError):
                    check_release(tag, dist)

    def test_requires_one_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            with self.assertRaises(ValueError):
                check_release("v0.1.0", dist)
            (dist / "one.whl").touch()
            (dist / "two.whl").touch()
            with self.assertRaises(ValueError):
                check_release("v0.1.0", dist)
