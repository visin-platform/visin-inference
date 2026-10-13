"""Reject tags that do not exactly match the version in the built wheel."""

import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


def check_release(tag: str, dist: Path = Path("dist")) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("release tags must be vMAJOR.MINOR.PATCH")
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_path).decode())
    version = metadata["Version"]
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match wheel version {version!r}")
    return version


if __name__ == "__main__":
    print(check_release(sys.argv[1]))
