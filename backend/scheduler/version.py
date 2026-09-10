# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import re
from pathlib import Path
from typing import Optional

__all__ = ["get_app_version"]

_PACKAGE_DIR = Path(__file__).resolve().parent

# The CI-generated CalVer file: at the repo root in a dev checkout, next to the
# scheduler package (/home) in the Docker image.
_VERSION_FILE_CANDIDATES = (_PACKAGE_DIR.parents[1] / "version.py",
                            _PACKAGE_DIR.parent / "version.py")


def get_app_version(version_file: Optional[Path] = None) -> str:
    """Version string from the CI-generated version.py, or "development"."""
    candidates = (version_file,) if version_file is not None else _VERSION_FILE_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                              candidate.read_text())
            if match:
                return match.group(1)
    return "development"
