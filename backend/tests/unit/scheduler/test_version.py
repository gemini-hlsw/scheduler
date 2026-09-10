# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from scheduler.version import get_app_version


def test_reads_version_from_version_file(tmp_path):
    version_file = tmp_path / "version.py"
    version_file.write_text('"""CI generated."""\n__version__ = "2026.07.9"\n')
    assert get_app_version(version_file) == "2026.07.9"


def test_missing_file_defaults_to_development(tmp_path):
    assert get_app_version(tmp_path / "version.py") == "development"


def test_env_var_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_VERSION", "not-this-one")
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "2026.07.9"\n')
    assert get_app_version(version_file) == "2026.07.9"
    assert get_app_version(tmp_path / "absent.py") == "development"


def test_repo_checkout_resolves_a_version():
    # In a dev checkout (and in the Docker image, which copies version.py to
    # /home) the default resolution finds the CI-generated file.
    version = get_app_version()
    assert version != "development"
