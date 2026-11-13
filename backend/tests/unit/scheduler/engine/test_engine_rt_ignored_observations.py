# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Every plan is built against the ignore list as it stands when the build starts."""

import inspect

from scheduler.engine.engineRT import EngineRT


def test_the_build_hands_the_ignore_list_to_the_collector():
    source = inspect.getsource(EngineRT.build)

    assert "ignored_observation_ids=ignored_observations_store.snapshot()" in source
