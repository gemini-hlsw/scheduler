# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""collector.visibility_strategy wiring: config default + explicit override, and
the async sight loader awaits (no nested event loop)."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.components.collector import collector as collector_mod
from scheduler.core.components.collector.collector import Collector


def _resolve(use_local, strategy):
    obj = Collector.__new__(Collector)
    obj.use_local_visibility = use_local
    fake_config = SimpleNamespace(collector=SimpleNamespace(visibility_strategy=strategy))
    with patch.object(collector_mod, "config", fake_config):
        return obj._use_local_visibility()


def test_explicit_override_wins_over_config():
    # Explicit True/False ignores the config.
    assert _resolve(True, "sight") is True
    assert _resolve(False, "local") is False


def test_defers_to_config_when_unset():
    assert _resolve(None, "local") is True
    assert _resolve(None, "sight") is False


def test_config_strategy_is_case_insensitive_and_safe():
    assert _resolve(None, "LOCAL") is True
    assert _resolve(None, " Local ") is True
    # Anything that isn't "local" means use the sight service.
    assert _resolve(None, "sight") is False
    assert _resolve(None, "whatever") is False


@pytest.mark.asyncio
async def test_async_sight_loader_awaits_and_applies():
    """The async loader must await _fetch_sight_data (not asyncio.run, which would
    raise inside a running loop) and feed the result to _apply_sight_visibility."""
    obj = Collector.__new__(Collector)
    obj.start_vis_time = datetime(2026, 2, 1)
    obj.end_vis_time = datetime(2026, 2, 5)
    obj.sites = frozenset({Site.GS})
    obj._fetch_sight_data = AsyncMock(return_value={"sentinel": 1})
    obj._apply_sight_visibility = MagicMock()

    filtered = {}
    await obj._async_load_visibility_from_sight(filtered, sem=None)

    obj._fetch_sight_data.assert_awaited_once()
    # _fetch_sight_data(filtered_observations, start_date, end_date, site_ids)
    args = obj._fetch_sight_data.await_args.args
    assert args[0] is filtered
    assert args[1] == obj.start_vis_time.date()
    assert args[2] == obj.end_vis_time.date()
    assert args[3] == ["GS"]
    obj._apply_sight_visibility.assert_called_once_with(filtered, {"sentinel": 1})
