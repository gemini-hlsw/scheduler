# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

import pytest

from definitions import ROOT_DIR
from scheduler.graphql_mid.server import schema


@pytest.mark.asyncio
async def test_change_origin_mutation():
    mut = """
        mutation changeOrigin{
                  changeOrigin(mode: VALIDATION, newOrigin: "OCS") {
                    fromOrigin
                    toOrigin
                  }
                }
    """

    result = await schema.execute(mut)
    assert result.errors is None
    assert result.data['changeOrigin']['toOrigin'] == 'OCS'
    assert result.data['changeOrigin']['fromOrigin'] == 'OCS'


@pytest.mark.asyncio
async def test_load_files_mutation():
    _resources_data_path = Path(ROOT_DIR) / 'scheduler' / 'services' / 'resource' / 'data'
    _calendar_path = Path(ROOT_DIR) / 'tests' / 'data' / 'telescope_schedule.xlsx'
    _gmos_fpu_path = _resources_data_path / 'GMOSS_FPUr201789.txt'
    _gmos_grat_path = _resources_data_path / 'GMOSS_GRAT201789.txt'

    with (_calendar_path.open('rb') as calendar,
          _gmos_fpu_path.open('r') as gmos_fpu,
          _gmos_grat_path.open('r') as gmos_grat):
        mut = """
            mutation loadFilesSuccess($service: Strings!,
                                      $site: Sites!,
                                      $calendar: Upload,
                                      $fpu: Upload,
                                      $gratings: Upload){
                loadSourcesFiles(filesInput: {
                    service: "RESOURCE",
                    calendar: calendar1,
                    gmosFpus: fpu1,
                    gmosGratings: grat1
                    site: "GS"
                }) {
                    service,
                    loaded,
                    msg
                }
                }
        """
        res = await schema.execute(mut, variable_values={
            'service': 'RESOURCE',
            'site': 'GS',
            'calendar': calendar,
            'fpu': gmos_fpu,
            'gratings': gmos_grat,
        })
        assert res
