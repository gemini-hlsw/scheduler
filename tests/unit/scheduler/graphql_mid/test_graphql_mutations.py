# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import pytest
from pathlib import Path
import os
from scheduler.graphql_mid.server import  schema
from definitions import ROOT_DIR


@pytest.mark.asyncio
async def test_change_origin_mutation():
    mut = """
        mutation changeOrigin{
                  changeOrigin(newOrigin:"OCS"){
                    toOrigin
                    fromOrigin
                  }
                }
    """

    result = await schema.execute(mut)
    assert result.errors is None
    assert result.data['changeOrigin']['toOrigin'] == 'OCS'
    assert result.data['changeOrigin']['fromOrigin'] == 'OCS'

@pytest.mark.asyncio
async def test_load_files_mutation():

    _resources_data_path = os.path.join(ROOT_DIR, 'scheduler', 'services', 'resource', 'data')
    calendar = Path(os.path.join(ROOT_DIR,'tests','data'),'telescope_schedule.xlsx').open('rb')
    gmos_fpu = Path(_resources_data_path,'GS_GMOS_FPUr201789.txt').open('rb')
    gmos_grat= Path(_resources_data_path, 'GS_GMOS_GRAT201789.txt').open('rb')
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
    },)