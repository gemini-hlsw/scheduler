from common.minimodel import *
from common.timeutils import sex2dec
from api.ocs import OcsProgramProvider

from datetime import datetime, timedelta
import json
import os


def get_api_program() -> Program:
    """
    Load the GN-2022A-Q-999 program from the JSON file.
    """
    path = os.path.join('..', 'data', 'GN-2022A-Q-999.json')

    with open(path, 'r') as f:
        data = json.loads(f.read())
        return OcsProgramProvider.parse_program(data['PROGRAM_BASIC'])


def create_minimodel_program() -> Program:
    """
    Create the GN-2022A-Q-999 program and all its properties directly from the
    mini-model.

    Note that we do not have to worry about atoms as we have no obslog data.
    """
    # *** CREATE THE GMOSN-2 OBSERVATION ***
    gmosn2conditions = Conditions(
        cc=CloudCover.CCANY,
        iq=ImageQuality.IQANY,
        sb=SkyBackground.SBANY,
        wv=WaterVapor.WVANY,
    )

    gmosn2constraints = Constraints(
        conditions=gmosn2conditions,
        elevation_type=ElevationType.AIRMASS,
        elevation_min=1.1,
        elevation_max=2.1,
        timing_windows=[],
        strehl=None
    )

    gmosn2_target_1 = SiderealTarget(
        name='M11',
        magnitudes={
            Magnitude(MagnitudeBands.B, 6.32),
            Magnitude(MagnitudeBands.V, 5.8)
        },
        type=TargetType.BASE,
        ra=sex2dec('18:51:03.840', todegree=True),
        dec=sex2dec('353:43:40.80', todegree=True),
        pm_ra=-1.568,
        pm_dec=-4.144,
        epoch=2000.0,
    )

    gmosn2_target_2 = SiderealTarget(
        name='419-102509',
        magnitudes={
            Magnitude(MagnitudeBands.B, 12.261),
            Magnitude(MagnitudeBands.g, 12.046),
            Magnitude(MagnitudeBands.V, 11.983),
            Magnitude(MagnitudeBands.UC, 12.0),
            Magnitude(MagnitudeBands.r, 11.927),
            Magnitude(MagnitudeBands.i, 11.875),
            Magnitude(MagnitudeBands.J, 11.11),
            Magnitude(MagnitudeBands.H, 11.0),
            Magnitude(MagnitudeBands.K, 10.894)
        },
        type=TargetType.GUIDESTAR,
        ra=sex2dec('18:50:50.990', todegree=True),
        dec=sex2dec('353:44:28.68', todegree=True),
        pm_ra=-0.6,
        pm_dec=-7.4,
        epoch=2000
    )

    gmosn2_targets = [gmosn2_target_1, gmosn2_target_2]
    gmosn2_guiding = {
        Resource(id='GMOS OIWFS', name='GMOS OIWFS'): gmosn2_target_2
    }

    gmosn2 = Observation(
        id='GN-2022A-Q-999-3',
        internal_id='1a4f101b-de28-4ed1-959f-607b6618705c',
        order=0,
        title='GMOSN-2',
        site=Site.GN,
        status=ObservationStatus.PHASE2,
        active=True,
        priority=Priority.LOW,
        resources=set(),
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(milliseconds=360000),
        exec_time=timedelta(),  # This is based on the atoms, which we have none.
        obs_class=ObservationClass.SCIENCE,
        targets=gmosn2_targets,
        guiding=gmosn2_guiding,
        sequence=[],
        constraints=gmosn2constraints,
        too_type=None
    )

    # Create the trivial AND group containing the gmosn2 observation.
    gmosn2_group = AndGroup(
        id=gmosn2.id,
        group_name=gmosn2.title,
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gmosn2,
        group_option=AndOption.ANYORDER
    )

    # *** CREATE THE GNIRS-2 OBSERVATION ***
    gnirs2conditions = Conditions(
        cc=CloudCover.CCANY,
        iq=ImageQuality.IQANY,
        sb=SkyBackground.SB20,
        wv=WaterVapor.WVANY
    )

    gnirs2constraints = Constraints(
        conditions=gnirs2conditions,
        elevation_type=ElevationType.NONE,
        elevation_min=1.0,
        elevation_max=2.0,
        timing_windows=[
            TimingWindow(
                start=datetime.fromtimestamp(1641946051128),
                duration=timedelta(milliseconds=86400000),
                repeat=0,
                period=None
            )
        ],
        strehl=None
    )

    gnirs2_target_1 = SiderealTarget(
        name='M22',
        magnitudes={
            Magnitude(MagnitudeBands.B, 7.16),
            Magnitude(MagnitudeBands.V, 6.17),
            Magnitude(MagnitudeBands.K, 1.71)
        },
        type=TargetType.BASE,
        ra=sex2dec('18:36:23.940', todegree=True),
        dec=sex2dec('336:05:42.90', todegree=True),
        pm_ra=9.82,
        pm_dec=-5.54,
        epoch=2000.0
    )

    gnirs2_target_2 = SiderealTarget(
        name='331-171970',
        magnitudes={
            Magnitude(MagnitudeBands.B, 12.888),
            Magnitude(MagnitudeBands.g, 11.93),
            Magnitude(MagnitudeBands.V, 11.051),
            Magnitude(MagnitudeBands.UC, 10.586),
            Magnitude(MagnitudeBands.r, 10.343),
            Magnitude(MagnitudeBands.i, 9.839),
            Magnitude(MagnitudeBands.J, 7.754),
            Magnitude(MagnitudeBands.H, 6.993),
            Magnitude(MagnitudeBands.K, 6.769)
        },
        type=TargetType.GUIDESTAR,
        ra=sex2dec('18:36:36.196', todegree=True),
        dec=sex2dec('336:00:20.55', todegree=True),
        pm_ra=10.6,
        pm_dec=0.1,
        epoch=2000.0
    )

    gnirs2_targets = [gnirs2_target_1, gnirs2_target_2]
    gnirs2_guiding = {
        Resource('PWFS2', 'PWFS2'): gnirs2_target_2
    }

    gnirs2 = Observation(
        id='GN-2022A-Q-999-4',
        internal_id='aef545e2-c330-4c71-9521-c18a9cb3ee34',
        order=0,
        title='GNIRS-2',
        site=Site.GN,
        status=ObservationStatus.READY,
        active=True,
        priority=Priority.HIGH,
        resources=set(),
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(milliseconds=90000),
        exec_time=timedelta(),  # This is based on the atoms, which we have none.
        obs_class=ObservationClass.SCIENCE,
        targets=gnirs2_targets,
        guiding=gnirs2_guiding,
        sequence=[],
        constraints=gnirs2constraints,
        too_type=None
    )

    # Create the trivial AND group containing the gnirs2 observation.
    gnirs2_group = AndGroup(
        id=gnirs2.id,
        group_name=gnirs2.title,
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gnirs2,
        group_option=AndOption.ANYORDER
    )

    # Create the Scheduling Group (AND group) containing the gmosn2 and gnirs2 groups.
    sched_group = AndGroup(
        id='2',
        group_name='TestGroup',
        number_to_observe=2,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=[gmosn2_group, gnirs2_group],
        group_option=AndOption.CONSEC_ORDERED
    )

    # Continue on, creating the gnirs1 observation and the gmosn1 observation.
    # Put these in trivial AndGroups.
    # Connect all the AndGroups in an AndGroup, which is the root group of the Program.
    # Finally, create the Program and return it.

    return ...


def test_ocs_api():
    """
    Test the OCS API's ability to populate the mini-model.
    This involves two steps:
    1. Parse the JSON using the API.
    2. Create a mini-model representation of the program directly by instantiating objects from the mini-model
       and piecing them together. Look at the JSON to ensure that you are creating the objects in the right order
       for things like lists when it comes to groups and targets.

    Then compare using assert equality.
    """
    program1 = get_api_program()
    program2 = create_minimodel_program()
    assert program1 == program2
