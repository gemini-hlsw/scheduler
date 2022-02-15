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
    # *** SHARED RESOURCES ***
    gmosn = Resource(id='GMOS-N')
    gmos_oiwfs = Resource(id='GMOS OIWFS')

    gnirs = Resource(id='GNIRS')
    pwfs2 = Resource(id='PWFS2')

    # *** GMOSN-2 OBSERVATION ***
    gmosn2_conditions = Conditions(
        cc=CloudCover.CCANY,
        iq=ImageQuality.IQANY,
        sb=SkyBackground.SBANY,
        wv=WaterVapor.WVANY
    )

    gmosn2_constraints = Constraints(
        conditions=gmosn2_conditions,
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
        epoch=2000.0
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
        epoch=2000.0
    )

    gmosn2_targets = [gmosn2_target_1, gmosn2_target_2]
    gmosn2_guiding = {
        gmos_oiwfs: gmosn2_target_2
    }

    gmosn2_sequence = [
        Atom(
            id=1,
            exec_time=timedelta(microseconds=84300),
            prog_time=timedelta(microseconds=84300),
            part_time=timedelta(),
            observed=False,
            qa_state=QAState.NONE,
            guide_state=False,
            resources={gmosn},
            wavelength={0.475}
        )
    ]

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
        acq_overhead=timedelta(minutes=6),
        # TODO: Check this. This is based on the atoms calculation.
        exec_time=timedelta(seconds=360, microseconds=84300),
        obs_class=ObservationClass.SCIENCE,
        targets=gmosn2_targets,
        guiding=gmosn2_guiding,
        sequence=gmosn2_sequence,
        constraints=gmosn2_constraints,
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the GMOSN-2 observation.
    gmosn2_group = AndGroup(
        id=gmosn2.id,
        group_name=gmosn2.title,
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gmosn2,
        group_option=AndOption.ANYORDER
    )

    # *** GNIRS2 OBSERVATION ***
    gnirs2_conditions = Conditions(
        cc=CloudCover.CCANY,
        iq=ImageQuality.IQANY,
        sb=SkyBackground.SB20,
        wv=WaterVapor.WVANY
    )

    gnirs2_constraints = Constraints(
        conditions=gnirs2_conditions,
        elevation_type=ElevationType.NONE,
        elevation_min=1.0,
        elevation_max=2.0,
        timing_windows=[
            TimingWindow(
                start=datetime(2022, 1, 11, 14, 7, 31, 128000),
                duration=timedelta(days=1),
                repeat=TimingWindow.NON_REPEATING,
                period=TimingWindow.NO_PERIOD
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
        pwfs2: gnirs2_target_2
    }

    gnirs2_sequence = [
        Atom(
            id=1,
            exec_time=timedelta(microseconds=26190),
            prog_time=timedelta(microseconds=26190),
            part_time=timedelta(),
            observed=False,
            qa_state=QAState.NONE,
            guide_state=False,
            resources={gnirs},
            wavelength={2.2}
        )
    ]

    gnirs2 = Observation(
        id='GN-2022A-Q-999-4',
        internal_id='aef545e2-c330-4c71-9521-c18a9cb3ee34',
        order=1,
        title='GNIRS-2',
        site=Site.GN,
        status=ObservationStatus.READY,
        active=True,
        priority=Priority.HIGH,
        resources=set(),
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(minutes=15),
        # TODO: Check this. This is based on the atoms calculation.
        exec_time=timedelta(seconds=900, microseconds=26190),
        obs_class=ObservationClass.SCIENCE,
        targets=gnirs2_targets,
        guiding=gnirs2_guiding,
        sequence=gnirs2_sequence,
        constraints=gnirs2_constraints,
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the GNIRS-2 observation.
    gnirs2_group = AndGroup(
        id=gnirs2.id,
        group_name=gnirs2.title,
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gnirs2,
        group_option=AndOption.ANYORDER
    )

    # *** AND GROUP CONTAINING THE GMOSN-2 AND GNIRS-2 GROUPS ***
    sched_group = AndGroup(
        id='2',
        group_name='TestGroup',
        number_to_observe=2,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=[gmosn2_group, gnirs2_group],
        group_option=AndOption.CONSEC_ORDERED
    )

    # *** GNIRS1 OBSERVATION ***
    gnirs1_conditions = Conditions(
        cc=CloudCover.CC50,
        iq=ImageQuality.IQ20,
        sb=SkyBackground.SB20,
        wv=WaterVapor.WV20
    )

    gnirs1_constraints = Constraints(
        conditions=gnirs1_conditions,
        elevation_type=ElevationType.NONE,
        elevation_min=0.0,
        elevation_max=0.0,
        timing_windows=[],
        strehl=None
    )

    gnirs1_target_1 = SiderealTarget(
        name='M10',
        magnitudes={
            Magnitude(MagnitudeBands.g, value=6.842),
            Magnitude(MagnitudeBands.V, value=4.98),
            Magnitude(MagnitudeBands.K, value=3.6)
        },
        type=TargetType.BASE,
        ra=254.2877083333333,
        dec=5338.495333333333,
        pm_ra=-4.72,
        pm_dec=-6.54,
        epoch=2000.0
    )

    gnirs1_target_2 = SiderealTarget(
        name='430-067087',
        magnitudes={
            Magnitude(MagnitudeBands.V, value=11.78),
            Magnitude(MagnitudeBands.K, value=8.916),
            Magnitude(MagnitudeBands.i, value=11.04),
            Magnitude(MagnitudeBands.UC, value=11.637),
            Magnitude(MagnitudeBands.g, value=12.321),
            Magnitude(MagnitudeBands.B, value=12.927),
            Magnitude(MagnitudeBands.r, value=11.389),
            Magnitude(MagnitudeBands.H, value=9.082),
            Magnitude(MagnitudeBands.J, value=9.628)
        },
        type=TargetType.GUIDESTAR,
        ra=254.3009583333333,
        dec=5337.1376666666665,
        pm_ra=-2.7,
        pm_dec=5.7,
        epoch=2000.0
    )

    gnirs1_targets = [gnirs1_target_1, gnirs1_target_2]

    gnirs1_guiding = {
        pwfs2: gnirs1_target_2
    }

    gnirs1_sequence = [
        Atom(
            id=1,
            exec_time=timedelta(microseconds=26190),
            prog_time=timedelta(microseconds=26190),
            part_time=timedelta(),
            observed=False,
            qa_state=QAState.NONE,
            guide_state=False,
            resources={gnirs},
            wavelength={2.2}
        )
    ]

    gnirs1_observation = Observation(
        id='GN-2022A-Q-999-2',
        internal_id='f1e411e3-ec93-430a-ac1d-1c5db3a103e6',
        order=0,
        title='GNIRS-1',
        site=Site.GN,
        status=ObservationStatus.PHASE2,
        active=True,
        priority=Priority.LOW,
        resources=set(),
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(minutes=15),
        exec_time=timedelta(seconds=900, microseconds=26190),
        obs_class=ObservationClass.SCIENCE,
        targets=gnirs1_targets,
        guiding=gnirs1_guiding,
        sequence=gnirs1_sequence,
        constraints=gnirs1_constraints,
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the gnirs1 observation.
    gnirs1_group = AndGroup(
        id=gnirs1_observation.id,
        group_name='GNIRS-1',
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gnirs1_observation,
        group_option=AndOption.ANYORDER
    )

    # *** GMOSN-1 OBSERVATION ***
    gmosn1_conditions = Conditions(
        cc=CloudCover.CC70,
        iq=ImageQuality.IQ20,
        sb=SkyBackground.SB50,
        wv=WaterVapor.WVANY
    )

    gmosn1_timing_windows = [
        TimingWindow(
            start=datetime(2022, 1, 11, 12, 27, 30, 498000),
            duration=timedelta(hours=24),
            repeat=TimingWindow.NON_REPEATING,
            period=TimingWindow.NO_PERIOD
        ),
        # TODO: ??? -0.1 second duration???
        TimingWindow(
            start=datetime(2022, 1, 12, 12, 28, 42),
            duration=TimingWindow.INFINITE_DURATION,
            repeat=TimingWindow.NON_REPEATING,
            period=TimingWindow.NO_PERIOD
        ),
        TimingWindow(
            start=datetime(2022, 1, 13, 12, 29, 37),
            duration=timedelta(hours=12),
            repeat=TimingWindow.OCS_INFINITE_REPEATS,
            period=timedelta(hours=36)
        ),
        TimingWindow(
            start=datetime(2022, 1, 14, 12, 32, 31),
            duration=timedelta(days=1),
            repeat=10,
            period=timedelta(days=2)
        )
    ]

    gmosn1_constraints = Constraints(
        conditions=gmosn1_conditions,
        elevation_type=ElevationType.HOUR_ANGLE,
        elevation_min=-4.0,
        elevation_max=4.0,
        timing_windows=gmosn1_timing_windows,
        strehl=None
    )

    gmosn1_target_1 = SiderealTarget(
        name='M15',
        magnitudes={
            Magnitude(MagnitudeBands.z, value=6.288),
            Magnitude(MagnitudeBands.r, value=6.692),
            Magnitude(MagnitudeBands.B, value=3.0),
            Magnitude(MagnitudeBands.i, value=6.439),
            Magnitude(MagnitudeBands.g, value=7.101)
        },
        type=TargetType.BASE,
        ra=322.49304166666667,
        dec=182.505,
        pm_ra=-0.63,
        pm_dec=-3.8,
        epoch=2000.0
    )

    gmosn1_target_2 = SiderealTarget(
        name='512-132424',
        magnitudes={
            Magnitude(MagnitudeBands.i, value=11.833),
            Magnitude(MagnitudeBands.J, value=10.455),
            Magnitude(MagnitudeBands.H, value=9.796),
            Magnitude(MagnitudeBands.r, value=12.37),
            Magnitude(MagnitudeBands.UC, value=12.388),
            Magnitude(MagnitudeBands.B, value=14.185),
            Magnitude(MagnitudeBands.K, value=9.695),
            Magnitude(MagnitudeBands.g, value=13.473),
            Magnitude(MagnitudeBands.V, value=12.834)
        },
        type=TargetType.GUIDESTAR,
        ra=322.47885,
        dec=183.343625,
        pm_ra=-7.2,
        pm_dec=-6.8,
        epoch=2000.0
    )

    gmosn1_target_3 = SiderealTarget(
        name='512-132390',
        magnitudes={
            Magnitude(MagnitudeBands.g, value=16.335),
            Magnitude(MagnitudeBands.B, value=16.708),
            Magnitude(MagnitudeBands.H, value=13.997),
            Magnitude(MagnitudeBands.V, value=16.062),
            Magnitude(MagnitudeBands.UC, value=16.017),
            Magnitude(MagnitudeBands.K, value=13.845),
            Magnitude(MagnitudeBands.r, value=15.77),
            Magnitude(MagnitudeBands.J, value=14.455)
        },
        type=TargetType.TUNING_STAR,
        ra=322.4453041666667,
        dec=183.24004166666666,
        pm_ra=4.9,
        pm_dec=0.9,
        epoch=2000.0
    )

    gmosn1_target_4 = SiderealTarget(
        name='511-136970',
        magnitudes={
            Magnitude(MagnitudeBands.H, 13.003),
            Magnitude(MagnitudeBands.K, 12.884),
            Magnitude(MagnitudeBands.UC, 14.91),
            Magnitude(MagnitudeBands.J, 13.504)
        },
        type=TargetType.BLIND_OFFSET,
        ra=322.4290291666667,
        dec=182.47258333333335,
        pm_ra=-21.0,
        pm_dec=-12.9,
        epoch=2000.0
    )

    gmosn1_targets = [gmosn1_target_1,
                      gmosn1_target_2,
                      gmosn1_target_3,
                      gmosn1_target_4]

    gmonsn1_guiding = {
        gmos_oiwfs: gmosn1_target_2
    }

    gmosn1_sequence = [
        Atom(
            id=1,
            exec_time=timedelta(microseconds=392500),
            prog_time=timedelta(microseconds=392500),
            part_time=timedelta(),
            observed=False,
            qa_state=QAState.NONE,
            guide_state=False,
            resources={gmosn},
            wavelength={0.475}
        )
    ]

    gmosn1_observation = Observation(
        id='GN-2022A-Q-999-1',
        internal_id='acc39a30-97a8-42de-98a6-5e77cc95d3ec',
        order=0,
        title='GMOSN-1',
        site=Site.GN,
        status=ObservationStatus.ONGOING,
        active=True,
        priority=Priority.LOW,
        resources=set(),
        setuptime_type=SetupTimeType.REACQUISITION,
        acq_overhead=timedelta(minutes=5),
        exec_time=timedelta(seconds=300, microseconds=392500),
        obs_class=ObservationClass.SCIENCE,
        targets=gmosn1_targets,
        guiding=gmonsn1_guiding,
        sequence=gmosn1_sequence,
        constraints=gmosn1_constraints,
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the gnirs1 observation.
    gmosn1_group = AndGroup(
        id=gmosn1_observation.id,
        group_name='GMOSN-1',
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gmosn1_observation,
        group_option=AndOption.ANYORDER
    )

    # *** ROOT GROUP ***
    root_children = [sched_group, gnirs1_group, gmosn1_group]
    # root_children = [sched_group, gmosn1_group, gnirs1_group]

    root_group = AndGroup(
        id='Root',
        group_name='Root',
        number_to_observe=3,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=root_children,
        group_option=AndOption.CONSEC_ORDERED
    )

    # *** TIME ALLOCATION ***
    time_allocation_us = TimeAllocation(
        category=TimeAccountingCode.US,
        program_awarded=timedelta(hours=4),
        partner_awarded=timedelta(hours=2),
        program_used=timedelta(),
        partner_used=timedelta()
    )

    time_allocation_ca = TimeAllocation(
        category=TimeAccountingCode.CA,
        program_awarded=timedelta(hours=2),
        partner_awarded=timedelta(hours=1),
        program_used=timedelta(),
        partner_used=timedelta()
    )

    time_allocation = {time_allocation_us, time_allocation_ca}

    # *** PROGRAM ***
    return Program(
        id='GN-2022A-Q-999',
        internal_id='c396b9c9-9bdd-4eec-be83-81162090d032',
        semester=Semester(year=2022, half=SemesterHalf.A),
        band=Band.BAND2,
        thesis=True,
        mode=ProgramMode.QUEUE,
        type=ProgramTypes.Q,
        start=datetime(2022, 8, 1, 0, 0) - Program.FUZZY_BOUNDARY,
        end=datetime(2023, 1, 31, 0, 0) + Program.FUZZY_BOUNDARY,
        allocated_time=time_allocation,
        root_group=root_group,
        too_type=TooType.RAPID
    )


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

