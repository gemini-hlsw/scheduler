# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
from pathlib import Path
from datetime import datetime, timedelta

from lucupy.helpers import dmsstr2deg
from lucupy.minimodel import (AndGroup, AndOption, Atom, Band, CloudCover, Conditions, Constraints, ElevationType,
                              GroupID, ImageQuality, Magnitude, MagnitudeBands, ObservationClass, ObservationID,
                              ObservationMode, ObservationStatus, Priority, Program, ProgramID, ProgramMode,
                              ProgramTypes, QAState, Resource, ROOT_GROUP_ID, Semester, SemesterHalf, SetupTimeType,
                              SiderealTarget, Site, SkyBackground, TargetName, TargetType, TimeAccountingCode,
                              TimeAllocation, TimingWindow, TooType, WaterVapor, Wavelength)
from lucupy.observatory.gemini.geminiobservation import GeminiObservation
from lucupy.timeutils import sex2dec

from scheduler.core.programprovider.ocs import OcsProgramProvider
from scheduler.core.sources import Sources
from lucupy.types import ZeroTime


def get_api_program() -> Program:
    """
    Load the GN-2022A-Q-999 program from the JSON file.
    """
    sources = Sources()
    with open(Path('tests') / 'data' / 'GN-2022A-Q-999.json') as f:
        data = json.loads(f.read())
        obs_classes = frozenset({ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL})
        return OcsProgramProvider(obs_classes, sources).parse_program(data['PROGRAM_BASIC'])


def create_minimodel_program() -> Program:
    """
    Create the GN-2022A-Q-999 program and all its properties directly from the
    mini-model.

    Note that we do not have to worry about atoms as we have no obslog data.
    """
    program_id = ProgramID('GN-2022A-Q-999')

    # *** SHARED RESOURCES ***
    gmosn = Resource(id='GMOS-N')
    mirror = Resource(id='mirror')
    gmos_oiwfs = Resource(id='GMOS OIWFS')

    gnirs = Resource(id='GNIRS')
    pwfs2 = Resource(id='PWFS2')

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
                start=datetime(2022, 1, 12, 0, 7, 31, 128000),
                duration=timedelta(days=1),
                repeat=TimingWindow.NON_REPEATING,
                period=TimingWindow.NO_PERIOD
            )
        ],
        strehl=None
    )

    gnirs2_target_1 = SiderealTarget(
        name=TargetName('M22'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.B, 7.16),
            Magnitude(MagnitudeBands.V, 6.17),
            Magnitude(MagnitudeBands.K, 1.71)
        }),
        type=TargetType.BASE,
        ra=sex2dec('18:36:23.940', todegree=True),
        dec=dmsstr2deg('336:05:42.90'),
        pm_ra=9.82,
        pm_dec=-5.54,
        epoch=2000.0
    )

    gnirs2_target_2 = SiderealTarget(
        name=TargetName('331-171970'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.B, 12.888),
            Magnitude(MagnitudeBands.g, 11.93),
            Magnitude(MagnitudeBands.V, 11.051),
            Magnitude(MagnitudeBands.UC, 10.586),
            Magnitude(MagnitudeBands.r, 10.343),
            Magnitude(MagnitudeBands.i, 9.839),
            Magnitude(MagnitudeBands.J, 7.754),
            Magnitude(MagnitudeBands.H, 6.993),
            Magnitude(MagnitudeBands.K, 6.769)
        }),
        type=TargetType.GUIDESTAR,
        ra=sex2dec('18:36:36.196', todegree=True),
        dec=dmsstr2deg('336:00:20.55'),
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
            id=0,
            exec_time=timedelta(microseconds=26190),
            prog_time=timedelta(microseconds=26190),
            program_used=ZeroTime,
            partner_used=ZeroTime,
            part_time=ZeroTime,
            observed=False,
            qa_state=QAState.NONE,
            guide_state=True,
            resources=frozenset({gnirs}),
            wavelengths=frozenset({Wavelength(2.2)}),
            obs_mode=ObservationMode.IMAGING
        )
    ]

    gnirs2 = GeminiObservation(
        id=ObservationID('GN-2022A-Q-999-4'),
        internal_id='aef545e2-c330-4c71-9521-c18a9cb3ee34',
        order=1,
        title='GNIRS-2',
        site=Site.GN,
        status=ObservationStatus.READY,
        active=True,
        priority=Priority.HIGH,
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(minutes=15),
        obs_class=ObservationClass.SCIENCE,
        targets=gnirs2_targets,
        guiding=gnirs2_guiding,
        sequence=gnirs2_sequence,
        constraints=gnirs2_constraints,
        belongs_to=ProgramID('GN-2022A-Q-999'),
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the GNIRS-2 observation.
    gnirs2_group = AndGroup(
        id=GroupID(gnirs2.id.id),
        program_id=program_id,
        group_name=gnirs2.title,
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gnirs2,
        group_option=AndOption.ANYORDER
    )

    # *** AND GROUP CONTAINING THE GMOSN-2 AND GNIRS-2 GROUPS ***
    sched_group = AndGroup(
        id=GroupID('2'),
        program_id=program_id,
        group_name='TestGroup',
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=[gnirs2_group],
        group_option=AndOption.CONSEC_ORDERED
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
            start=datetime(2022, 1, 11, 22, 27, 30, 498000),
            duration=timedelta(hours=24),
            repeat=TimingWindow.NON_REPEATING,
            period=TimingWindow.NO_PERIOD
        ),
        TimingWindow(
            start=datetime(2022, 1, 12, 22, 28, 42),
            duration=TimingWindow.INFINITE_DURATION,
            repeat=TimingWindow.NON_REPEATING,
            period=TimingWindow.NO_PERIOD
        ),
        TimingWindow(
            start=datetime(2022, 1, 13, 22, 29, 37),
            duration=timedelta(hours=12),
            repeat=TimingWindow.OCS_INFINITE_REPEATS,
            period=timedelta(hours=36)
        ),
        TimingWindow(
            start=datetime(2022, 1, 14, 22, 32, 31),
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
        name=TargetName('M15'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.z, value=6.288),
            Magnitude(MagnitudeBands.r, value=6.692),
            Magnitude(MagnitudeBands.B, value=3.0),
            Magnitude(MagnitudeBands.i, value=6.439),
            Magnitude(MagnitudeBands.g, value=7.101)
        }),
        type=TargetType.BASE,
        ra=sex2dec('21:29:58.330', todegree=True),
        dec=dmsstr2deg('12:10:01.20'),
        pm_ra=-0.63,
        pm_dec=-3.8,
        epoch=2000.0
    )

    gmosn1_target_2 = SiderealTarget(
        name=TargetName('512-132424'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.i, value=11.833),
            Magnitude(MagnitudeBands.J, value=10.455),
            Magnitude(MagnitudeBands.H, value=9.796),
            Magnitude(MagnitudeBands.r, value=12.37),
            Magnitude(MagnitudeBands.UC, value=12.388),
            Magnitude(MagnitudeBands.B, value=14.185),
            Magnitude(MagnitudeBands.K, value=9.695),
            Magnitude(MagnitudeBands.g, value=13.473),
            Magnitude(MagnitudeBands.V, value=12.834)
        }),
        type=TargetType.GUIDESTAR,
        ra=sex2dec('21:29:54.924', todegree=True),
        dec=dmsstr2deg('12:13:22.47'),
        pm_ra=-7.2,
        pm_dec=-6.8,
        epoch=2000.0
    )

    gmosn1_target_3 = SiderealTarget(
        name=TargetName('512-132390'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.g, value=16.335),
            Magnitude(MagnitudeBands.B, value=16.708),
            Magnitude(MagnitudeBands.H, value=13.997),
            Magnitude(MagnitudeBands.V, value=16.062),
            Magnitude(MagnitudeBands.UC, value=16.017),
            Magnitude(MagnitudeBands.K, value=13.845),
            Magnitude(MagnitudeBands.r, value=15.77),
            Magnitude(MagnitudeBands.J, value=14.455)
        }),
        type=TargetType.TUNING_STAR,
        ra=sex2dec('21:29:46.873', todegree=True),
        dec=dmsstr2deg('12:12:57.61'),
        pm_ra=4.9,
        pm_dec=0.9,
        epoch=2000.0
    )

    gmosn1_target_4 = SiderealTarget(
        name=TargetName('511-136970'),
        magnitudes=frozenset({
            Magnitude(MagnitudeBands.H, 13.003),
            Magnitude(MagnitudeBands.K, 12.884),
            Magnitude(MagnitudeBands.UC, 14.91),
            Magnitude(MagnitudeBands.J, 13.504)
        }),
        type=TargetType.BLIND_OFFSET,
        ra=sex2dec('21:29:42.967', todegree=True),
        dec=dmsstr2deg('12:09:53.42'),
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
            id=0,
            exec_time=timedelta(microseconds=392500),
            prog_time=timedelta(microseconds=392500),
            part_time=ZeroTime,
            program_used=ZeroTime,
            partner_used=ZeroTime,
            observed=False,
            qa_state=QAState.NONE,
            guide_state=True,
            resources=frozenset({gmosn, mirror}),
            wavelengths=frozenset({Wavelength(0.475)}),
            obs_mode=ObservationMode.IMAGING
        )
    ]

    gmosn1_observation = GeminiObservation(
        id=ObservationID('GN-2022A-Q-999-1'),
        internal_id='acc39a30-97a8-42de-98a6-5e77cc95d3ec',
        order=3,
        title='GMOSN-1',
        site=Site.GN,
        status=ObservationStatus.ONGOING,
        active=True,
        priority=Priority.LOW,
        setuptime_type=SetupTimeType.REACQUISITION,
        acq_overhead=timedelta(minutes=5),
        obs_class=ObservationClass.SCIENCE,
        targets=gmosn1_targets,
        guiding=gmonsn1_guiding,
        sequence=gmosn1_sequence,
        constraints=gmosn1_constraints,
        belongs_to=program_id,
        too_type=TooType.RAPID
    )

    # Create the trivial AND group containing the gnirs1 observation.
    gmosn1_group = AndGroup(
        id=GroupID(gmosn1_observation.id.id),
        program_id=program_id,
        group_name='GMOSN-1',
        number_to_observe=1,
        delay_min=timedelta.min,
        delay_max=timedelta.max,
        children=gmosn1_observation,
        group_option=AndOption.ANYORDER
    )

    # *** ROOT GROUP ***
    # root_children = [sched_group, gnirs1_group, gmosn1_group]
    root_children = [sched_group, gmosn1_group]

    root_group = AndGroup(
        id=ROOT_GROUP_ID,
        program_id=program_id,
        group_name=ROOT_GROUP_ID.id,
        number_to_observe=2,
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
        program_used=ZeroTime,
        partner_used=ZeroTime
    )

    time_allocation_ca = TimeAllocation(
        category=TimeAccountingCode.CA,
        program_awarded=timedelta(hours=2),
        partner_awarded=timedelta(hours=1),
        program_used=ZeroTime,
        partner_used=ZeroTime
    )

    time_allocation = frozenset({time_allocation_us, time_allocation_ca})

    # *** PROGRAM ***
    return Program(
        id=program_id,
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
