import os
import logging
from common.minimodel import *

from api.observatory.abstract import ObservatoryProperties
from api.observatory.gemini import GeminiProperties
from api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from common.output import print_collector_info, print_plans
from components.collector import *
from components.optimizer.dummy import DummyOptimizer
from components.selector import Selector
from components.optimizer import Optimizer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join('..', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector = Collector(
        start_time=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
        end_time=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
        time_slot_length=TimeDelta(1.0 * u.min),
        sites=ALL_SITES,
        semesters={Semester(2018, SemesterHalf.B)},
        program_types={ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
        obs_classes={ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
    )
    collector.load_programs(program_provider=OcsProgramProvider(),
                            data=programs)

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)

    selector = Selector(collector=collector)

    # Execute the Selector.
    # Not sure the best way to display the output.
    selection = selector.select()
    program_data = selection.program_info['GN-2018B-Q-104']
    group_data = program_data.group_data['GN-2018B-Q-104-11']
    group = group_data.group
    print(group.exec_time())
    print(group.total_used())
    print(group.instruments())

    # Notes for data access:
    # The Selector returns all the data that an Optimizer needs in order to generate plans.
    # This comprises a Selection object, which has fields:
    #    program_ids: set of ProgramID of programs with schedulable groups ONLY
    #    program_info: a map from ProgramID to ProgramInfo
    #    night_events: the night events by Site (probably not necessary, see NightEvents below)
    #
    # *** ProgramInfo ***
    # This contains:
    #     program: a reference to the Program object in the mini-model
    #     group_data: a map from GroupID to GroupData (which has group and group_info members) for SCHEDULABLE groups
    #     observations: a map from ObservationID to reference to Observation for SCHEDULABLE observations
    #     target_info: a map from ObservationID to TargetInfo, which contains info for the observation's target
    #                  (see TargetInfo below)
    #     observation_ids: set of ObservationID that are schedulable
    #     group_ids: set of GroupID that are schedulable
    #
    # *** NightEvents ***
    # I don't know if this will be needed, but the ProgramInfo includes NightEvents calculations.
    # The NightEvents are the calculations for a given site across all nights.
    #
    # This may have to be modified in the future to accept date ranges, since now they return everything the Collector
    # was initialized with in terms of period length and time granularity.
    #
    # For all the values:
    #     midnight
    #     sunset
    #     sunrise
    #     twilight_evening_12
    #     twilight_morning_12
    #     moonrise
    #     moonset
    # they are AstroPy Time arrays. If you index them by night index, you get the specific Time for the night given.
    # When initialized, the following other fields are populated, also indexed by night index:
    #     night_length: Time in hours for each night
    #     times: Index by night index to get Time array of the Time of each time slot during a night in jday format
    #     utc_times: As per times above, but Python datetime objects representing the times in UTC
    #     local_times: Same as utc_times, but represents local times in the timezone associated with the site
    #     local_sidereal_times: Same as local_times, but uses vskyutil library to calculate local sidereal times
    #     sun_pos: Index by night index to get SkyCoord array of sun position for each time slot during a night
    #     sun_alt, sun_az, sun_par_ang: Same as above but in Alt - Az, and parallactic angle
    #     moon_pos, moon_dist: Similar to above
    #     moon_alt, moon_az, moon_par_ang: Similar to above
    #     sun_moon_ang: Array indexed by night index to get Angle array giving angle for each time slot during night
    #
    # *** TargetInfo ***
    # Contains the TargetInfo for a night broken into time slots over that night.
    # Contains the following fields:
    #      coord: SkyCoord array representing the position of the target for each time slot during the night
    #             Uses proper motion for SiderealTargets, ephemeris data for NonsiderealTargets
    #      alt, az, par_ang: Angle array for each time slot
    #      hourangle: Angle array for each time slot
    #      airmass: numpy array of float for each time slot
    #      sky_brightness: numpy array of SkyBackground for each time slot
    #                      Values are converted from calculations to bins as per common.sky.brightness.py
    #                          method convert_to_sky_background, L255.
    #      visibility_slot_idx: numpy array of indices of the time slots where target is visible (using np.where)
    #      visibility_time: a single Time entry, the length of the visibility_slot_idx times the slot length
    #      rem_visibility_time: a single Time entry, the remaining visibility time of the target from this day in the
    #                           period onward
    #      rem_visibility_frac: float, remaining observation time / rem_visibility_time
    #
    # *** GroupInfo ***
    # This is calculated by the Selector for each AND group.
    # If an AND group contains an OR group, at this point, it throws a NotImplementedError.
    #
    # The GroupInfo object for a Group contains the information for the given Group.
    #       minimum_conditions: Conditions object representing the minimum required conditions by group and all
    #           subgroups
    #       is_splittable: bool indicating if the group can be split
    #       standards: float, time in h (not yet populated) representing the sum of the standards for group and all
    #            subgroups
    #       resource_night_availability: numpy array of bool indexed by night index indicating if all resources for
    #            group and subgroups are available
    #       conditions_score: list indexed by night index with entries numpy array of floats for each time slot
    #            indicating the score component contributed by the predicted conditions being able to meet the
    #            minimum conditions (with penalties if actual conditions are better)
    #       wind_score: same as above but for wind
    #       schedulable_slot_indices: list indexed by night index with entries numpy array of time slot indices where
    #             the group can be scheduled (determined using numpy.where)
    #       scores: list indexed by night index with entries numpy array of float indicating the final scores for the
    #             group for each time slot in the night
    #
    # Note that the actual scores are generated using the Ranker (components.ranker.app.py, Ranker class), which
    # follows the old implementation but is generalized to multi-night, and cleaned up significantly.

    # BRYAN:
    # To get group information for any group (including ones that are not schedulable / have zero scores),
    # go through the Selector:
    #
    # selector.get_group_info(group_id)
    #
    # Selection only includes data for schedulable things.
    #
    # To get, for example, scores:
    #
    # selector.get_group_info(group_id).scores
    # selection.program_info(program_id).group_data(group_id).group_info.scores

    # Sergio preliminary work:
    # Output the data in a spreadsheet.
    # for program in collector._programs.values():
    #    if program.id == 'GS-2018B-Q-101':
    #        atoms_to_sheet(program)
    
    # gm = GreedyMax(some_parameter=1)  # Set parameters for specific algorithm
    # print(selection.program_info)
    dummy = DummyOptimizer()
    optimizer = Optimizer(selection, algorithm=dummy)
    plans = optimizer.schedule()
    print_plans(plans)

    import graphqlserver
    planmanager = graphqlserver.PlanManager()
    planmanager.set_plans(plans)
    graphqlserver.start_graphql_server()
    print('DONE')
