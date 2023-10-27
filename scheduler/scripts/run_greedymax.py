# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import logging

from lucupy.minimodel import ALL_SITES, CloudCover, ImageQuality, NightIndex
from lucupy.minimodel.semester import SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.builder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.eventsqueue import EventQueue
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.statscalculator import StatCalculator
from scheduler.services import logger_factory


if __name__ == '__main__':
    logger = logger_factory.create_logger(__name__, logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )

    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    # num_nights_to_schedule = int(round(end.jd - start.jd)) + 1
    num_nights_to_schedule = 3

    builder = ValidationBuilder(Sources(),
                                EventQueue([i for i in range(num_nights_to_schedule)], ALL_SITES))

    collector = builder.build_collector(
        start=start,
        end=end,
        sites=ALL_SITES,
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    # Create the Collector and load the programs.
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)

    ValidationBuilder.update_collector(collector)  # ZeroTime observations

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)

    # Execute the Selector.
    # Not sure the best way to display the output.
    # TODO: Iterate over num_nights_to_schedule and schedule each night.
    # TODO: How does this affect the selection? We need to select on all three nights or the
    # TODO: scoring doesn't work and gives us an entirely different result, but it seems like
    # TODO: we should be only selecting for the night for which we want to schedule.
    # TODO: Talk to Bryan about this. What will happen if we have more observations? Will we still
    # TODO: only schedule one night regardless of how many nights we select?
    # TODO: Note that we are still *not using* num_nights_to_schedule. This should probably be used
    # TODO: by the loop below.

    # TODO: SET THE WEATHER HERE.
    # CC values are CC50, CC70, CC85, CCANY. Default is CC50 if not passed to build_selector.
    cc = CloudCover.CC50

    # IQ values are IQ20, IQ70, IQ85, and IQANY. Default is IQ70 if not passed to build_selector.
    iq = ImageQuality.IQ70

    selector = builder.build_selector(collector,
                                      num_nights_to_schedule=num_nights_to_schedule,
                                      default_cc=cc,
                                      default_iq=iq)

    # Prepare the optimizer.
    optimizer_blueprint = OptimizerBlueprint(
        "GreedyMax"
    )
    optimizer = builder.build_optimizer(
        blueprint=optimizer_blueprint
    )

    # The total nights for which visibility calculations have been done.
    total_nights = len(collector.time_grid)

    # Example: ensure that the first 400 time slots for night_idx 1 for GS are empty.
    # starting_time_slots = {Site.GS: {NightIndex(1): 400}}
    # starting_time_slots = None

    # Create the overall plans by night.
    overall_plans = {}
    for night_idx in range(selector.num_nights_to_schedule):
        # We score one night at a time.
        night_indices = np.array([night_idx])

        # Retrieve the Selection and run the Optimizer to get the plans.
        selection = selector.select(night_indices=night_indices)
        plans = optimizer.schedule(selection)

        # The Optimizer currently returns plans for all nights. We only want the first set.
        night_plans = plans[0]

        # Store the plans in the overall_plans array for that night.
        # TODO: This might be an issue. We may need to index nights (plans) in optimizer by night_idx.
        overall_plans[night_idx] = night_plans

        # Perform the time accounting on the plans.
        collector.time_accounting(night_plans)

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

    # gm = GreedyMax(some_parameter=1)  # Set parameters for specific algorithm
    # print(selection.program_info)

    # TODO: pass these as parameters
    # Period has been eliminated. This is now determined by the selection.
    # optimizer.period = 1  # number of nights for which to make plans in a single pass
    # optimizer_blueprint.algorithm.show_plots = True # show plots

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    print_plans(overall_plans)

    # Timeline tests
    # for tl in optimizer_blueprint.algorithm.timelines.values():
    #     print(f'Night {tl.night_idx + 1}')
    #     #     for site, ne in gm_optimizer.night_events.items():
    #     for site in optimizer.night_events.keys():
    #         print(f'\t {site}')
    #         print(f'\t {tl[site].start} {tl[site].end} {tl[site].total_time_slots} \
    #         {tl[site].is_full}')
    #         tl[site].print(optimizer_blueprint.algorithm.obs_group_ids)
    #         # print(tl.timelines[site].get_earliest_available_interval())

    print('DONE')
