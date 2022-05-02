import os

from api.observatory.gemini import GeminiProperties
from api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from common.output import print_collector_info
from components.collector import *
from components.selector import Selector

if __name__ == '__main__':
    # Reduce logging to ERROR only to display the tqdm bars nicely.
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.ERROR)

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join('..', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector = Collector(
        start_time=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
        end_time=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
        time_slot_length=1.0 * u.min,
        sites=ALL_SITES,
        semesters={Semester(2018, SemesterHalf.B)},
        program_types={ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
        obs_classes={ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
    )
    collector.load_programs(program_provider=OcsProgramProvider(),
                            data=programs)

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)

    selector = Selector(
        collector=collector,
        properties=GeminiProperties
    )

    # Execute the Selector.
    # Not sure the best way to display the output.
    results = selector.select()

    # Notes for data access:
    #
    # *** PROGRAMS ***
    # 1. Programs are stored in the Collector. All the Program IDs in the Collector can be accessed with:
    #       Collector.get_program_ids()
    #    which returns an iterable of program IDs.
    # 2. To get a specific program:
    #       Collector.get_program(program_id)
    #    which returns the mini-model representation of the Program.
    #
    # *** OBSERVATIONS ***
    # 1. To get a list of the Observation IDs:
    #        Collector.get_observation_ids(optional prog_id, None is default)
    #    If there is a program ID supplied, only the observation IDs in that program are returned.
    #    If nothing is supplied (equivalent to None being supplied), all observation IDs are returned.
    # 2. To get a specific Observation:
    #        Collector.get_observation(observation_id)
    #    which returns the mini-model representation of the Observation.
    #
    # *** TARGETS ***
    # The Collector stores target calculations in objects of the TargetInfo class. There is one TargetInfo for each
    # target for each night.
    # 1. To get the base target of an Observation, you can go through the mini-model or more easily:
    #        Collector.get_base_target(observation_id)
    # 2. To get the TargetInfo associated with an Observation's base target:
    #        Collector.get_target_info(observation_id)
    #    If there is no target associated with the Observation, None is returned.
    #    Otherwise, a map is returned from the NightIndex in the period (0 for the first night, 1 for the second, etc,
    #    and we could just change this to a List, I suppose) to the TargetInfo for that night for that target.
    #
    # *** NIGHTEVENTS ***
    # The NightEvents class are the calculations for a given site. They include the values across all nights and
    # can be retrieved with:
    #     Collector.get_night_events(Site.GN or Site.GS)
    # This may have to be modified in the future to accept date ranges, since now they return everything the Collector
    # was initialized with in terms of period length and time granularity.
    # The NightEvents class is described in the component/Collector/__init__.py file at L19.
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
    # *** TARGETINFO ***
    # Contains the TargetInfo for a night broken into time slots over that night.
    # See TARGETS above on how to get the TargetInfo.
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
    # *** GROUPS ***
    # This is handled in the Selector. Groups can be accessed either through the mini-model Program obtained via
    # the Collector as described above in PROGRAMS, or via these Selector methods:
    # 1. selector_instance.get_group_ids(): returns a list of all group IDs
    # 2. selector_instance.get_group(group_id): gets specific group by ID
    #
    # *** GROUPINFO ***
    # This is calculated by the Selector for each AND group.
    # If an AND group contains an OR group, at this point, it throws a NotImplementedError.
    # Obtain by using:
    # 1. selector_instance.get_group_info(group_id): returns a map from night index to GroupInfo
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
    # Note that the actual scores are generated using the Ranker (components.ranker.__init__.py, Ranker class), which
    # follows the old implementation but is generalized to multi-night, and cleaned up significantly.


    # Sergio preliminary work:
    # Output the data in a spreadsheet.
    # for program in collector._programs.values():
    #    if program.id == 'GS-2018B-Q-101':
    #        atoms_to_sheet(program)
