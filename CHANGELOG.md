# Changelog

All notable changes to this project will be documented in this file.
This changelog is managed by [Towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

## 2026.10.3 (2026-10-21)

### Features

- Updates to the weather panels to add new inputs and match the latest weather service schema.
- Add pagination to the Available Programs view in the UI, so the current 179 programs can be easily displayed independently of the screen size.

### Bug Fixes

- Currently the ODB subscription changes triggers in a time anchored depending on the site, that may lead to compute wrong plans for the other sites. All time will be used as UTC now instead and the UI will hide the simulated now option for now.
- Remove Sky Background from weather inputs, fix trailing 0 at beginning of the number inputs, temporarily disable water vapor input.
- Fix the Updates subscription to use new timing window fields
- Rearrange the operation panels, set the default wind speed to 5 m/s in the ui
- Fix Nighttracker dates and idle time on day times

## 2026.10.2 (2026-10-19)

No significant changes.

## 2026.10.1 (2026-10-19)

### Features

- Update change_monitor with changes to match_conditions
- The Ranker can now be chosen per run from the Validation UI, so the Default (multiplicative) and Additive (weighted sum) scoring algorithms can be compared side by side without a redeploy. The Ranker Tweaker shows which reading applies, because met/vis/wha are exponents under Default and linear weights under Additive. Selecting a Ranker per run is rejected outside VALIDATION; other modes continue to use config.ranker.name.
- Add a telemetry module that records operation durations and process memory as OpenTelemetry metrics, exported over OTLP to Grafana Cloud
- Start using ImageQuality as a continuous range instead of bins and keep it retro-compatible
- Parse elevation limit from program providers if possible, otherwise do it when the target information is retrieved for nonsidereal targets. Store the elevation limit inside the Constraints class
- Add target names in scheduler UI table
- Add memory limit cap on aggregator to stop errors in Heroku and add a script that allow parallel computation for Sight
- Modify the way the information is retrieved from the ODB and Sight, the observations are filtered down using the available resources first only for the requested to schedule nights, then filtered down again depending on their visibility from Sight and finally retrieved the information about their remaining visibility in a single batch.
- Allow disabling telemetry

### Bug Fixes

- Stamp weather change events with the simulated clock instead of the wall clock, so a plan requested during a simulated night starts inside that night
- Retrieve the observation data from the visits when calculating the timeline stats, collector may have None type observations
- Fix semester first and last date for non-sidereal target ephemeris files
- Change the condition if night times are set outside the twilight events, sent an error to alert the user and fail the run

  Re-derive the night to schedule on every build instead of once at start-up, and pick it from each site's real twilights rather than a fixed 19:00 UT threshold
- Add IGRINS-2 to the scheduler plot instruments list
- Carousel render only current item and neighbours to avoid unnecessary memory usage if multiple plans are requested.
- Update the default SchedulerParameters to use the whole semester as default for the remaining visibility calculation.
- Highchart run out of colors in the palette, added 7 extra colors. So far 13 out of 18 colors in the palette are being used.
- Fix linter issues
- Fix issues with ODB event handler in Night monitor:
  - The path for created was throwing a TypeError. That path is unlikely as most addition are handle by Defined -> Ready path
  - On deleted observations the reference label was incorrectly being matched with ObservationID.
  - ObservationStatus stored in the last plan was being modified by GM instead of compared with the real ODB retrieved data.
  - IDLE Timer had the wrong length for partial observations.
- Fix Visibility tab clashing with some unparsable programs and speed the responde by building a cache so each interaction does not trigger a live ODB query
- When Sight has no stored visibility for an observation and calculates it on the spot, use the observation's own constraints (timing windows, program dates, elevation and sky background) instead of empty defaults, so it no longer shows as visible outside its limits.

### Improvements

- For updated observations now the Scheduler check if conditions are matched with the current weather conditions and if any changes related to visibility happened requiring a new schedule.
  Also start the idle watch when a plan takes effect, not only after an ODB event, so a night that stays quiet still gets replanned; stand every timer down at end of night. Stop the Sight Stage-1 bulk read from stalling the event loop during a replan: the unpacking now runs off the loop and returns arrays instead of round-tripping them through Python lists. Read the ODB visibility change log against the wall clock rather than the simulated night, so a past simulation no longer treats historical edits as fresh changes and recomputes visibility for nothing.
- Deduplicate AdditiveRanker, which was a verbatim copy of DefaultRanker changing one expression. Both Rankers now share a single implementation in the Ranker base and differ only in how they combine the score terms, and there is one RankerParameters class instead of two identical ones (AdditiveRanker previously worked only by duck typing on the other class's parameters). Also fixes a latent crash where passing band_params explicitly left the attribute unset. Scoring output is unchanged, pinned by new bit-exact characterization tests.
- Add ephemerides for all validation semesters to ephemerides.tar, remove non-sidereal from skipped in the UI
- Speed up Sight reads for plans: fetch Stage 1 arrays and build target info only for the nights being planned (one in realtime, the full range in validation), look up Stage 1 targets in one query, and drop the unused `visibility_data.constraints` column (migration 012), which held most of the table's data.

## 2026.09.1 (2026-09-10)

### Features

- Add visibility coverage, aggregator ETA and visible-tonight endpoints
- Add endpoint to keep with updates from the ODB on targets/observations if recalculations are needed
- GNIRS/F2 support, schema change updates, improved check of used time
- Separate resource/env files for the REALTIME/OPS mode
- Add 26B GS resource files, fix GNIRS central_wavelength parsing
- Improve setting Band for GPP calibration
- Add NightlyTimelineStore accessable from task and NightMonitor
- ODB changes are now triggering new plans if we are following last generated plan and a new script was created to test and interact with the ODB for that
- VAL: new ephemeris files, visitor fixes; SIM: improved visitor/MOS support

### Bug Fixes

- Add fixes to the async process, specially on the process manager implementation and the task managment for async process in NightMonitor
- Fix placement of rToOs so they are scheduled as soon as possible
- Reorganize observation too_status section
- Modify the filter that was blocking Rapid ToOs when the program type was not matching the calendar program filter. Also a few GS ToOs were being activated iun the wrong date when past the noon for UTC.
- Fix drifting calculation in moond distance that was affecting the SB mask in Visibility calculations
- Fix F2 Wavelengths filters
- Fix observations interrupted by a weather closure or fault being marked as fully observed, which stopped their remaining atoms from ever being rescheduled, and report the completion of an interrupted visit as the atoms it actually observed rather than the atoms it was planned to reach
- Update UI and backend to fix the time tracking and add the correct info to the display
- Tie the available programs list to the build parameters night instead of today
- The visibility computation is failing due an array length mismatch, astropy is returning a misleading TypeError
- Alopeke instrument visits were black in the scheduler plot

### Improvements

- Update horizons client to retrieve full semester data for each target with a period of 4h and interpolate the retrieved data
- Update UI and server to display every sequence atom, the stitched plan in the UI, closure windows and the trigger event position througout the night
- Add Observation to Visit instead of a subset of fields of the same class
- Add visit steps start and end to the schedule observation table
- Use yyyy-mm-dd and 24 hours format to display the information in the operation tab
- Use yyyy-mm-dd and 24 hours format for dates in the result table
- Offload operations that are CPU and I/O bound to threads so they are not able to block the websockets connections to subscription in the same loop. It adds also some improvements in logging messaging regarding the subscriptions handle and connection
- Remove Collector ClassVar to attributes so the mutable state belongs to each process instead of a global state
- In realtime clear the timelines if the visibility range is modified. Remove get_final_plan unused functions
- Enable non-sidereal targets in the Sight aggregator

### Internal Changes

- GSCHED-1036

## 2026.07.2 (2026-07-06)

No significant changes.

## 2026.07.1 (2026-07-02)

### Features

- Add odb-obscalc subscription using gpp-client to decide if trigger new plan calculation
- Add Sight: new visibility service that uses a DB to accomplish faster retrieval for visibility
- Add steps in plan visits, fix GPP program parsing
- Add visibility aggregato background runner for Sight
- Add Sight support to subscriptions and fix configuration for sight
- Scheduler rToOs as soon as possible, GHOST support
- Added CI/CD pipeline with automated testing, CalVer versioning, and Towncrier changelog management.

### Bug Fixes

- Update uv.lock for lucupy 0.2.10
- Modify the deploy to dev and promotion to prod actions:
  - Add missing Heroku API keys
  - Modify Dockerfile path for building
- Fix time accounting record in Collector. making incomplete AND groups showing up in final plan
- Add missing frontend utils file
- Fix initial conditions for OCSEnvService that was setting previous day conditions
- Gppprogramprovider modified to use snake_case keys
- docker-compose update to fix backend healthcheck and add weather service
- Root group changed to folder/OR, GM pseudo time-accounting updates
- Rever ranker/default.py file, needed by Sight
- Igrins2 observations are parsed and some resources are set to None
- Add IGRINS-2 acquisition overhead
- Update versions to avoid critical vulnerabilities
- Modify visible ranges on Sight fetch method that was causing the airmass mask to be applied and causing low altitude observations
- Fix the release process: changelog is now built in a reviewable release PR (Prepare Release workflow) instead of pushing to the protected main branch, promotion pre-flight verifies the Heroku app setup, and the GitHub Release is created via the API
- Fix TooType comparisons related to None
- Fix visibility issues: SB background not being applied, wrong TimingWindow seletion and wrong cumulative visibility
- Program schema has changed, subtype is not part of the key type anymore, it was separated in multiple options, use gemini fragment

### Improvements

- Full setup time from query, remove hardcoding
- Add new groups in uv to handle different versions of gpp-client according to the new structure to separate DEV and Prod environment
- Igrins2 gpp name match to ocs, add instrument wavelength
- Improve the schedule queue event creation and handling
- Make the engineRT compute plans starting on current time for events instead of at night start
- Fix fill_sight and some redudant calls when storing data

### Internal Changes

- noissue
