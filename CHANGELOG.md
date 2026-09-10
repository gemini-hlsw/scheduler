# Changelog

All notable changes to this project will be documented in this file.
This changelog is managed by [Towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

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
