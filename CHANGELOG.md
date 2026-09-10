# Changelog

All notable changes to this project will be documented in this file.
This changelog is managed by [Towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

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
