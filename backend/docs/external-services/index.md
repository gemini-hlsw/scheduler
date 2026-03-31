# External Services

The Scheduler also provides several external service that emulates
GPP services or functions so it can work with historical OCS data.

These services are for Scheduler use only and allow the system to run in Validation Mode.
They contain data from 2018A to 2019B.

The following list shows all the currently implemented services:

- [Resource](resource.md): Emulates and interacts with GPP Resource
- Environment: Emulates and interacts with GPP Environment
- Ephemeris: Allows the scheduler to handle Non-Sidereal Targets using
  the ephemerides files from Horizons.
- Horizons: Nasa Horizons client
- Proper Motion: Calculates the proper motion for Target coordinates
- Visibility: Allows the Scheduler to calculate the visibility values and fraction
  for an specific Target.
