# Resource Data

This directory contains static resource data files for OCS validation and GPP simulation modes. Common files are in the 
'common' directory.

Eventually, these should be removed from the Scheduler repository and made available elsewhere.

As of now, the files are as follows:

## Engineering Tasks

**Name:** `<Site>-engtasks.txt`

**Description:** These are tasks where scheduling cannot be done. They are known in advance to the Scheduler and will be
scheduled around. Note that `shutdown` is not included here: this is in the `telescope_schedules.xlsx` file (see below).


### Entry Format

* Night date: `YYYY-MM-DD`
* Start time: `HH:MM` (local) or `twi` for 12 degree evening twilight
* End time: `HH:MM` (local) or `twi` for 12 degree morning twilight
* Reason (optional): `[...]`

## Unexpected Loss

**Name:** `<Site>_unexpected_loss.txt`

**Description:** These are events that cannot be predicted by the Scheduler (examples: TMT protests, closing the
telescope due to inclement weather).

### Entry Format

* Night date: `YYYY-MM-DD`
* Start time: `HH:MM` (local) or `twi` for 12 degree evening twilight
* End time: `HH:MM` (local) or `twi` for 12 degree morning twilight
* Reason (optional): `[...]`

## Fault Reports

**Name:** `<site>-faults.txt`

**Description:** A list of fault reports that occurred at the site. Note that some of these happen during daytime events
and are thus not relevant for scheduling purpose.

* FR code: `FR-#####`
* Date (NOT night date): `YYYY-MM-DD`
* Start time: `HH:MM` (local?)
* Duration (in hours): `num hours`
* Reason: `[...]`
