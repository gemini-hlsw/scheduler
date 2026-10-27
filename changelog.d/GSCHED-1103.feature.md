The Simulation tab now has the build parameters panel and both run buttons, so plans can be computed there as in the Operation tab. The Operation tab only runs plans from now and no longer offers the UT visibility range, since operations always plan tonight.

On the Operation tab, the top bar shows, next to the connection status, the default night of the operation server for each site: its 12-degree evening and morning twilights in site local time. It is known before the first plan, custom night times in the build parameters don't change it, and it rolls over to the next night once tonight is over. It comes from the new `operationNight` query.

On the Operation tab, the build parameters take the night start and end as a time of day (HH:mm) only. The date follows from the default night shown in the top bar, so 22:00 falls on the evening and 03:00 on the next morning. Times outside the twilights are refused. The Simulation tab keeps its free date and time inputs.

On the Operation tab, the Visibility setting chooses how far the visibility calculation runs past tonight: to the end of the semester (the default), or a number of nights. It is sent as the new `visibilityNights` build parameter. The server counts it from the night it is planning, so it keeps rolling over each night, and it never runs past the semester end.
