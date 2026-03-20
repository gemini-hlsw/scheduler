# Weather

Simple apollo server project to handle manual weather updates.

The service have:

1. `weather`, a query which uses the site input `GN` or `GS` and returns the latest weather value for that site.
2. `updateWeather`, a mutation which receives the site and the new desired weather state, this is `imageQuality`, `cloudCover`, `windDirection` and `windSpeed`.
3. `weatherUpdates`, a subscription channel which receives the site and will return the lates update value every time it is manually changed.
