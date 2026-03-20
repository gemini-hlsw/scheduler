# Events

The scheduler reacts to different events throughout the night allowing him to emulate
the behavior that could happen if the telescope is shutdown or a rapid ToO is activated.

The following are the types of events that are currently handle by the scheduler:

## Base event

:::scheduler.core.events.queue.Event

## Abstract events
These events are general purpose events that describe a non-specific behavior in the night.

:::scheduler.core.events.queue.RoutineEvent

:::scheduler.core.events.queue.InterruptionEvent

## Twilight events
They marked the start and end of the night. Without them the event cycle won't start and 
the final plan can't be created.

:::scheduler.core.events.queue.TwilightEvent

:::scheduler.core.events.queue.EveningTwilightEvent

:::scheduler.core.events.queue.MorningTwilightEvent

## Interruption and Resolution events
Are all the events that might interrupt the current plan and might trigger a different plan
with different conditions. Some interruptions create a lasting effect that might be resolved later
in the night.

:::scheduler.core.events.queue.WeatherChangeEvent

:::scheduler.core.events.queue.ToOActivationEvent

:::scheduler.core.events.queue.FaultEvent

:::scheduler.core.events.queue.WeatherClosureEvent

:::scheduler.core.events.queue.InterruptionResolutionEvent

:::scheduler.core.events.queue.FaultResolutionEvent

:::scheduler.core.events.queue.WeatherClosureResolutionEvent

# Event Queue
The `Event Queue` allows the scheduler to keep all events ordered chronologically.
This is separated in two: One that compromises all the nights and sites and one that
is specific for a night.

:::scheduler.core.events.queue.EventQueue

:::scheduler.core.events.queue.NightEventQueue

# Event Cycle
Encapsulates all the behavior the events can be affected through the night and pass
the event to be process by the `Change Monitor` and controls when the time accounting and
the plans creation happens.

:::scheduler.core.events.cycle.EventCycle


# Change Monitor
Matches the event with their specific behavior. These are usually modifications to both
`Collector` and `Selector`.

:::scheduler.core.components.changemonitor.ChangeMonitor

The `Change Monitor` gives updates to the `Event Cycle` through Time Coordinate Record so the 
cycle knows which updates are needed after that event.

:::scheduler.core.components.changemonitor.TimeCoordinateRecord
