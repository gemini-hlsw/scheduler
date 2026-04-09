# High-Level Architecture

Below shows a High-Level architecture for the Scheduler process.
This exact proces is done in both the GraphQL server


```mermaid
flowchart LR
  SP[Scheduler Parameters]
  subgraph Engine["Engine"]
    direction LR
    Builder[Builder]
    subgraph CorePipeline["Scheduler Core Pipeline"]
      direction TB
      Collector[Collector]
      Selector[Selector]
      Optimizer[Optimizer]
    end
    EventQueue[EventQueue]
    EventCycle[EventCycle]
  end
  Timelines[Timelines]
  RunSummary[Run Summary]

  SP -- new() --> Builder
  Builder --> CorePipeline
  CorePipeline --> EventCycle
  EventQueue --> EventCycle
  EventCycle -- run() --> Timelines
  EventCycle -- run() --> RunSummary
```

## Scheduler Parameters

::: scheduler.engine.SchedulerParameters

## Engine
::: scheduler.engine.Engine

