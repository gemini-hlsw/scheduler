#!/usr/bin/env python
"""
Drive an observation through its execution lifecycle from the command line.

    add       DEFINED -> READY      setObservationWorkflowState
    remove    READY   -> DEFINED    setObservationWorkflowState
    execute   READY   -> ONGOING    recordVisit + slew/sequence/step events
    finish    ONGOING -> COMPLETED  setObservationWorkflowState

``execute`` is the odd one out. ONGOING is not a settable workflow state for a
normal observation: the ODB derives it from execution, so ``execute`` starts
real execution and the workflow follows on its own.

The ``addSequenceEvent`` START is load-bearing, not decoration. A visit plus
both slew stages plus a step START_STEP still leaves the ODB reporting
``NOT_STARTED``, and the workflow then reverts from ONGOING back to READY.
Only once the sequence START is recorded does ``executionState`` move to
ONGOING and stay there. Do not drop it from the chain.

The observation may be given either as an ID (``o-1a2b``) or as a reference
label (``G-2025A-0001-Q-0001``); labels are resolved to an ID first because
these mutations only accept IDs.

Examples
--------
    python observation_lifecycle.py add o-1a2b
    python observation_lifecycle.py remove o-1a2b
    python observation_lifecycle.py execute G-2025A-0001-Q-0001
    python observation_lifecycle.py finish o-1a2b --retry

Requires ``GPP_TOKEN`` (or ``GPP_DEVELOPMENT_TOKEN`` on a development install)
in the environment or in a local ``.env``, same as the ``gpp`` CLI, plus the
``gpp-dev`` or ``gpp-prod`` dependency group.
"""

__all__ = ["app"]

import asyncio
import re
from collections.abc import Coroutine
from typing import Annotated, Any

import typer
from gpp_client.cli import output
from gpp_client.client import GPPClient
from gpp_client.exceptions import GPPError
from gpp_client.generated import GraphQLClientError, ObservationWorkflowState

# ObservationId is `o-[1-9a-f][0-9a-f]*` per the schema; anything else is
# treated as a reference label.
_OBSERVATION_ID_RE = re.compile(r"^o-[1-9a-f][0-9a-f]*$")

# States the ODB derives from execution rather than storing as a user state.
_EXECUTION_STATES = ("ONGOING", "COMPLETED")

# How long --retry waits for the observation's background calculation to settle.
_CALC_RETRY_ATTEMPTS = 10
_CALC_RETRY_DELAY = 1.0

# Sequence preference order. Acquisition runs first where an instrument has
# one; GHOST and IGRINS-2 have no acquisition sequence in the schema, so they
# fall through to the first science step.
_SEQUENCE_KINDS = ("acquisition", "science")

# gpp-client has no generated methods for recordVisit/addSlewEvent/addStepEvent
# yet, so these go over the raw GraphQL transport. Replace them with domain
# calls once the operations ship in a gpp-client release.
_NEXT_STEP_QUERY = """
query SchedulerNextStep($observationId: ObservationId!) {
  executionConfig(observationId: $observationId, futureLimit: 0) {
    instrument
    flamingos2 {
      acquisition { nextAtom { id steps { id } } }
      science { nextAtom { id steps { id } } }
    }
    ghost {
      science { nextAtom { id steps { id } } }
    }
    gmosNorth {
      acquisition { nextAtom { id steps { id } } }
      science { nextAtom { id steps { id } } }
    }
    gmosSouth {
      acquisition { nextAtom { id steps { id } } }
      science { nextAtom { id steps { id } } }
    }
    gnirs {
      acquisition { nextAtom { id steps { id } } }
      science { nextAtom { id steps { id } } }
    }
    igrins2 {
      science { nextAtom { id steps { id } } }
    }
  }
}
"""

_RECORD_VISIT_MUTATION = """
mutation SchedulerRecordVisit($observationId: ObservationId!) {
  recordVisit(input: { observationId: $observationId }) {
    visit { id site }
  }
}
"""

_ADD_SLEW_EVENT_MUTATION = """
mutation SchedulerAddSlewEvent(
  $observationId: ObservationId!
  $slewStage: SlewStage!
) {
  addSlewEvent(
    input: { observationId: $observationId, slewStage: $slewStage }
  ) {
    event { id slewStage visit { id } }
  }
}
"""

_ADD_SEQUENCE_EVENT_MUTATION = """
mutation SchedulerAddSequenceEvent(
  $visitId: VisitId!
  $command: SequenceCommand!
) {
  addSequenceEvent(input: { visitId: $visitId, command: $command }) {
    event { id command visit { id } }
  }
}
"""

_ADD_STEP_EVENT_MUTATION = """
mutation SchedulerAddStepEvent(
  $visitId: VisitId!
  $stepId: StepId!
  $stepStage: StepStage!
) {
  addStepEvent(
    input: { visitId: $visitId, stepId: $stepId, stepStage: $stepStage }
  ) {
    event { id stepStage step { id } }
  }
}
"""

# The workflow read goes over the raw transport too, deliberately. The ODB has
# already added `GENERIC_WARNING` to ObservationValidationCode, which is in
# neither the checked-in schema nor gpp-client's generated enum, so
# `client.workflow_state` blows up with a pydantic error on any observation
# carrying one. Selecting `messages` without `code` sidesteps the enum, so new
# validation codes can't break this script.
_WORKFLOW_FIELDS = """
    workflow {
      calculationState
      value {
        state
        validTransitions
        validationErrors { messages }
      }
    }
    execution { executionState }
"""

_WORKFLOW_BY_ID_QUERY = (
    """
query SchedulerWorkflowById($observationId: ObservationId!) {
  observation(observationId: $observationId) {
    id
"""
    + _WORKFLOW_FIELDS
    + """
  }
}
"""
)

_WORKFLOW_BY_REFERENCE_QUERY = (
    """
query SchedulerWorkflowByReference($observationReference: ObservationReferenceLabel!) {
  observation(observationReference: $observationReference) {
    id
"""
    + _WORKFLOW_FIELDS
    + """
  }
}
"""
)

_SET_WORKFLOW_STATE_MUTATION = """
mutation SchedulerSetWorkflowState(
  $observationId: ObservationId!
  $state: ObservationWorkflowState!
) {
  setObservationWorkflowState(
    input: { observationId: $observationId, state: $state }
  ) {
    state
    validTransitions
    validationErrors { messages }
  }
}
"""

app = typer.Typer(
    name="observation-lifecycle",
    help="Move an observation between DEFINED, READY, ONGOING and COMPLETED.",
    no_args_is_help=True,
    add_completion=False,
)

ObservationArg = Annotated[
    str,
    typer.Argument(
        metavar="OBSERVATION",
        help="Observation ID (o-1a2b) or reference label (G-2025A-0001-Q-0001).",
    ),
]
RetryOpt = Annotated[
    bool,
    typer.Option(
        "--retry",
        help=(
            "Retry while the observation's background calculation is not READY, "
            "instead of failing immediately."
        ),
    ),
]


async def _gql(
    client: GPPClient,
    document: str,
    variables: dict[str, Any],
) -> dict[str, Any]:
    """
    Run a raw GraphQL document through the client's transport.

    Parameters
    ----------
    client : GPPClient
        Connected GPP client.
    document : str
        GraphQL document to execute.
    variables : dict[str, Any]
        Variables for the document.

    Returns
    -------
    dict[str, Any]
        The ``data`` payload of the response.
    """
    response = await client._graphql.execute(query=document, variables=variables)
    return client._graphql.get_data(response)


async def _fetch_observation(
    client: GPPClient, observation: str
) -> tuple[str, dict[str, Any]]:
    """
    Return the observation ID and payload for an ID or a reference label.

    Parameters
    ----------
    client : GPPClient
        Connected GPP client.
    observation : str
        Observation ID or reference label.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The observation ID and the observation payload, which carries both
        ``workflow`` and ``execution``.

    Raises
    ------
    typer.BadParameter
        If nothing matches, or the observation has no workflow.
    """
    if _OBSERVATION_ID_RE.match(observation):
        data = await _gql(client, _WORKFLOW_BY_ID_QUERY, {"observationId": observation})
    else:
        data = await _gql(
            client,
            _WORKFLOW_BY_REFERENCE_QUERY,
            {"observationReference": observation},
        )

    obs = data.get("observation")
    if obs is None:
        raise typer.BadParameter(f"No observation found for '{observation}'.")
    if obs["id"] != observation:
        output.dim_info(f"Resolved {observation} to {obs['id']}")

    if obs.get("workflow") is None:
        raise typer.BadParameter(f"{obs['id']} has no workflow.")
    return obs["id"], obs


def _report_warnings(validation_errors: list[dict[str, Any]]) -> None:
    """
    Print any validation messages attached to a workflow.

    Parameters
    ----------
    validation_errors : list[dict[str, Any]]
        The ``validationErrors`` payload.
    """
    for error in validation_errors:
        for message in error["messages"]:
            output.warning(message)


async def _next_step_id(client: GPPClient, observation_id: str) -> str:
    """
    Return the ID of the next step to execute for an observation.

    Works for every instrument in the schema. Acquisition is preferred where
    the instrument has one, matching the order the sequence is actually
    executed in; GHOST and IGRINS-2 fall through to the first science step.

    Parameters
    ----------
    client : GPPClient
        Connected GPP client.
    observation_id : str
        The observation ID.

    Returns
    -------
    str
        The step ID.

    Raises
    ------
    typer.BadParameter
        If no sequence can be generated, or it contains no steps.
    """
    data = await _gql(client, _NEXT_STEP_QUERY, {"observationId": observation_id})
    config = data.get("executionConfig")
    if config is None:
        raise typer.BadParameter(
            f"No sequence could be generated for {observation_id}; it cannot be "
            "executed yet."
        )

    # Exactly one instrument field is non-null, so whichever one is populated
    # is the mode in use. Reading them off the response rather than a hardcoded
    # list means only the query above needs touching for a new instrument.
    for key, instrument_config in config.items():
        if key == "instrument" or not isinstance(instrument_config, dict):
            continue
        for kind in _SEQUENCE_KINDS:
            sequence = instrument_config.get(kind)
            if sequence is None:
                continue
            steps = sequence["nextAtom"]["steps"]
            if steps:
                output.dim_info(f"Next {kind} step is {steps[0]['id']}")
                return steps[0]["id"]

    raise typer.BadParameter(
        f"The sequence for {observation_id} ({config.get('instrument')}) has no "
        "steps left to execute."
    )


async def _transition(
    observation: str,
    target_state: ObservationWorkflowState,
    retry: bool,
) -> None:
    """
    Set an observation's workflow state.

    The target is checked against the observation's ``validTransitions`` first,
    so an out-of-order call fails here rather than being rejected by the ODB.

    Parameters
    ----------
    observation : str
        Observation ID or reference label.
    target_state : ObservationWorkflowState
        State to transition into.
    retry : bool
        Whether to wait out a not-yet-READY background calculation.

    Raises
    ------
    typer.BadParameter
        If the calculation never settles, or the transition is not allowed.
    """
    async with GPPClient() as client:
        observation_id, obs = await _fetch_observation(client, observation)
        workflow = obs["workflow"]

        # validTransitions is only trustworthy once the background calculation
        # has settled, so wait for it before reading anything off the workflow.
        attempts = _CALC_RETRY_ATTEMPTS if retry else 1
        for attempt in range(1, attempts + 1):
            if workflow["calculationState"] == "READY":
                break
            if attempt == attempts:
                raise typer.BadParameter(
                    f"{observation_id} is still calculating "
                    f"({workflow['calculationState']}). Retry with --retry."
                )
            await asyncio.sleep(_CALC_RETRY_DELAY)
            _, obs = await _fetch_observation(client, observation_id)
            workflow = obs["workflow"]

        current = workflow["value"]["state"]
        _report_warnings(workflow["value"]["validationErrors"])

        if current == target_state.value:
            output.success(f"{observation_id} is already {current}.")
            return

        valid = workflow["value"]["validTransitions"]
        if target_state.value not in valid:
            raise typer.BadParameter(
                f"Cannot move {observation_id} from {current} to "
                f"{target_state.value}. Valid transitions are: "
                f"{', '.join(valid) or 'none'}."
            )

        # Printed because the ODB's read and write paths can disagree: the
        # query reports ONGOING and offers COMPLETED, while the mutation
        # rejects it claiming the current state is READY. Showing what we read
        # makes that contradiction visible in the failure itself.
        output.dim_info(
            f"{observation_id} reads {current}, offering {', '.join(valid)}"
        )

        # ObservationWorkflowService.getWorkflowsModesAndRoles keeps execution
        # state only for observations whose validation map is *empty*, so a
        # mere warning makes the mutation recompute the state as READY and
        # reject an execution transition the cached workflow just advertised.
        # Clearing the warning is the only workaround from this side.
        if workflow["value"]["validationErrors"] and current in _EXECUTION_STATES:
            output.warning(
                "This observation carries a validation warning, which makes the "
                "ODB recompute its state as READY for the mutation. Expect a "
                "rejection until the warning is cleared."
            )
        output.procedure(f"Setting {observation_id} to {target_state.value}...")
        data = await _gql(
            client,
            _SET_WORKFLOW_STATE_MUTATION,
            {"observationId": observation_id, "state": target_state.value},
        )

    payload = data.get("setObservationWorkflowState")
    if payload is None:
        output.warning(f"The ODB returned no workflow for {observation_id}.")
        return

    output.success(f"{observation_id} is now {payload['state']}.")


async def _start_execution(observation: str) -> None:
    """
    Start executing an observation, which is what drives it to ONGOING.

    Opens a visit, slews to the target and starts the next step. The workflow
    state is re-read afterwards, but it is a background calculation, so it may
    still lag by the time this returns.

    Parameters
    ----------
    observation : str
        Observation ID or reference label.
    """
    async with GPPClient() as client:
        observation_id, _ = await _fetch_observation(client, observation)
        step_id = await _next_step_id(client, observation_id)

        output.procedure(f"Recording a visit for {observation_id}...")
        visit = await _gql(
            client, _RECORD_VISIT_MUTATION, {"observationId": observation_id}
        )
        visit_id = visit["recordVisit"]["visit"]["id"]
        output.dim_info(f"Visit {visit_id}")

        # Both stages are sent: a slew that never ends would leave the
        # telescope logically mid-slew while the step is already running.
        for slew_stage in ("START_SLEW", "END_SLEW"):
            output.procedure(f"Slew {slew_stage}...")
            await _gql(
                client,
                _ADD_SLEW_EVENT_MUTATION,
                {"observationId": observation_id, "slewStage": slew_stage},
            )

        # A visit plus slew events is not enough on its own: the ODB still
        # reported NOT_STARTED for one, and the workflow reverted from ONGOING
        # back to READY. START tells the ODB the sequence itself began.
        output.procedure("Starting the sequence...")
        await _gql(
            client,
            _ADD_SEQUENCE_EVENT_MUTATION,
            {"visitId": visit_id, "command": "START"},
        )

        output.procedure(f"Starting step {step_id}...")
        await _gql(
            client,
            _ADD_STEP_EVENT_MUTATION,
            {"visitId": visit_id, "stepId": step_id, "stepStage": "START_STEP"},
        )

        _, obs = await _fetch_observation(client, observation_id)

    execution_state = obs["execution"]["executionState"]
    workflow_state = obs["workflow"]["value"]["state"]
    output.success(
        f"{observation_id}: executionState {execution_state}, workflow "
        f"{workflow_state}."
    )
    # executionState is the honest signal. The workflow value is a cached
    # calculation that can read ONGOING briefly and then revert to READY, so a
    # NOT_STARTED here means nothing durable was recorded.
    if execution_state == "NOT_STARTED":
        output.warning(
            "The ODB still reports NOT_STARTED, so this observation will fall "
            "back to READY. Execution did not take."
        )


def _run(coro: Coroutine[Any, Any, None]) -> None:
    """
    Run a coroutine, turning client errors into a clean non-zero exit.

    Parameters
    ----------
    coro : Coroutine[Any, Any, None]
        The coroutine to run.
    """
    try:
        asyncio.run(coro)
    except (GPPError, GraphQLClientError) as exc:
        output.fail(str(exc))
        raise typer.Exit(code=1)


@app.command("add")
def add(observation: ObservationArg, retry: RetryOpt = False) -> None:
    """
    Mark the observation READY for execution (DEFINED -> READY).
    """
    _run(_transition(observation, ObservationWorkflowState.READY, retry))


@app.command("remove")
def remove(observation: ObservationArg, retry: RetryOpt = False) -> None:
    """
    Withdraw the observation from the ready pool (READY -> DEFINED).
    """
    _run(_transition(observation, ObservationWorkflowState.DEFINED, retry))


@app.command("execute")
def execute(observation: ObservationArg) -> None:
    """
    Start executing the observation, which drives it to ONGOING.
    """
    _run(_start_execution(observation))


@app.command("finish")
def finish(observation: ObservationArg, retry: RetryOpt = False) -> None:
    """
    Mark the observation COMPLETED (ONGOING -> COMPLETED).
    """
    _run(_transition(observation, ObservationWorkflowState.COMPLETED, retry))


if __name__ == "__main__":
    app()
