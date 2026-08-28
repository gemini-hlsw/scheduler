#!/usr/bin/env python
"""
Probe the ``obscalcUpdate`` subscription with three different input filters at once.

    as-implemented   input: {executableOnly: true, newCalculationState: {EQ: READY}}
    no-calc-state    input: {executableOnly: true}
    no-input         obscalcUpdate with no input at all

The first is exactly what ``ODBEventSource.OBSERVATION_EDIT`` subscribes to via
``GPPClient.scheduler.subscribe_to_calculation_updates``. The other two widen the
filter one step at a time, so an event that shows up only under ``no-input`` tells
you the server-side filter is what is dropping it, not the transport.

All three run concurrently over their own websocket against the same ODB, and every
raw frame is printed as it arrives: ``connection_ack``, ``next``, ``error`` and
``complete``. Payloads are printed unvalidated (raw JSON), on purpose: pydantic
validation of the generated models can blow up on enum drift and would hide the very
response you are trying to look at.

Note that ``executableOnly`` defaults to ``false`` server-side, so ``no-input`` is
the broadest of the three, not merely "the same without the state filter".

Examples
--------
    python obscalc_subscription_probe.py
    python obscalc_subscription_probe.py --duration 120
    python obscalc_subscription_probe.py --only no-input --compact

Requires ``GPP_TOKEN`` (or ``GPP_DEVELOPMENT_TOKEN`` on a development install) in the
environment or in a local ``.env``, same as the ``gpp`` CLI.
"""

__all__ = ["app"]

import asyncio
import json
from datetime import UTC, datetime
from typing import Annotated, Any, Optional
from uuid import uuid4

import typer
from gpp_client.client import GPPClient
from websockets import connect as ws_connect
from websockets.typing import Subprotocol

GRAPHQL_TRANSPORT_WS = "graphql-transport-ws"

# Copied verbatim from the generated ``SchedulerObservationsUpdates`` operation so the
# three variants differ only in the input filter, never in the shape of the response.
_SELECTION = """
  oldCalculationState
  newCalculationState
  editType
  value {
    id
    reference { label }
    observationTime
    program { active { end start } }
    workflow { value { state } }
    execution {
      visits {
        matches {
          observation { id }
          atomRecords { matches { executionState id } }
        }
      }
    }
    targetEnvironment {
      asterism {
        name
        sidereal { ra { hours hms degrees } dec { degrees dms } epoch }
        nonsidereal { des keyType key }
      }
      explicitBase { ra { hms } dec { dms } }
    }
    constraintSet {
      imageQuality
      cloudExtinction
      skyBackground
      waterVapor
      elevationRange {
        airMass { min max }
        hourAngle { minHours maxHours }
      }
    }
    timingWindows {
      inclusion
      startUtc
      end {
        __typename
        ... on TimingWindowEndAt { atUtc }
        ... on TimingWindowEndAfter {
          after { seconds }
          repeat { period { seconds } times }
        }
      }
    }
    instrument
  }
"""

# name -> the argument list spliced into ``obscalcUpdate(...)``. Empty means no args.
_VARIANTS: dict[str, str] = {
    "as-implemented": "(input: {executableOnly: true, newCalculationState: {EQ: READY}})",
    "no-calc-state": "(input: {executableOnly: true})",
    "no-input": "",
}


def _query(variant: str) -> str:
    operation = "Probe" + variant.title().replace("-", "")
    return (
        f"subscription {operation} {{\n"
        f"  obscalcUpdate{_VARIANTS[variant]} {{{_SELECTION}}}\n"
        f"}}"
    )


def _stamp() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]


def _emit(variant: str, kind: str, detail: str = "") -> None:
    # Flushed per line: three subscriptions interleave, and a buffered stdout would
    # reorder them relative to what the ODB actually sent.
    print(f"[{_stamp()}] {variant:<15} {kind}{(' ' + detail) if detail else ''}", flush=True)


def _summarize(payload: dict[str, Any]) -> str:
    update = (payload.get("data") or {}).get("obscalcUpdate") or {}
    value = update.get("value") or {}
    return (
        f"{update.get('editType')} "
        f"{(update.get('oldCalculationState'))}->{update.get('newCalculationState')} "
        f"{(value.get('reference') or {}).get('label')} "
        f"({value.get('id')}) "
        f"workflow={((value.get('workflow') or {}).get('value') or {}).get('state')}"
    )


async def _run_variant(
    variant: str,
    ws_url: str,
    headers: dict[str, str],
    init_payload: dict[str, Any],
    compact: bool,
) -> None:
    query = _query(variant)
    operation_id = str(uuid4())
    _emit(variant, "connecting", ws_url)

    async with ws_connect(
        ws_url,
        subprotocols=[Subprotocol(GRAPHQL_TRANSPORT_WS)],
        additional_headers=headers,
    ) as websocket:
        await websocket.send(
            json.dumps({"type": "connection_init", "payload": init_payload})
        )

        async for raw in websocket:
            message = json.loads(raw)
            kind = message.get("type")

            if kind == "connection_ack":
                _emit(variant, "connection_ack")
                await websocket.send(
                    json.dumps(
                        {
                            "id": operation_id,
                            "type": "subscribe",
                            "payload": {"query": query, "operationName": None},
                        }
                    )
                )
                _emit(variant, "subscribed", "waiting for events...")
            elif kind == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            elif kind == "next":
                payload = message.get("payload") or {}
                _emit(variant, "next", _summarize(payload))
                if not compact:
                    print(json.dumps(payload, indent=2), flush=True)
            elif kind == "error":
                # A malformed or rejected subscription lands here rather than as a
                # transport failure, and is the most likely reason for a silent sub.
                _emit(variant, "ERROR", json.dumps(message.get("payload")))
            elif kind == "complete":
                _emit(variant, "complete", "server closed this subscription")
                return
            else:
                _emit(variant, f"({kind})", json.dumps(message)[:200])


app = typer.Typer(add_completion=False)


@app.command()
def main(
    only: Annotated[
        Optional[str],
        typer.Option(help=f"Run a single variant: {', '.join(_VARIANTS)}."),
    ] = None,
    duration: Annotated[
        Optional[float],
        typer.Option(help="Stop after this many seconds. Default: run until Ctrl-C."),
    ] = None,
    compact: Annotated[
        bool, typer.Option(help="One summary line per event instead of full JSON.")
    ] = False,
    show_queries: Annotated[
        bool, typer.Option(help="Print each subscription document before connecting.")
    ] = False,
) -> None:
    """Run the obscalcUpdate subscription variants side by side."""
    variants = list(_VARIANTS)
    if only is not None:
        if only not in _VARIANTS:
            raise typer.BadParameter(f"Unknown variant {only!r}. Pick one of {variants}.")
        variants = [only]

    if show_queries:
        for variant in variants:
            print(f"--- {variant} ---\n{_query(variant)}\n", flush=True)

    asyncio.run(_probe(variants, duration, compact))


async def _probe(variants: list[str], duration: Optional[float], compact: bool) -> None:
    client = GPPClient()
    graphql = client.graphql
    ws_url = graphql.ws_url
    headers = dict(graphql.ws_headers or {})
    init_payload = dict(graphql.ws_connection_init_payload or {})

    tasks = [
        asyncio.create_task(
            _run_variant(variant, ws_url, headers, init_payload, compact),
            name=variant,
        )
        for variant in variants
    ]
    gathered = asyncio.gather(*tasks, return_exceptions=True)

    try:
        if duration is None:
            results = await gathered
        else:
            try:
                results = await asyncio.wait_for(asyncio.shield(gathered), duration)
            except asyncio.TimeoutError:
                print(f"\n[{_stamp()}] duration reached, stopping.", flush=True)
                results = []
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n[{_stamp()}] interrupted, stopping.", flush=True)
        results = []
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

    for variant, result in zip(variants, results):
        if isinstance(result, BaseException) and not isinstance(
            result, asyncio.CancelledError
        ):
            _emit(variant, "FAILED", f"{type(result).__name__}: {result}")


if __name__ == "__main__":
    app()
