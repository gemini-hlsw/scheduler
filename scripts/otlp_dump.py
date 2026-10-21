#!/usr/bin/env python3
"""A local OTLP/HTTP endpoint that dumps metrics to a file instead of Grafana Cloud.

Point anything that exports OTLP over HTTP at it (the stress scripts, a local backend) and
every data point lands as one JSON line, so nothing counts against the cloud quota.

Usage (from the repo root), in two terminals:

    # 1. The endpoint. Listens on 127.0.0.1:4318 (the OTLP/HTTP default) until Ctrl+C.
    .venv/bin/python scripts/otlp_dump.py serve

    # 2. Anything that exports metrics, pointed at it. Unset the Grafana headers: they carry
    #    the cloud credentials and the local endpoint has no use for them.
    env -u OTEL_EXPORTER_OTLP_HEADERS OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \\
        .venv/bin/python scripts/stress_night_monitor.py cases --visibility local --label nm-cases

    # 3. Read it back: totals per metric and operation, per run.
    .venv/bin/python scripts/otlp_dump.py summary

Output (default stress/results/otlp/metrics.jsonl) holds one line per data point:
{"received", "metric", "type", "unit", "attrs", "resource", "start", "time", value fields}.
Histograms carry count, sum, min, max, bounds and buckets. The scheduler exports cumulative
values every 60s (OTEL_METRIC_EXPORT_INTERVAL, in ms, changes that), so the last line of a
series is its total for that run. Traces and logs are accepted and counted, not stored.
"""

import argparse
import gzip
import json
import sys
import threading
import zlib
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "stress" / "results" / "otlp" / "metrics.jsonl"
# The resource attributes worth keeping on every line; the rest is the same on all of them.
RESOURCE_KEYS = (
    "service.name",
    "service.instance.id",
    "deployment.environment",
    "service.version",
)


def _iso(nanos: int) -> Optional[str]:
    if not nanos:
        return None
    return datetime.fromtimestamp(nanos / 1e9, UTC).isoformat(timespec="milliseconds")


def _any_value(value) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [_any_value(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return _attributes(value.kvlist_value.values)
    if kind == "bytes_value":
        return value.bytes_value.hex()
    return getattr(value, kind)


def _attributes(pairs) -> Dict[str, Any]:
    return {pair.key: _any_value(pair.value) for pair in pairs}


def _point_value(point) -> Dict[str, Any]:
    kind = point.WhichOneof("value")
    return {"value": getattr(point, kind) if kind else None}


def _histogram(point) -> Dict[str, Any]:
    return {
        "count": point.count,
        "sum": point.sum if point.HasField("sum") else None,
        "min": point.min if point.HasField("min") else None,
        "max": point.max if point.HasField("max") else None,
        "bounds": list(point.explicit_bounds),
        "buckets": list(point.bucket_counts),
    }


def flatten(request, received: str) -> Iterator[Dict[str, Any]]:
    """One row per data point of an ExportMetricsServiceRequest."""
    for resource_metrics in request.resource_metrics:
        resource = _attributes(resource_metrics.resource.attributes)
        resource = {k: resource[k] for k in RESOURCE_KEYS if k in resource}
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                kind = metric.WhichOneof("data")
                if kind is None:
                    continue
                data = getattr(metric, kind)
                for point in data.data_points:
                    row = {
                        "received": received,
                        "metric": metric.name,
                        "type": kind,
                        "unit": metric.unit,
                        "attrs": _attributes(point.attributes),
                        "resource": resource,
                        "start": _iso(point.start_time_unix_nano),
                        "time": _iso(point.time_unix_nano),
                    }
                    if kind == "histogram":
                        row.update(_histogram(point))
                    elif kind == "exponential_histogram":
                        row.update(
                            {
                                "count": point.count,
                                "sum": point.sum if point.HasField("sum") else None,
                                "min": point.min if point.HasField("min") else None,
                                "max": point.max if point.HasField("max") else None,
                            }
                        )
                    else:
                        row.update(_point_value(point))
                    yield row


# --------------------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------------------


class _Sink:
    """Appends rows to the output file; requests arrive on several threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a")
        self._lock = threading.Lock()
        self.counts: Counter = Counter()

    def write(self, rows: List[Dict[str, Any]]) -> None:
        with self._lock:
            for row in rows:
                self._file.write(json.dumps(row) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _decode_body(handler: BaseHTTPRequestHandler) -> bytes:
    body = handler.rfile.read(int(handler.headers.get("Content-Length") or 0))
    encoding = (handler.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        return zlib.decompress(body)
    return body


def _make_handler(sink: _Sink, verbose: bool):
    from google.protobuf import json_format
    from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    routes = {
        "/v1/metrics": (
            metrics_service_pb2.ExportMetricsServiceRequest,
            metrics_service_pb2.ExportMetricsServiceResponse,
        ),
        "/v1/traces": (
            trace_service_pb2.ExportTraceServiceRequest,
            trace_service_pb2.ExportTraceServiceResponse,
        ),
        "/v1/logs": (
            logs_service_pb2.ExportLogsServiceRequest,
            logs_service_pb2.ExportLogsServiceResponse,
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
            route = routes.get(self.path.split("?")[0])
            if route is None:
                self.send_error(
                    404, "OTLP/HTTP only serves /v1/metrics, /v1/traces and /v1/logs"
                )
                return
            request_cls, response_cls = route
            try:
                body = _decode_body(self)
                request = request_cls()
                if "json" in (self.headers.get("Content-Type") or ""):
                    json_format.Parse(body, request)
                else:
                    request.ParseFromString(body)
            except Exception as exc:
                self.send_error(
                    400, f"Could not decode the request: {type(exc).__name__}"
                )
                return

            received = datetime.now(UTC).isoformat(timespec="milliseconds")
            if self.path.startswith("/v1/metrics"):
                rows = list(flatten(request, received))
                sink.write(rows)
                sink.counts["points"] += len(rows)
                if verbose:
                    print(
                        f"{received}  {len(rows)} data points "
                        f"({', '.join(f'{k} {v}' for k, v in Counter(r['metric'] for r in rows).items())})",
                        flush=True,
                    )
            sink.counts[self.path] += 1

            payload = response_cls().SerializeToString()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            # The default logs every request line to stderr; the verbose summary above is enough.
            pass

    return Handler


def serve(args) -> int:
    sink = _Sink(Path(args.out))
    server = ThreadingHTTPServer(
        (args.host, args.port), _make_handler(sink, not args.quiet)
    )
    print(
        f"OTLP/HTTP endpoint on http://{args.host}:{args.port}, writing {_shown(sink.path)}"
    )
    print(
        f"Point exporters at it with OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:{args.port}. "
        f"Ctrl+C to stop.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sink.close()
    print(
        f"\nStopped. {sink.counts['/v1/metrics']} metric exports, {sink.counts['points']} data points"
        + (
            f", {sink.counts['/v1/traces']} trace and {sink.counts['/v1/logs']} log exports (not stored)"
            if sink.counts["/v1/traces"] or sink.counts["/v1/logs"]
            else ""
        )
        + f" in {_shown(sink.path)}."
    )
    return 0


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------


def _rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _fmt_s(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds >= 100:
        return f"{seconds:.0f}s"
    if seconds >= 1:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"


def summary(args) -> int:
    path = Path(args.out)
    if not path.exists():
        sys.exit(
            f"Nothing at {_shown(path)} yet: run `serve` and point an exporter at it."
        )

    # Cumulative series: the last row per (run, metric, attributes) is that run's total.
    last: Dict[Tuple, Dict[str, Any]] = {}
    for row in _rows(path):
        if args.metric and args.metric not in row["metric"]:
            continue
        env = row.get("resource", {}).get("deployment.environment")
        if args.environment and env != args.environment:
            continue
        key = (
            row.get("start"),
            row.get("resource", {}).get("service.instance.id"),
            env,
            row["metric"],
            json.dumps(row.get("attrs", {}), sort_keys=True),
        )
        last[key] = row

    runs: Dict[Tuple, List[Dict[str, Any]]] = {}
    for (start, instance, env, *_rest), row in sorted(
        last.items(), key=lambda item: str(item[0])
    ):
        runs.setdefault((start, instance, env), []).append(row)

    for (start, instance, env), rows in runs.items():
        print(
            f"\nRun started {start} ({instance}{', ' + env if env else ''}), last export {max(r['time'] or '' for r in rows)}"
        )
        print(
            f"  {'metric':40} {'attributes':42} {'count':>7} {'mean':>8} {'max':>8} {'total':>9}"
        )
        for row in sorted(
            rows,
            key=lambda r: (r["metric"], json.dumps(r.get("attrs", {}), sort_keys=True)),
        ):
            attrs = ", ".join(
                f"{k}={v}" for k, v in sorted(row.get("attrs", {}).items())
            )
            if row["type"] in ("histogram", "exponential_histogram"):
                count, total = row.get("count") or 0, row.get("sum")
                mean = total / count if count and total is not None else None
                seconds = row.get("unit") == "s"
                show = (
                    _fmt_s if seconds else (lambda v: "-" if v is None else f"{v:.3g}")
                )
                print(
                    f"  {row['metric'][:40]:40} {attrs[:42]:42} {count:>7} {show(mean):>8} "
                    f"{show(row.get('max')):>8} {show(total):>9}"
                )
            else:
                value = row.get("value")
                print(
                    f"  {row['metric'][:40]:40} {attrs[:42]:42} {'':>7} {'':>8} {'':>8} {value!s:>9}"
                )
    if not runs:
        print("No matching data points.")
    return 0


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "serve", help="Listen for OTLP/HTTP exports and append them to the file."
    )
    run.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind (default 127.0.0.1, this machine only).",
    )
    run.add_argument(
        "--port",
        type=int,
        default=4318,
        help="Port (default 4318, the OTLP/HTTP default).",
    )
    run.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"JSONL file to append to (default {_shown(DEFAULT_OUT)}).",
    )
    run.add_argument(
        "--quiet", action="store_true", help="Do not print a line per export."
    )

    show = sub.add_parser("summary", help="Totals per metric and attributes, per run.")
    show.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"JSONL file to read (default {_shown(DEFAULT_OUT)}).",
    )
    show.add_argument(
        "--metric",
        help="Only metrics whose name contains this, e.g. operation.duration.",
    )
    show.add_argument(
        "--environment", help="Only this deployment.environment, e.g. stress."
    )

    args = parser.parse_args()
    return serve(args) if args.command == "serve" else summary(args)


if __name__ == "__main__":
    sys.exit(main())
