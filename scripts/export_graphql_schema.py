#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

SCHEMA_PATH = "backend.scheduler.graphql_mid.server"
SCHEMA_OUTPUT = Path("backend/scheduler.graphql")
DOCS_DIR = Path("docs/api")


def load_env(env_file: Path) -> None:
    """Load a .env file into os.environ (simple key=value, no interpolation)."""
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def export_schema() -> str | None:
    """Run strawberry export-schema and return the SDL string."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "backend")

    result = subprocess.run(
        ["uv", "run", "strawberry", "export-schema", SCHEMA_PATH],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"⚠ Could not export schema from {SCHEMA_PATH}")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        return None

    return result.stdout


def write_docs(sdl: str) -> None:
    """Write the SDL to docs directory and generate a markdown summary."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy SDL to docs
    docs_schema = DOCS_DIR / "schema.graphql"
    docs_schema.write_text(sdl)
    print(f"Schema copied to {docs_schema}")

    # Generate markdown summary
    summary_path = DOCS_DIR / "schema-summary.md"
    summary = "\n".join([
        "# GraphQL Schema Reference",
        "",
        "Auto-generated from the Strawberry schema. For interactive exploration,",
        "use the [GraphQL Playground](graphql.md).",
        "",
        "## SDL Schema",
        "",
        "```graphql",
        sdl.strip(),
        "```",
        "",
    ])
    summary_path.write_text(summary)
    print(f"Schema summary written to {summary_path}")


def main() -> None:
    # Load backend .env for local runs (CI sets env vars differently)
    load_env(Path("backend/.env"))

    print(f"Exporting schema from {SCHEMA_PATH}...")

    sdl = export_schema()
    if sdl is None:
        print("  Skipping schema export.")
        sys.exit(0)

    # Write main schema file
    SCHEMA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_OUTPUT.write_text(sdl)
    print(f"Schema exported to {SCHEMA_OUTPUT}")

    # Write docs
    write_docs(sdl)


if __name__ == "__main__":
    main()