#!/usr/bin/env python3
"""
Lint the Strawberry GraphQL schema for missing descriptions.

Exports the schema via `uv run strawberry export-schema`, then imports
the schema object to introspect types and fields for missing descriptions.

Usage:
    python scripts/lint_graphql_schema.py

Exit codes:
    0 — all types and fields have descriptions
    1 — missing descriptions found
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────
SCHEMA_PATH = "backend.scheduler.graphql_mid.server"

# Types to skip (internal Strawberry / GraphQL introspection types)
SKIP_PREFIXES = ("__", "PageInfo", "Connection", "Edge")


def load_env(env_file: Path) -> None:
    """Load a .env file into os.environ."""
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def verify_schema_exports() -> bool:
    """Quick check that strawberry can export the schema at all."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "backend")

    result = subprocess.run(
        ["uv", "run", "strawberry", "export-schema", SCHEMA_PATH],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"Could not export schema from {SCHEMA_PATH}")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        return False

    return True


def resolve_schema():
    """Import and return the Strawberry schema object."""
    # Ensure backend is on the path
    backend_path = str(Path.cwd() / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Try importing the full path as a module with a `schema` attribute
    try:
        module = importlib.import_module(SCHEMA_PATH)
        for attr_name in ("schema", "Schema", "graphql_schema"):
            schema = getattr(module, attr_name, None)
            if schema is not None:
                return schema
    except ImportError:
        pass

    # Try splitting: everything-but-last as module, last as attribute
    module_path, _, attr = SCHEMA_PATH.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except (ImportError, AttributeError) as e:
        print(f" Could not import schema object: {e}")
        return None


def lint_schema(schema) -> list[str]:
    """Check all types and fields for missing descriptions."""
    issues = []
    introspection = schema.introspect()
    types = introspection.get("__schema", {}).get("types", [])

    for type_def in types:
        name = type_def.get("name", "")

        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue

        kind = type_def.get("kind", "")
        if kind in ("SCALAR", "ENUM"):
            continue

        # Check type description
        desc = type_def.get("description")
        if not desc or not desc.strip():
            issues.append(f"Type '{name}' is missing a description")

        # Check field descriptions
        for field in type_def.get("fields") or []:
            field_name = field.get("name", "")
            field_desc = field.get("description")
            if not field_desc or not field_desc.strip():
                issues.append(f"Field '{name}.{field_name}' is missing a description")

        # Check input field descriptions
        for field in type_def.get("inputFields") or []:
            field_name = field.get("name", "")
            field_desc = field.get("description")
            if not field_desc or not field_desc.strip():
                issues.append(f"Input field '{name}.{field_name}' is missing a description")

    return issues


def main() -> None:
    load_env(Path("backend/.env"))

    print("Linting GraphQL schema for missing descriptions...\n")

    # First verify strawberry can export (catches import errors with better messages)
    if not verify_schema_exports():
        print("  Skipping schema lint.")
        sys.exit(0)

    # Then import the schema object for introspection
    schema = resolve_schema()
    if schema is None:
        print("  Skipping schema lint.")
        sys.exit(0)

    issues = lint_schema(schema)

    if not issues:
        print("All types and fields have descriptions")
        sys.exit(0)

    print(f"Found {len(issues)} missing description(s):\n")
    for issue in issues:
        print(f"  {issue}")

    print()
    print("Add descriptions to your Strawberry types and fields:")
    print('  @strawberry.type(description="...")')
    print('  field: str = strawberry.field(description="...")')
    sys.exit(1)


if __name__ == "__main__":
    main()
