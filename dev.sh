#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/packages/scheduler/scheduler
source packages/scheduler/.env

# First generate graphql types
uv run strawberry export-schema packages.scheduler.scheduler.graphql_mid.server > packages/schema/scheduler.graphql

# Install node dependencies
pnpm install
pnpm schedule codegen

# Start servers
pnpm weather dev &
pnpm web dev &
uv run python packages/scheduler/scheduler/main.py