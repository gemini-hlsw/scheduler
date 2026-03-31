#!/bin/bash
export PYTHONPATH=$(pwd)/backend
source backend/.env

uv run strawberry export-schema backend.scheduler.graphql_mid.server > backend/scheduler.graphql
