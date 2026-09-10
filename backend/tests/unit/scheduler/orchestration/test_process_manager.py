# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import importlib

import pytest
from fastapi.testclient import TestClient

from scheduler.orchestration.process_manager import ProcessManager


def test_get_operation_id_is_none_when_unset():
    pm = ProcessManager()
    assert pm.get_operation_id() is None


def test_get_operation_id_returns_the_operation_process_id():
    pm = ProcessManager()
    pm.operation_process_id = "operation"
    assert pm.get_operation_id() == "operation"


def test_get_operation_process_is_none_outside_operation_mode():
    # Tests run with SCHEDULER_MODE=VALIDATION: no operation process is a
    # normal state and must not raise.
    pm = ProcessManager()
    assert pm.get_operation_process() is None


def test_get_operation_process_returns_the_process_when_set():
    pm = ProcessManager()
    sentinel = object()
    pm.active_processes["operation"] = sentinel
    pm.operation_process_id = "operation"
    assert pm.get_operation_process() is sentinel


def test_get_operation_process_raises_in_operation_mode_when_missing(monkeypatch):
    # In OPERATION (realtime) mode a missing operation process is a real
    # fault and must fail loudly, not return None.
    # scheduler.orchestration re-exports the process_manager singleton under
    # the same name as the submodule, so fetch the actual module to patch it.
    pm_module = importlib.import_module("scheduler.orchestration.process_manager")
    monkeypatch.setattr(pm_module, "is_operation", True)
    pm = ProcessManager()
    with pytest.raises(RuntimeError):
        pm.get_operation_process()


def test_get_operation_id_endpoint(monkeypatch):
    """GET /get_operation_id answers 200 with the id, or null when no
    operation process is running — never a 500."""
    from scheduler.app import app
    from scheduler.orchestration.process_manager import process_manager

    # No context manager: lifespan (DB init, process manager start) must not run.
    client = TestClient(app)

    monkeypatch.setattr(process_manager, "operation_process_id", None)
    response = client.get("/get_operation_id")
    assert response.status_code == 200
    assert response.json() == {"message": None}

    monkeypatch.setattr(process_manager, "operation_process_id", "operation")
    response = client.get("/get_operation_id")
    assert response.status_code == 200
    assert response.json() == {"message": "operation"}
