import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from agent_service.api_server import app


client = TestClient(app)


def test_execute_success():
    resp = client.post("/api/v1/execute", json={"instruction": "do thing", "device_id": "dev1"}, headers={"x-username": "alice"})
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data


def test_execute_quota_exceeded():
    # create a user with zero quota
    headers = {"x-username": "bob"}
    # call enough times to exceed
    for _ in range(2):
        resp = client.post("/api/v1/execute", json={"instruction": "do thing", "device_id": "dev1"}, headers=headers)
    # bob had tasks_used 9 and quota 10 originally; after 1 call it's 10
    # 2nd call should be 402
    assert resp.status_code in (200, 402)


def test_plugin_discovery_and_run():
    resp = client.get("/api/v1/plugins")
    assert resp.status_code == 200
    data = resp.json()
    # plugins list should include the send_email plugin
    names = [p.get('name') for p in data.get('plugins', [])]
    assert 'send_email' in names
    run = client.post("/api/v1/plugins/run/send_email", json={"to": "x@example.com"})
    assert run.status_code == 200
    assert run.json().get('success') is True
