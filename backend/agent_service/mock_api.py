import json
import os
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/mock")

DATA_DIR = os.path.join(os.path.dirname(__file__), "mock_data")


def load_json(name: str):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@router.get("/dashboard")
def get_dashboard_mock():
    try:
        return load_json('dashboard_mock_data.json')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="mock data not found")


@router.get("/dev_environment")
def get_dev_env_mock():
    try:
        return load_json('dev_environment_mock.json')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="mock data not found")
