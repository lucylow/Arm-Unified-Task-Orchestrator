from typing import Dict, Any
from datetime import datetime

# Central in-memory store used by agent_service modules. Replace with real DB in production.
users_db: Dict[str, Dict[str, Any]] = {
    "alice": {"username": "alice", "subscription": "premium", "quota": 100, "tasks_used": 42, "history": []},
    "bob":   {"username": "bob", "subscription": "basic", "quota": 10,  "tasks_used": 9,  "history": []}
}

def record_user_task(username: str, record: Dict[str, Any]):
    u = users_db.get(username)
    if not u:
        return
    rec = dict(record)
    rec.setdefault('timestamp', datetime.utcnow().isoformat() + 'Z')
    u.setdefault('history', []).append(rec)
