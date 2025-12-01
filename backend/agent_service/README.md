Agent Service
===============

Minimal FastAPI-based Agent-as-a-Service prototype. Contains:
- subscription & quota checks
- task execution endpoint
- natural-language parsing adapter (mock)
- plugin marketplace with example plugin

Run tests (from repository root):

```powershell
python -m pip install -r src/agent_service/requirements.txt
pytest -q
```

Replace LLM and billing adapters with real providers for production.
