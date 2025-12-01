from typing import Dict, Any


class LLMSandbox:
    """A tiny mock LLM adapter that 'parses' natural language into a task dict.
    This is intentionally simplistic — replace with real LLM integration.
    """

    def parse_to_task(self, query: str) -> Dict[str, Any]:
        # Very naive parsing: look for 'to' and 'device' tokens
        # Example: "Turn on device 123 with brightness 80"
        task = {"instruction": query, "device_id": "unknown", "parameters": {}}
        words = query.split()
        for i, w in enumerate(words):
            if w.lower() in ("device", "device:") and i + 1 < len(words):
                task["device_id"] = words[i + 1]
            if w.lower().startswith("brightness"):
                try:
                    val = int(w.split("=")[-1])
                    task["parameters"]["brightness"] = val
                except Exception:
                    pass
        # fallback device id
        if task["device_id"] == "unknown":
            task["device_id"] = "default-device"
        return task
