import os
import json
import numpy as np
from typing import Dict, Any, List
from plugins.base_plugin import BasePlugin


class MemoryCachePlugin(BasePlugin):
    """
    MemoryCachePlugin provides short-term and semantic memory to the agent.
    It stores key-value embeddings and contextual traces from previous tasks.
    """

    def __init__(self):
        self.memory_dir = None
        self.cache_limit = 100  # max entries
        self.memory_index: List[Dict[str, Any]] = []
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Prepare memory directory and config parameters."""
        self.memory_dir = config.get("memory_dir", "./agent_memory/")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.cache_limit = config.get("cache_limit", 100)
        self.initialized = True
        print(f"[MemoryCache] Initialized at {self.memory_dir}")

    def _prune_memory(self):
        """Remove oldest memory if over cache limit."""
        if len(self.memory_index) > self.cache_limit:
            self.memory_index = self.memory_index[-self.cache_limit:]
            print(f"[MemoryCache] Pruned to {self.cache_limit} entries.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts:
            {
                "embedding": [float],
                "summary": "Agent observed X and did Y",
                "timestamp": float
            }
        Returns:
            {
                "recent_memories": [summaries],
                "semantic_similarity": float
            }
        """
        if not self.initialized:
            raise RuntimeError("MemoryCachePlugin must be initialized before use.")

        embedding = np.array(input_data.get("embedding", []))
        summary = input_data.get("summary", "No summary provided.")
        timestamp = input_data.get("timestamp")

        memory_entry = {
            "summary": summary,
            "embedding": embedding.tolist(),
            "timestamp": timestamp
        }

        self.memory_index.append(memory_entry)
        self._prune_memory()

        # Persist memory
        with open(os.path.join(self.memory_dir, "memory_log.json"), "w") as f:
            json.dump(self.memory_index, f, indent=2)

        # Compute semantic similarity to last memory
        if len(self.memory_index) > 1 and embedding.size > 0:
            prev = np.array(self.memory_index[-2]["embedding"])
            try:
                sim = float(np.dot(embedding, prev) / (np.linalg.norm(embedding) * np.linalg.norm(prev)))
            except Exception:
                sim = 0.0
        else:
            sim = 0.0

        return {
            "recent_memories": [m["summary"] for m in self.memory_index[-3:]],
            "semantic_similarity": round(float(sim), 4)
        }

    def shutdown(self) -> None:
        """Save state and clean up."""
        print("[MemoryCache] Shutting down...")
        self.initialized = False
