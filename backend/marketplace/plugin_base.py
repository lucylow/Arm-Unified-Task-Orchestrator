from abc import ABC, abstractmethod
from typing import Any, Dict


class WorkflowPlugin(ABC):
    """Base class for workflow plugins. Plugins should subclass this and provide
    metadata and a run() method.
    """

    metadata = {"name": "base", "author": "unknown", "version": "0.0.1", "description": ""}

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()
