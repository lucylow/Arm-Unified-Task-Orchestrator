from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlugin(ABC):
    """
    Abstract Base Class for all AI Agent plugins.
    Defines the lifecycle and I/O interface.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration data."""
        raise NotImplementedError()

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core plugin logic. Takes data, returns enhanced data."""
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully clean up resources."""
        raise NotImplementedError()
