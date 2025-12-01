import importlib, os, sys
from typing import List
from .plugin_base import WorkflowPlugin

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "plugins")


class PluginManager:
    def __init__(self, plugin_dir: str = None):
        self.plugin_dir = plugin_dir or PLUGIN_DIR

    def discover_plugins(self) -> List[WorkflowPlugin]:
        plugins = []
        if not os.path.isdir(self.plugin_dir):
            return plugins
        # Add plugin dir to path
        if self.plugin_dir not in sys.path:
            sys.path.append(self.plugin_dir)
        for f in os.listdir(self.plugin_dir):
            if f.startswith("plugin_") and f.endswith(".py"):
                name = f.replace('.py', '')
                try:
                    mod = importlib.import_module(name)
                except Exception:
                    continue
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and issubclass(obj, WorkflowPlugin) and obj is not WorkflowPlugin:
                        try:
                            plugins.append(obj())
                        except Exception:
                            continue
        return plugins

    def run_plugin(self, plugin_name: str, input_data):
        for plugin in self.discover_plugins():
            if plugin.metadata.get("name", "").lower() == plugin_name.lower() or plugin.__class__.__name__.lower() == plugin_name.lower():
                try:
                    return plugin.run(input_data)
                except Exception as e:
                    return {"success": False, "msg": f"Plugin error: {e}"}
        return {"success": False, "msg": "Plugin not found"}
