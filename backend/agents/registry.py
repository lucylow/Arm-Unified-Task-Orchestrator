"""
Plugin Registry System
-----------------------
Dynamically discovers, loads, initializes, and manages agent plugins.

Scans the `plugins/` directory for plugin classes and supports init/reload/shutdown.
"""

import os
import importlib.util
import traceback
from typing import Dict, Type, Any
from plugins.base_plugin import BasePlugin


class PluginRegistry:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_meta: Dict[str, Dict[str, Any]] = {}
        print(f"[PluginRegistry] Initialized registry for directory: {plugin_dir}")

    # --------------------------------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------------------------------
    def discover_plugins(self):
        """Scan plugin directory and identify valid Python modules."""
        print("[PluginRegistry] Discovering plugins...")
        if not os.path.isdir(self.plugin_dir):
            raise FileNotFoundError(f"Plugin directory not found: {self.plugin_dir}")

        for file_name in os.listdir(self.plugin_dir):
            if not file_name.endswith(".py") or file_name.startswith("__"):
                continue

            plugin_name = file_name[:-3]
            plugin_path = os.path.join(self.plugin_dir, file_name)
            try:
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Identify any class that subclasses BasePlugin
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                        self.plugin_meta[plugin_name] = {
                            "class": obj,
                            "path": plugin_path,
                            "status": "discovered",
                        }
                        print(f"  ✅ Found plugin: {plugin_name} ({obj.__name__})")
            except Exception as e:
                print(f"  ⚠️ Failed to import {plugin_name}: {e}")
                traceback.print_exc()

        if not self.plugin_meta:
            print("[PluginRegistry] No valid plugins found.")

    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------
    def initialize_plugins(self, config_overrides: Dict[str, Any] = None):
        """Instantiate and initialize all discovered plugins."""
        print("[PluginRegistry] Initializing plugins...")
        config_overrides = config_overrides or {}

        for name, meta in self.plugin_meta.items():
            try:
                cls = meta["class"]
                plugin = cls()
                plugin_config = config_overrides.get(name, {})
                plugin.initialize(plugin_config)
                self.plugins[name] = plugin
                meta["status"] = "initialized"
                print(f"  ✅ {name} initialized successfully.")
            except Exception as e:
                meta["status"] = "failed"
                print(f"  ❌ Failed to initialize {name}: {e}")
                traceback.print_exc()

    # --------------------------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------------------------
    def execute_plugin(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a plugin by name."""
        if name not in self.plugins:
            raise KeyError(f"Plugin '{name}' not loaded or initialized.")

        plugin = self.plugins[name]
        print(f"[PluginRegistry] Executing plugin '{name}'...")
        return plugin.process(input_data)

    # --------------------------------------------------------------------------
    # RELOADING
    # --------------------------------------------------------------------------
    def reload_plugin(self, name: str):
        """Reload a plugin dynamically (for hot updates)."""
        if name not in self.plugin_meta:
            raise KeyError(f"No such plugin: {name}")

        meta = self.plugin_meta[name]
        try:
            print(f"[PluginRegistry] Reloading plugin '{name}'...")
            spec = importlib.util.spec_from_file_location(name, meta["path"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class again
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                    cls = obj
                    plugin = cls()
                    plugin.initialize({})
                    self.plugins[name] = plugin
                    meta["status"] = "reloaded"
                    print(f"  🔄 Reloaded plugin: {name}")
        except Exception as e:
            print(f"  ❌ Reload failed for {name}: {e}")
            traceback.print_exc()

    # --------------------------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------------------------
    def shutdown_all(self):
        """Gracefully shutdown all active plugins."""
        print("[PluginRegistry] Shutting down all plugins...")
        for name, plugin in self.plugins.items():
            try:
                plugin.shutdown()
                self.plugin_meta[name]["status"] = "stopped"
                print(f"  📴 {name} stopped.")
            except Exception as e:
                print(f"  ⚠️ Error during shutdown of {name}: {e}")

        self.plugins.clear()
        print("[PluginRegistry] All plugins shut down cleanly.")
