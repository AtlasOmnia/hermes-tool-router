"""Public-API adapter for Hermes's dynamic tool registry."""

from __future__ import annotations

from typing import Any, Iterable


class RegistryAdapter:
    """Keep version-specific registry access behind a small tested boundary."""

    def __init__(self, registry: Any):
        self._registry = registry

    @classmethod
    def live(cls) -> "RegistryAdapter":
        from tools.registry import registry

        return cls(registry)

    def available_toolsets(self) -> set[str]:
        return set(self._registry.get_registered_toolset_names())

    def tools_for_toolset(self, name: str) -> set[str]:
        return set(self._registry.get_tool_names_for_toolset(name) or [])

    def toolset_for_tool(self, tool_name: str) -> str | None:
        entry = self._registry.get_entry(tool_name)
        return str(entry.toolset) if entry is not None and getattr(entry, "toolset", None) else None

    def definitions_for_tools(self, names: Iterable[str]) -> list[dict]:
        return list(self._registry.get_definitions(set(names), quiet=True) or [])

    def resolve_alias(self, name: str) -> str:
        target = self._registry.get_toolset_alias_target(name)
        return str(target or name)
