"""Pytest collection wrapper for smoke_hardening.py."""

from __future__ import annotations

from smoke_hardening import (  # noqa: F401
    test_deterministic_routes,
    test_first_turn_surface_is_sticky_and_does_not_shrink_or_reclassify,
    test_late_pre_llm_compatibility_routes_before_request_without_explicit_agent,
    test_pre_llm_call_skips_after_core_hook_routed_turn,
    test_recovery_schema_and_core_hook_surface,
    test_request_toolset_git_expansion,
    test_request_toolset_unknown_suggestion,
    test_tool_request_middleware_expands_registered_pruned_tool_before_validation,
)