"""Pytest collection wrapper for smoke_hardening.py."""

from __future__ import annotations

from smoke_hardening import (  # noqa: F401
    test_deterministic_routes,
    test_pre_llm_call_skips_after_core_hook_routed_turn,
    test_recovery_schema_and_core_hook_surface,
    test_request_toolset_git_expansion,
    test_request_toolset_unknown_suggestion,
)