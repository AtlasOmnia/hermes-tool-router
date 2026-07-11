from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_config_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_config_test.config import _get_profile_config, _is_classifier_enabled


def test_profile_resolution_never_selects_first_enabled_profile_when_identity_unknown(monkeypatch):
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_ACTIVE_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    cfg = {
        "global": {"enabled": False},
        "profiles": {
            "alpha": {"enabled": True},
            "beta": {"enabled": True},
        },
    }
    resolved = _get_profile_config(cfg)
    assert resolved == {"enabled": False, "_profile_name": "default"}


def test_explicit_profile_is_respected(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "beta")
    cfg = {"global": {"enabled": False}, "profiles": {"beta": {"enabled": True, "floor_toolsets": []}}}
    resolved = _get_profile_config(cfg)
    assert resolved["enabled"] is True
    assert resolved["_profile_name"] == "beta"


def test_process_override_can_disable_router_for_ab_evaluation(monkeypatch):
    from hermes_tool_router_config_test.config import _is_router_active

    monkeypatch.setenv("HERMES_PROFILE", "edith")
    monkeypatch.setenv("HERMES_TOKEN_ROUTER_ENABLED", "false")
    assert _is_router_active({"profiles": {"edith": {"enabled": True}}}) is False


def test_classifier_is_opt_in():
    assert _is_classifier_enabled({}) is False
    assert _is_classifier_enabled({"classifier": {"enabled": False}}) is False
    assert _is_classifier_enabled({"classifier": {"enabled": True}}) is True


def test_classifier_model_and_provider_use_nested_v2_config():
    from hermes_tool_router_config_test.config import (
        _get_classifier_connection,
        _get_router_model,
        _get_router_provider,
    )

    cfg = {
        "classifier": {
            "provider": "custom",
            "model": "router-local",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key_env": "LOCAL_ROUTER_KEY",
        }
    }
    assert _get_router_provider(cfg) == "custom"
    assert _get_router_model(cfg) == "router-local"
    assert _get_classifier_connection(cfg) == (
        "http://127.0.0.1:1234/v1",
        "LOCAL_ROUTER_KEY",
    )
