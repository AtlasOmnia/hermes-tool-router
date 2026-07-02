"""Configuration and profile resolution for hermes-token-router."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-token-router"
CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_ROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_ROUTER_PROVIDER = "openrouter"
def _load_config() -> dict:
    """Load router config from plugin's config.yaml.

    Returns a dict with per-profile settings and global defaults merged.
    """
    try:
        import yaml
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("%s: failed to load config: %s", PLUGIN_NAME, exc)
    return {}
def _get_profile_config(cfg: dict) -> dict:
    """Return the resolved config for the current Hermes profile.

    Merges: profile-specific settings → global defaults → built-in defaults.
    """
    # Determine active profile from env (in priority order).
    explicit_profile = os.environ.get("HERMES_PROFILE") or \
                       os.environ.get("HERMES_ACTIVE_PROFILE")

    if explicit_profile:
        profile_name = explicit_profile
    else:
        # No explicit profile env; prefer any enabled profile in our config.
        # This lets the router work even when Hermes doesn't export HERMES_PROFILE.
        profiles_cfg = cfg.get("profiles", {}) or {}
        candidate = next(
            (name for name, v in profiles_cfg.items() if v.get("enabled")),
            None,
        )

        # Fallback: infer from HERMES_HOME path.
        hermes_home = os.environ.get("HERMES_HOME", "")
        inferred = "default"
        if hermes_home:
            profiles_root = Path(hermes_home).parent / "profiles"
            if profiles_root.exists():
                for child in profiles_root.iterdir():
                    if child.is_dir() and str(child.resolve()) == Path(hermes_home).resolve():
                        inferred = child.name
                        break

        # Use enabled config profile if present; else use inferred.
        profile_name = candidate or inferred

    # Look up per-profile config
    profiles_config = cfg.get("profiles", {})
    profile_cfg = profiles_config.get(profile_name, {})

    # Fall back to global
    global_cfg = cfg.get("global", {})
    if profile_cfg.get("enabled", global_cfg.get("enabled", False)):
        result = dict(global_cfg)
        result.update(profile_cfg)
        result["_profile_name"] = profile_name
        return result

    # Not enabled for this profile
    return {"enabled": False, "_profile_name": profile_name}
def _get_router_model(profile_cfg: dict) -> str:
    """Return the effective router model from config."""
    router_model = profile_cfg.get("router_model", DEFAULT_ROUTER_MODEL)
    if not isinstance(router_model, str) or not router_model.strip():
        return DEFAULT_ROUTER_MODEL
    return router_model.strip()
def _get_router_provider(profile_cfg: dict) -> str:
    """Return the effective router provider from config."""
    router_provider = profile_cfg.get("router_provider", DEFAULT_ROUTER_PROVIDER)
    if not isinstance(router_provider, str) or not router_provider.strip():
        return DEFAULT_ROUTER_PROVIDER
    return router_provider.strip()
def _is_router_active(cfg: dict = None) -> bool:
    """Check if router is enabled for the current profile."""
    if cfg is None:
        cfg = _load_config()
    profile_cfg = _get_profile_config(cfg)
    enabled = profile_cfg.get("enabled", False)
    profile_name = profile_cfg.get("_profile_name", "unknown")
    logger.debug(
        "%s: profile=%s enabled=%s",
        PLUGIN_NAME, profile_name, enabled,
    )
    return enabled
