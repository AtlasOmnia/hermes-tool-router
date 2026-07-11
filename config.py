"""Configuration and profile resolution for hermes-token-router."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-token-router"
CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_ROUTER_MODEL = "deepseek-chat"
DEFAULT_ROUTER_PROVIDER = "deepseek"
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
        # No explicit profile env: infer only from the canonical Hermes home.
        # Never activate the first enabled profile by insertion order; applying
        # another profile's policy is less safe than failing closed here.
        hermes_home = os.environ.get("HERMES_HOME", "")
        inferred = "default"
        if hermes_home:
            home_path = Path(hermes_home).expanduser().resolve()
            if home_path.parent.name == "profiles":
                inferred = home_path.name
        profile_name = inferred

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
def _is_classifier_enabled(profile_cfg: dict) -> bool:
    """The external classifier is opt-in; deterministic routing works without it."""
    classifier = profile_cfg.get("classifier", {})
    return bool(isinstance(classifier, dict) and classifier.get("enabled", False))


def _get_classifier_connection(profile_cfg: dict) -> tuple[str, str]:
    """Return custom OpenAI-compatible base URL and API-key env name."""
    classifier = profile_cfg.get("classifier", {})
    if not isinstance(classifier, dict):
        return "", ""
    base_url = classifier.get("base_url")
    api_key_env = classifier.get("api_key_env")
    return (
        base_url.strip() if isinstance(base_url, str) else "",
        api_key_env.strip() if isinstance(api_key_env, str) else "",
    )


def _get_router_model(profile_cfg: dict) -> str:
    """Return the effective classifier model (v2 nested config, then legacy)."""
    classifier = profile_cfg.get("classifier", {})
    nested = classifier.get("model") if isinstance(classifier, dict) else None
    router_model = nested or profile_cfg.get("router_model", DEFAULT_ROUTER_MODEL)
    if not isinstance(router_model, str) or not router_model.strip():
        return DEFAULT_ROUTER_MODEL
    return router_model.strip()
def _get_router_provider(profile_cfg: dict) -> str:
    """Return the effective classifier provider (v2 nested config, then legacy)."""
    classifier = profile_cfg.get("classifier", {})
    nested = classifier.get("provider") if isinstance(classifier, dict) else None
    router_provider = nested or profile_cfg.get("router_provider", DEFAULT_ROUTER_PROVIDER)
    if not isinstance(router_provider, str) or not router_provider.strip():
        return DEFAULT_ROUTER_PROVIDER
    return router_provider.strip()
def _is_router_active(cfg: dict = None) -> bool:
    """Check if router is enabled for the current profile or process override."""
    override = os.environ.get("HERMES_TOKEN_ROUTER_ENABLED", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
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
