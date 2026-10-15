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
def _get_profile_config(cfg: dict, profile_name: str | None = None) -> dict:
    """Return the resolved config for the current Hermes profile.

    Merges: profile-specific settings → global defaults → built-in defaults.
    """
    # Explicit agent-bound name wins: a multiplexed gateway's env holds the
    # LAUNCH profile; turn-time callers pass the agent's own.
    if profile_name:
        return _get_profile_settings(cfg, profile_name)

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

    merged = _get_profile_settings(cfg, profile_name)
    if merged.get("enabled", False):
        return merged

    # Not enabled for this profile
    return {"enabled": False, "_profile_name": profile_name}


def _current_profile_name() -> str:
    explicit_profile = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_ACTIVE_PROFILE")
    if explicit_profile:
        return explicit_profile
    hermes_home = os.environ.get("HERMES_HOME", "")
    if hermes_home:
        home_path = Path(hermes_home).expanduser().resolve()
        if home_path.parent.name == "profiles":
            return home_path.name
    return "default"


def _get_profile_settings(cfg: dict, profile_name: str | None = None) -> dict:
    """Merged global + profile settings, whether or not routing is enabled.

    Used when /router turns routing on for a profile whose config says off,
    so the operator's floors, thresholds and classifier settings still apply.
    """
    if profile_name is None:
        profile_name = _current_profile_name()
    profiles_config = (cfg or {}).get("profiles", {}) or {}
    profile_cfg = profiles_config.get(profile_name, {}) or {}
    global_cfg = (cfg or {}).get("global", {}) or {}
    result = dict(global_cfg)
    result.update(profile_cfg)
    result["enabled"] = bool(profile_cfg.get("enabled", global_cfg.get("enabled", False)))
    result["_profile_name"] = profile_name
    return result


def _env_override() -> bool | None:
    """HERMES_TOKEN_ROUTER_ENABLED: True/False when set to a recognised value."""
    override = os.environ.get("HERMES_TOKEN_ROUTER_ENABLED", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    return None
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


def _get_classifier_extra_body(profile_cfg: dict) -> dict:
    """Extra body for the classifier chat.completions call.

    Convenience: classifier.disable_thinking: true ->
    {"chat_template_kwargs": {"enable_thinking": false}} (llama.cpp
    reasoning-template models, e.g. MiniCPM5, which otherwise answer in
    reasoning_content and leave content empty -> strict parser fails open).
    classifier.extra_body (dict) is merged on top. Returns {} when unset.
    """
    classifier = profile_cfg.get("classifier", {})
    if not isinstance(classifier, dict):
        return {}
    extra: dict = {}
    if classifier.get("disable_thinking"):
        kwargs = extra.setdefault("chat_template_kwargs", {})
        kwargs["enable_thinking"] = False
    configured = classifier.get("extra_body")
    if isinstance(configured, dict):
        for k, v in configured.items():
            if k == "chat_template_kwargs" and isinstance(v, dict) and isinstance(extra.get(k), dict):
                extra[k].update(v)
            else:
                extra[k] = v
    return extra


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
def _is_router_active(cfg: dict = None, profile_name: str | None = None) -> bool:
    """Check if router is enabled for the current profile or process override."""
    override = os.environ.get("HERMES_TOKEN_ROUTER_ENABLED", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    if cfg is None:
        cfg = _load_config()
    profile_cfg = _get_profile_config(cfg, profile_name)
    enabled = profile_cfg.get("enabled", False)
    profile_name = profile_cfg.get("_profile_name", "unknown")
    logger.debug(
        "%s: profile=%s enabled=%s",
        PLUGIN_NAME, profile_name, enabled,
    )
    return enabled
