"""Phase 2.5 toolset router — lightweight classifier that predicts which toolsets
a turn needs before the main model runs, shrinking the tool surface payload.

Public API (imported by agent_init.py and conversation_loop.py):

    predict_toolsets(user_message, router_config, available_toolsets)
        -> (list[str], bool)   # (predicted_names, should_reduce)

    get_missing_router_toolset_for_tool(tool_name, agent)
        -> str | None          # toolset name or None if unknown / already recalled

    expand_tools_on_router_recall(agent, toolset_name)
        -> bool                # True if tools were added
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("run_agent")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a tool-routing classifier. Given a user message, select which toolsets \
are required to handle it from the provided catalog. Reply with ONLY a JSON \
object:

{"toolsets": ["name1", "name2"], "confidence": 0.95}

Rules:
- Include only toolsets that are genuinely needed for the request.
- "confidence" is your certainty that the selected set is complete (0.0–1.0).
- Use "clarify" ONLY for pure greetings, one-word acknowledgements, or vague \
openers with zero embedded task (e.g. "hi", "ok", "thanks"). Any message that \
contains an action, question about capabilities, or request — even phrased \
casually — should map to the relevant toolset, NOT clarify.
- Current time, current date, timezone, local system state, process, disk, port, \
CPU, or OS questions require "terminal".
- When in doubt between clarify and another toolset, prefer the other toolset.
- Do not explain. Output JSON only.\
"""

# Minimum toolsets always loaded — keeps the agent useful for follow-up requests
# and avoids loading zero tools when the router says "nothing needed".
_FLOOR_TOOLSETS = ["clarify"]


def _build_user_prompt(user_message: str, available: Dict[str, str]) -> str:
    catalog_lines = "\n".join(
        f"  {name}: {desc}" for name, desc in sorted(available.items())
    )
    return f"Available toolsets:\n{catalog_lines}\n\nUser message:\n{user_message}"


# ---------------------------------------------------------------------------
# predict_toolsets
# ---------------------------------------------------------------------------

def predict_toolsets(
    user_message: str,
    router_config: Dict[str, Any],
    available_toolsets: Dict[str, str],
) -> Tuple[List[str], bool]:
    """Call the router classifier and return (predicted_names, should_reduce).

    Returns ([], False) on any failure so the caller falls back to full tools.
    """
    if not user_message or not available_toolsets:
        return [], False
    long_message_decline_chars = int(router_config.get("long_message_decline_chars") or 600)
    if long_message_decline_chars > 0 and len(user_message) >= long_message_decline_chars:
        logger.info(
            "toolset_router: declining reduction for long message chars=%s threshold=%s",
            len(user_message),
            long_message_decline_chars,
        )
        return [], False

    provider = (router_config.get("provider") or "openai-codex").strip()
    model = (router_config.get("model") or "gpt-5.4-mini").strip()
    max_tokens = int(router_config.get("max_tokens") or 200)
    temperature = float(router_config.get("temperature") if router_config.get("temperature") is not None else 0.0)
    confidence_threshold = float(router_config.get("confidence_threshold") or 0.0)
    if "confidence_threshold" not in router_config:
        logger.warning(
            "router_config.confidence_threshold is not set; defaulting to 0.0. "
            "Use 0.7 for conservative public deployments."
        )
    explicit_base_url = router_config.get("base_url") or None
    explicit_api_key = router_config.get("api_key") or None

    try:
        from agent.auxiliary_client import resolve_provider_client

        client, resolved_model = resolve_provider_client(
            provider,
            model=model,
            explicit_base_url=explicit_base_url,
            explicit_api_key=explicit_api_key,
        )
        if client is None:
            logger.debug("toolset_router: could not resolve client for provider=%s", provider)
            return [], False

        effective_model = resolved_model or model

        response = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(user_message, available_toolsets)},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        predicted: List[str] = [
            ts for ts in (parsed.get("toolsets") or [])
            if ts in available_toolsets
        ]
        confidence: float = float(parsed.get("confidence", 1.0))

        should_reduce = confidence >= confidence_threshold
        if should_reduce and not predicted:
            # Floor: router said "no tools needed" — load the minimum viable set
            # so the session stays usable for follow-up requests.
            predicted = [ts for ts in _FLOOR_TOOLSETS if ts in available_toolsets]

        logger.info(
            "toolset_router: predicted=%s confidence=%.2f should_reduce=%s",
            predicted, confidence, should_reduce,
        )
        return predicted, should_reduce

    except Exception as exc:
        logger.debug("toolset_router.predict_toolsets failed: %s", exc)
        return [], False


# ---------------------------------------------------------------------------
# get_missing_router_toolset_for_tool
# ---------------------------------------------------------------------------

def get_missing_router_toolset_for_tool(
    tool_name: str,
    agent: Any,
) -> Optional[str]:
    """Return the toolset that owns *tool_name* if it was excluded by the router
    and hasn't been recalled yet.  Returns None when recall should not proceed.
    """
    # Only attempt recall when the router actually predicted a reduced set.
    if not getattr(agent, "_router_predicted_toolsets", None):
        return None

    recalled: set = getattr(agent, "_router_recalled_toolsets", set())

    try:
        from model_tools import get_toolset_for_tool
        toolset = get_toolset_for_tool(tool_name)
    except Exception:
        return None

    if not toolset:
        return None

    # Skip if already recalled this session.
    if toolset in recalled:
        return None

    # Skip if the toolset was in the predicted set (shouldn't happen here, but
    # guard anyway — the tool may have been registered under a different name).
    predicted = getattr(agent, "_router_predicted_toolsets", None) or []
    if toolset in predicted:
        return None

    return toolset


# ---------------------------------------------------------------------------
# expand_tools_on_router_recall
# ---------------------------------------------------------------------------

def expand_tools_on_router_recall(agent: Any, toolset_name: str) -> bool:
    """Add the tools from *toolset_name* to *agent*'s live tool surface.

    Returns True if at least one new tool was injected.
    """
    try:
        from model_tools import get_tool_definitions
        recalled: set = getattr(agent, "_router_recalled_toolsets", set())

        # Fetch tools for just this toolset
        new_tools = get_tool_definitions(
            enabled_toolsets=[toolset_name],
            disabled_toolsets=None,
            quiet_mode=True,
            merge_skill_required=False,
        )
        if not new_tools:
            return False

        existing = getattr(agent, "valid_tool_names", set())
        added = 0
        for tool in new_tools:
            name = (tool.get("function") or {}).get("name", "")
            if name and name not in existing:
                agent.tools.append(tool)
                agent.valid_tool_names.add(name)
                existing.add(name)
                added += 1

        if added:
            recalled.add(toolset_name)
            agent._router_recalled_toolsets = recalled
            try:
                from agent.router_feedback import record_event

                record_event(
                    agent,
                    "router_recall",
                    toolset=toolset_name,
                    added_tools=added,
                    loaded_tool_count=len(getattr(agent, "tools", []) or []),
                )
            except Exception:
                pass
            logger.debug(
                "toolset_router recall: added %d tools from toolset '%s'",
                added, toolset_name,
            )
            return True

        # All tools were already present — still mark as recalled so we don't loop.
        recalled.add(toolset_name)
        agent._router_recalled_toolsets = recalled
        return False

    except Exception as exc:
        logger.warning("toolset_router.expand_tools_on_router_recall failed: %s", exc)
        return False
