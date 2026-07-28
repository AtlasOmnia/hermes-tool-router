"""Toolset prediction policy for hermes-token-router."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

try:
    from .config import PLUGIN_NAME
    from .classifier import call_with_hard_timeout
    from .intent import INTENT_TOOLSETS, Intent, classify_intent
except ImportError:  # pragma: no cover - direct loader fallback
    from config import PLUGIN_NAME
    from classifier import call_with_hard_timeout
    from intent import INTENT_TOOLSETS, Intent, classify_intent

logger = logging.getLogger(__name__)

TOOLSET_DESCRIPTIONS: Dict[str, str] = {
    "terminal":         "Run shell commands, install packages, manage processes",
    "web":              "Search and extract content from web pages",
    "file":             "Read, write, search, and manage files",
    "memory":           "Store and retrieve long-term memories, manage recall",
    "skills":           "Create, edit, view, and manage Hermes skills",
    "browser":          "Navigate, click, fill forms, interact with web pages",
    "image_gen":        "Generate images via DALL-E or Stable Diffusion",
    "video_gen":        "Generate videos",
    "video":            "Analyze or generate video content",
    "vision":           "Analyze images — screenshots, photos, diagrams",
    "git":              "Git operations — commit, push, pull, branch, clone",
    "code_execution":   "Run sandboxed code (Python, Node, shell) safely",
    "kanban":           "Kanban board — create, assign, move tasks",
    "context_engine":   "Context-aware processing with custom rules",
    "mcp":              "MCP server tool discovery and interaction",
    "approval":         "Approve dangerous shell commands mid-execution",
    "tts":              "Text-to-speech audio generation",
    "acp":              "ACP filesystem and edit operations",
    "discord":          "Discord bot — send messages, manage channels",
    "slack":            "Slack integration — messages, reactions, threads",
    "telegram":         "Telegram bot integration",
    "delegation":       "Delegate tasks to subagents",
    "messaging":        "Send messages through configured gateway platforms",
}
TOOLSET_INTENT_RULES: List[tuple[str, Set[str]]] = [
    (r"\b(reddit|subreddit|r/[A-Za-z0-9_]+)\b", {"web", "browser"}),
    (r"\b(open|click|browse|browser|web\s?page|screenshot|log\s?in|form|navigate)\b|https?://", {"browser", "web"}),
    (r"\b(search|look\s?up|google|web|online|current|latest|today|news|weather|price|schedule)\b", {"web"}),
    (r"\b(file|folder|path|read|write|edit|save|delete|rename|copy|move|upload|download)\b", {"file"}),
    (r"\b(run|command|terminal|shell|install|build|test|pytest|npm|python|powershell|process)\b", {"terminal"}),
    (r"\b(git|commit|branch|checkout|diff|merge|push|pull|repo|repository|pr|pull request)\b", {"git", "file", "terminal"}),
    (r"\b(code|debug|fix|implement|refactor|compile|lint)\b", {"file", "terminal"}),
    (r"\b(image|picture|photo|diagram|screenshot|vision|analy[sz]e this)\b", {"vision"}),
    (r"\b(generate|create|draw|make).{0,40}\b(image|picture|photo|art)\b", {"image_gen"}),
    (r"\b(video|movie|clip|animation)\b", {"video_gen", "video"}),
    (r"\b(remember|memory|recall|save this|store this)\b", {"memory"}),
    (r"\b(skill|skills)\b", {"skills"}),
    (r"\b(delegate|subagent|sub-agent|multi-agent|worker agent)\b", {"delegation"}),
    (r"\b(discord)\b", {"discord", "messaging"}),
    (r"\b(slack)\b", {"slack", "messaging"}),
    (r"\b(telegram)\b", {"telegram", "messaging"}),
    (r"\b(speak|say aloud|tts|voice)\b", {"tts"}),
]
ACTION_HINT_RE = re.compile(
    r"\b("
    r"access|analy[sz]e|browse|build|call|change|check|click|commit|create|debug|"
    r"download|edit|find|fix|generate|install|look\s?up|make|modify|open|read|"
    r"remember|repair|research|run|save|screenshot|search|send|test|update|write"
    r")\b|https?://",
    re.IGNORECASE,
)
PLAIN_ANSWER_RE = re.compile(
    r"\b("
    r"what is|what are|who is|who are|why|how does|how do|explain|define|"
    r"compare|difference between|summari[sz]e concept|tell me about"
    r")\b",
    re.IGNORECASE,
)
ROUTER_SYSTEM_PROMPT = """You are a tool-router classifier. Given a user message and a list of available toolsets with descriptions, predict which toolsets the AI assistant will need to respond.

Rules:
1. Return ONLY JSON: {{"toolsets": ["name"], "confidence": 0.0}}.
2. confidence must be a number from 0.0 to 1.0.
3. Include ONLY toolsets directly relevant to the user's request.
4. Do not include terminal, file, or web unless the request actually needs them.
5. For ordinary questions that can be answered without tools, return an empty array.
6. If the message is ambiguous or uncertain, return "all" as the only toolset.
7. Be conservative; only include toolsets you are confident are needed.
8. Personal calendar, itinerary, reservation, or flight-detail lookups require "skills" and "terminal" when those toolsets are available.
Example response:
{{"toolsets": ["terminal", "file"], "confidence": 0.96}}

Available toolsets:
{toolset_descriptions}

User message: {user_message}
"""
def _get_router_client(
    router_model: str,
    router_provider: str = "deepseek",
    custom_base_url: str = "",
    custom_api_key_env: str = "",
):
    """Obtain an OpenAI-compatible client for the router.

    Supports:
      - deepseek: uses DEEPSEEK_API_KEY, standard Chat Completions API
      - openrouter: uses OPENROUTER_API_KEY, standard Chat Completions API
      - openai-codex: uses OAuth credential pool (Responses API — not yet implemented)

    Returns (client, model_name) or (None, None) on failure.
    """
    try:
        from openai import OpenAI

        if router_provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                logger.debug("%s: no DEEPSEEK_API_KEY available", PLUGIN_NAME)
                return None, None
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=8,
                max_retries=0,
            )
            return client, router_model

        elif router_provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                logger.debug("%s: no OPENROUTER_API_KEY available", PLUGIN_NAME)
                return None, None
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=8,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/hermes-agent/hermes-token-router",
                    "X-Title": "Hermes Tool Router",
                },
            )
            return client, router_model

        elif router_provider == "custom":
            if not custom_base_url:
                logger.debug("%s: custom classifier requires base_url", PLUGIN_NAME)
                return None, None
            api_key = os.environ.get(custom_api_key_env, "") if custom_api_key_env else "local"
            if custom_api_key_env and not api_key:
                logger.debug(
                    "%s: custom classifier key env %s is unset",
                    PLUGIN_NAME,
                    custom_api_key_env,
                )
                return None, None
            client = OpenAI(
                api_key=api_key,
                base_url=custom_base_url,
                timeout=2,
                max_retries=0,
            )
            return client, router_model

        elif router_provider == "openai-codex":
            logger.warning(
                "%s: router_provider=openai-codex is disabled here because the "
                "Codex backend requires the Responses API; using full toolset fallback",
                PLUGIN_NAME,
            )
            return None, None

        else:
            logger.warning(
                "%s: unknown router_provider '%s'", PLUGIN_NAME, router_provider
            )
            return None, None

    except Exception as exc:
        logger.warning(
            "%s: failed to initialize router client: %s",
            PLUGIN_NAME,
            exc,
        )
        return None, None
def _get_available_toolsets() -> Set[str]:
    """Return all known toolset names from the registry."""
    try:
        from tools.registry import registry
        return set(registry.get_registered_toolset_names())
    except Exception:
        return set()
def _build_toolset_description_block(available: Set[str]) -> str:
    """Build a formatted string of toolset descriptions for the router prompt."""
    lines = []
    for name in sorted(available):
        desc = TOOLSET_DESCRIPTIONS.get(name, f"Tools for {name}")
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
def _predict_toolsets_by_rules(
    user_message: str,
    available_toolsets: Set[str],
) -> tuple[Optional[Set[str]], str]:
    """Fast deterministic routing for obvious intents.

    Returns (toolsets, reason). toolsets=None means the LLM router should decide.
    An empty set is a valid prediction for plain-answer turns.
    """
    text = (user_message or "").strip()
    lowered = text.lower()
    if not lowered:
        return None, "empty"

    intent_result = classify_intent(text)
    if intent_result.intents == frozenset({Intent.ANSWER_ONLY}):
        return set(), f"intent:{intent_result.reason_code}"
    if Intent.FULL_SURFACE not in intent_result.intents:
        intent_toolsets: Set[str] = set()
        for intent in intent_result.intents:
            intent_toolsets.update(INTENT_TOOLSETS.get(intent, frozenset()))
        intent_toolsets &= available_toolsets
        if intent_toolsets:
            return intent_toolsets, f"intent:{intent_result.reason_code}"

    matched: Set[str] = set()
    matched_rules = 0
    for pattern, toolsets in TOOLSET_INTENT_RULES:
        if re.search(pattern, lowered, re.IGNORECASE):
            matched.update(toolsets)
            matched_rules += 1

    matched &= available_toolsets
    if matched:
        return matched, f"rules:{matched_rules}"

    words = re.findall(r"[a-z0-9']+", lowered)
    if len(words) <= 4 and not ACTION_HINT_RE.search(lowered):
        return set(), "plain_short"

    if PLAIN_ANSWER_RE.search(lowered) and not ACTION_HINT_RE.search(lowered):
        return set(), "plain_answer"

    if not ACTION_HINT_RE.search(lowered) and len(text) <= 240:
        return set(), "plain_default"

    return None, "needs_llm"
def _extract_confidence(result: Any) -> Optional[float]:
    """Extract confidence from router output; missing confidence is unknown."""
    if not isinstance(result, dict):
        return None

    for key in ("confidence", "probability", "prob", "score", "confidence_score"):
        raw = result.get(key)
        if raw is None or isinstance(raw, bool):
            continue

        parsed: Optional[float]
        if isinstance(raw, (int, float)):
            parsed = float(raw)
        elif isinstance(raw, str):
            try:
                parsed = float(raw.strip().replace("%", ""))
            except ValueError:
                parsed = None
        else:
            parsed = None

        if parsed is None or parsed < 0:
            continue

        if parsed > 1 and parsed <= 100:
            parsed = parsed / 100.0

        return parsed

    return None
def _predict_toolsets_via_llm(
    user_message: str,
    available_toolsets: Set[str],
    router_model: str,
    confidence_threshold: float,
    router_provider: str = "deepseek",
    hard_timeout_seconds: float = 1.2,
    custom_base_url: str = "",
    custom_api_key_env: str = "",
) -> Optional[Set[str]]:
    """Use the router model to predict which toolsets are needed.

    Returns a reduced set of toolsets, or None to use full set (bypass).
    On any error, returns None for safe fallback.
    """
    try:
        client, model = _get_router_client(
            router_model,
            router_provider,
            custom_base_url,
            custom_api_key_env,
        )
        if client is None:
            logger.debug(
                "%s: router client unavailable — full set fallback",
                PLUGIN_NAME,
            )
            return None


        toolset_desc = _build_toolset_description_block(available_toolsets)
        
        prompt = ROUTER_SYSTEM_PROMPT.format(
            toolset_descriptions=toolset_desc,
            user_message=user_message[:1500],  # Limit message length for router
        )

        start = time.monotonic()
        logger.debug(
            "%s: calling chat.completions.create provider=%s model=%s",
            PLUGIN_NAME,
            router_provider,
            model,
        )

        def invoke_classifier():
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )

        try:
            completed, response = call_with_hard_timeout(invoke_classifier, hard_timeout_seconds)
            if not completed or response is None:
                logger.warning(
                    "%s: router prediction timed out after %.2fs — full set fallback",
                    PLUGIN_NAME,
                    hard_timeout_seconds,
                )
                return None
        except Exception as thread_exc:
            logger.warning(
                "%s: router API call failed: %s — full set fallback",
                PLUGIN_NAME, thread_exc,
            )
            return None
        elapsed = time.monotonic() - start

        content = response.choices[0].message.content or ""
        
        logger.debug(
            "%s: router model=%s responded in %.2fs: %.100s",
            PLUGIN_NAME,
            model,
            elapsed,
            content.strip(),
        )

        # Parse JSON response
        content = content.strip()
        # Remove markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        result = json.loads(content)
        predicted_list = result.get("toolsets", [])

        confidence = _extract_confidence(result)
        if confidence is None:
            logger.info("%s: router omitted confidence; keeping full toolset", PLUGIN_NAME)
            return None
        if confidence < confidence_threshold:
            logger.info(
                "%s: router confidence %.3f below threshold %.3f; keeping full toolset",
                PLUGIN_NAME,
                confidence,
                confidence_threshold,
            )
            return None

        if not isinstance(predicted_list, list):
            logger.debug(
                "%s: router returned non-list: %s",
                PLUGIN_NAME, result,
            )
            return None

        # Handle "all" bypass signal
        if "all" in predicted_list:
            logger.debug(
                "%s: router signaled 'all' — full set fallback",
                PLUGIN_NAME,
            )
            return None

        # Filter to only known toolsets. An empty list is a valid no-tool prediction.
        predicted = set(predicted_list) & available_toolsets

        if not predicted and predicted_list:
            logger.debug(
                "%s: router returned empty set after filtering — full set fallback",
                PLUGIN_NAME,
            )
            return None

        # If predicted is the same as full set, no reduction needed
        if predicted == available_toolsets:
            logger.debug(
                "%s: router predicted all available toolsets — no reduction",
                PLUGIN_NAME,
            )
            return None

        logger.info(
            "%s: router model=%s predicted %d toolsets (from %d) in %.2fs: %s",
            PLUGIN_NAME,
            model,
            len(predicted),
            len(available_toolsets),
            elapsed,
            sorted(predicted),
        )

        return predicted

    except json.JSONDecodeError as exc:
        logger.warning(
            "%s: failed to parse router response: %s",
            PLUGIN_NAME, exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            "%s: router prediction failed: %s",
            PLUGIN_NAME, exc,
        )
        return None
