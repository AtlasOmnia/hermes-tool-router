"""High-precision intent classification for deterministic tool routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class Intent(str, Enum):
    ANSWER_ONLY = "answer_only"
    READ_LOCAL = "read_local"
    WRITE_LOCAL = "write_local"
    EXECUTE_LOCAL = "execute_local"
    EXECUTE_CODE = "execute_code"
    RESEARCH_WEB = "research_web"
    INTERACT_BROWSER = "interact_browser"
    CONTROL_DESKTOP = "control_desktop"
    ANALYZE_IMAGE = "analyze_image"
    GENERATE_IMAGE = "generate_image"
    ANALYZE_VIDEO = "analyze_video"
    GENERATE_VIDEO = "generate_video"
    MEMORY_OPERATION = "memory_operation"
    SKILL_OPERATION = "skill_operation"
    TODO_OPERATION = "todo_operation"
    SESSION_LOOKUP = "session_lookup"
    SCHEDULE_OPERATION = "schedule_operation"
    DELEGATE = "delegate"
    GIT_OPERATION = "git_operation"
    ASK_CLARIFICATION = "ask_clarification"
    FULL_SURFACE = "full_surface"


@dataclass(frozen=True)
class IntentResult:
    intents: FrozenSet[Intent]
    confidence: float
    reason_code: str


_CONCEPTUAL = re.compile(
    r"^\s*(what\s+(?:is|are)|who\s+(?:is|are)|why\b|explain\b|define\b|"
    r"compare\b|difference\s+between\b|how\s+(?:does|do|can|to)\b)",
    re.IGNORECASE,
)
_BROWSER_ACTION = re.compile(
    r"\b(log\s*in|sign\s*in|click|submit|fill|form|navigate|interact|checkout|purchase)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://", re.IGNORECASE)


def classify_intent(message: str) -> IntentResult:
    """Classify obvious intent; unresolved requests fail open upstream.

    Conceptual phrasing takes precedence over subject keywords so phrases such
    as "what is a Git repository" do not accidentally grant execution tools.
    """

    text = (message or "").strip()
    lower = text.lower()
    if not text:
        return IntentResult(frozenset({Intent.FULL_SURFACE}), 0.0, "empty")

    if _CONCEPTUAL.search(text) and not re.search(
        r"\b(current|latest|today|now|my|this\s+(?:file|repo|repository|image|screenshot))\b|https?://",
        lower,
    ):
        return IntentResult(frozenset({Intent.ANSWER_ONLY}), 0.99, "conceptual")

    explicit_intents: set[Intent] = set()
    if re.search(r"\b(code[_ -]?execution|execute_code|sandboxed code)\b", lower):
        explicit_intents.add(Intent.EXECUTE_CODE)
    if re.search(r"\b(skill_view|skills? tool|load (?:the )?[a-z0-9_-]+ skill)\b", lower):
        explicit_intents.add(Intent.SKILL_OPERATION)
    if re.search(r"\b(session[_ -]?search|previous session|past conversation|session history)\b", lower):
        explicit_intents.add(Intent.SESSION_LOOKUP)
    if re.search(r"\b(cronjob|cron job|scheduled jobs?)\b", lower):
        explicit_intents.add(Intent.SCHEDULE_OPERATION)
    if re.search(r"\b(todo tool|task list)\b", lower):
        explicit_intents.add(Intent.TODO_OPERATION)
    if explicit_intents:
        return IntentResult(frozenset(explicit_intents), 0.99, "explicit_tool_intent")

    if re.search(
        r"\b(computer[- _]?use|desktop control|control (?:the )?(?:desktop|computer)|"
        r"capture (?:the )?(?:safari|chrome|browser|desktop|screen|window)|"
        r"(?:safari|chrome) window|desktop window)\b",
        lower,
    ):
        return IntentResult(frozenset({Intent.CONTROL_DESKTOP}), 0.99, "desktop_control")

    if re.search(r"\b(screenshot|photo|picture|image|diagram)\b", lower):
        if re.search(r"\b(generate|create|draw|design|make)\b", lower):
            return IntentResult(frozenset({Intent.GENERATE_IMAGE}), 0.98, "generate_image")
        if re.search(r"\b(review|inspect|analy[sz]e|describe|read|look at)\b", lower):
            return IntentResult(frozenset({Intent.ANALYZE_IMAGE}), 0.98, "analyze_image")

    if re.search(r"\b(video|movie|clip|animation)\b", lower):
        if re.search(r"\b(generate|create|make|animate)\b", lower):
            return IntentResult(frozenset({Intent.GENERATE_VIDEO}), 0.96, "generate_video")
        if re.search(r"\b(analy[sz]e|review|describe|inspect)\b", lower):
            return IntentResult(frozenset({Intent.ANALYZE_VIDEO}), 0.96, "analyze_video")

    if _URL.search(text) and _BROWSER_ACTION.search(text):
        return IntentResult(frozenset({Intent.INTERACT_BROWSER}), 0.99, "interactive_url")

    intents: set[Intent] = set()
    if _URL.search(text) or re.search(
        r"\b(search|research|look\s*up|latest|current|today|news|weather|price|online)\b",
        lower,
    ):
        intents.add(Intent.RESEARCH_WEB)
    if _BROWSER_ACTION.search(text) or re.search(r"\bopen (?:the )?(?:site|website|browser)\b", lower):
        intents.add(Intent.INTERACT_BROWSER)
        intents.discard(Intent.RESEARCH_WEB)
    if re.search(r"\b(read|inspect|review|search)\b.{0,30}\b(file|folder|path|repo|repository)\b", lower):
        intents.add(Intent.READ_LOCAL)
    if re.search(r"\b(save|write|edit|modify|patch|rename|copy|move|delete)\b", lower):
        intents.add(Intent.WRITE_LOCAL)
    if re.search(r"\b(run|execute|install|build|test|pytest|npm|compile|lint|shell|terminal|command)\b", lower):
        intents.add(Intent.EXECUTE_LOCAL)
    if re.search(r"\b(commit|push|pull|checkout|merge|git\s+diff)\b", lower):
        intents.update({Intent.READ_LOCAL, Intent.WRITE_LOCAL, Intent.EXECUTE_LOCAL, Intent.GIT_OPERATION})
    if re.search(r"\b(remember|store this|save this to memory)\b", lower):
        intents.add(Intent.MEMORY_OPERATION)
    if re.search(r"\b(previous session|past conversation|session history|where did we leave)\b", lower):
        intents.add(Intent.SESSION_LOOKUP)
    if re.search(r"\b(schedule|cron|remind me|every day|every week)\b", lower):
        intents.add(Intent.SCHEDULE_OPERATION)
    if re.search(r"\b(delegate|subagent|sub-agent)\b", lower):
        intents.add(Intent.DELEGATE)

    if intents:
        return IntentResult(frozenset(intents), 0.95, "deterministic_actions")

    if _CONCEPTUAL.search(text):
        return IntentResult(frozenset({Intent.ANSWER_ONLY}), 0.95, "conceptual")

    return IntentResult(frozenset({Intent.FULL_SURFACE}), 0.0, "unresolved")


INTENT_TOOLSETS: dict[Intent, frozenset[str]] = {
    Intent.READ_LOCAL: frozenset({"file"}),
    Intent.WRITE_LOCAL: frozenset({"file"}),
    Intent.EXECUTE_LOCAL: frozenset({"terminal"}),
    Intent.EXECUTE_CODE: frozenset({"code_execution"}),
    Intent.RESEARCH_WEB: frozenset({"web"}),
    Intent.INTERACT_BROWSER: frozenset({"browser"}),
    Intent.CONTROL_DESKTOP: frozenset({"computer_use"}),
    Intent.ANALYZE_IMAGE: frozenset({"vision"}),
    Intent.GENERATE_IMAGE: frozenset({"image_gen"}),
    Intent.ANALYZE_VIDEO: frozenset({"video"}),
    Intent.GENERATE_VIDEO: frozenset({"video_gen"}),
    Intent.MEMORY_OPERATION: frozenset({"memory"}),
    Intent.SKILL_OPERATION: frozenset({"skills"}),
    Intent.TODO_OPERATION: frozenset({"todo"}),
    Intent.SESSION_LOOKUP: frozenset({"session_search"}),
    Intent.SCHEDULE_OPERATION: frozenset({"cronjob"}),
    Intent.DELEGATE: frozenset({"delegation"}),
    Intent.GIT_OPERATION: frozenset({"git"}),
    Intent.ASK_CLARIFICATION: frozenset({"clarify"}),
}
