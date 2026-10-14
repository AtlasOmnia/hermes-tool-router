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
    CAPABILITY_INQUIRY = "capability_inquiry"
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
# Questions about the assistant's own tools/capabilities.  Deliberately narrow:
# polite requests ("can you read this file") must keep their action routing, so
# a bare "can you <verb>" is NOT a capability inquiry; yes/no questions such as
# "can you take screenshots?" are left to the ordinary rules (which usually
# narrow them too, and the catalog still answers).  A match only counts when
# the message is one standalone sentence directed at the assistant and carries
# no task verb (see _is_capability_inquiry).
_CAPABILITY_INQUIRY = re.compile(
    r"\b(?:what|which)\b[^?.!\n]{0,40}\b(?:tools?|toolsets?|capabilit(?:y|ies)|abilities|integrations?)\b"
    r"|\bwhat\s+(?:else\s+)?(?:can|could)\s+you\s+(?:do|help(?:\s+me)?\s+with|access|use)"
    r"(?:\s+(?:for\s+me|for\s+us|here|today|in\s+this\s+session))?\s*[?.!]*\s*$"
    r"|\bwhat\s+(?:do|does)\s+you\s+have\s+access\s+to\b"
    r"|\bwhat\s+(?:are\s+you|is\s+this\s+assistant)\s+(?:able|capable)\b"
    r"|\b(?:list|show|enumerate|tell\s+me)\b[^?.!\n]{0,20}\b(?:your|available|loaded)\b[^?.!\n]{0,12}\b(?:tools?|toolsets?|capabilit(?:y|ies))\b",
    re.IGNORECASE,
)
_CAPABILITY_INQUIRY_MAX_CHARS = 160
_INVENTORY_TERMS = frozenset(
    [
        "tool", "tools", "toolset", "toolsets", "capability", "capabilities", "ability",
        "abilities", "integration", "integrations", "list", "show", "enumerate", "tell",
        "else", "able", "capable", "assistant", "toolbox", "disposal", "equipped", "support",
        "offer", "expose", "provide", "actually", "know", "u", "yourself",
    ]
)
_INVENTORY_FILLER = frozenset(
    [
        "a", "access", "all", "an", "any", "are", "available", "can", "could", "currently",
        "do", "does", "enabled", "for", "got", "have", "has", "help", "here", "hey", "hi",
        "i", "in", "is", "just", "loaded", "me", "now", "of", "ok", "okay", "please",
        "right", "session", "so", "the", "there", "these", "this", "to", "today", "use",
        "using", "what", "which", "with", "you", "your", "yours", "exactly", "thanks", "thank",
        "about",
        "quick", "question", "would", "like", "i'd", "id", "at", "as", "kind", "kinds", "sort",
        "sorts", "type", "types", "on", "hello", "there's", "what's", "whats", "you've", "youve"
    ]
)
_ASSISTANT_DIRECTED = re.compile(
    r"\b(?:you|your|yourself)\b"
    r"|\b(?:tools?|toolsets?)\s+(?:are\s+)?(?:loaded|available|enabled)"
    r"(?:\s+(?:right\s+)?now|\s+here|\s+in\s+this\s+session)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
# A task verb anywhere means the message asks for work, not an inventory.
_TASK_VERB = re.compile(
    r"\b(?:search|run|open|read|write|edit|install|build|test|send|create|fix|inspect|"
    r"download|upload|delete|deploy|analy[sz]e|summari[sz]e|find|look\s*up|check|review|"
    r"save|generate|make|draw|browse|click|commit|push)\b",
    re.IGNORECASE,
)
# "what tools are used in this repo" is about a local object, not the assistant.
_LOCAL_OBJECT = re.compile(
    r"\b(?:this|my|the|our|a|an|that)\s+(?:file|folder|repo|repository|project|directory|codebase|code|package|app)\b",
    re.IGNORECASE,
)
_BROWSER_ACTION = re.compile(
    r"\b(log\s*in|sign\s*in|click|submit|fill|form|navigate|interact|checkout|purchase)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://", re.IGNORECASE)


def _is_capability_inquiry(text: str) -> bool:
    if len(text) > _CAPABILITY_INQUIRY_MAX_CHARS or _URL.search(text):
        return False
    if _LOCAL_OBJECT.search(text) or _TASK_VERB.search(text):
        return False
    # Exactly one sentence: "What tools can you use? Then run pytest." is a task.
    sentences = [part for part in re.split(r"[.?!;\n]+", text) if part.strip()]
    if len(sentences) != 1 or re.search(r"\b(?:then|and\s+then|also)\b", text, re.IGNORECASE):
        return False
    match = _CAPABILITY_INQUIRY.search(text)
    if not match or not _ASSISTANT_DIRECTED.search(text):
        return False
    # The whole message must be an inventory question: every word, inside
    # or outside the matched phrase, must be inventory vocabulary or filler.
    # "List your tools and schedule a reminder" ("schedule", "reminder") and
    # "Show my cron jobs using your tools" ("my", "cron", "jobs") are tasks.
    words = re.findall(r"[a-z']+", text.lower())
    return all(word in _INVENTORY_FILLER or word in _INVENTORY_TERMS for word in words)


def classify_intent(message: str) -> IntentResult:
    """Classify obvious intent; unresolved requests fail open upstream.

    Conceptual phrasing takes precedence over subject keywords so phrases such
    as "what is a Git repository" do not accidentally grant execution tools.
    """

    text = (message or "").strip()
    lower = text.lower()
    if not text:
        return IntentResult(frozenset({Intent.FULL_SURFACE}), 0.0, "empty")

    # Checked before conceptual phrasing: "what are your capabilities?" is a
    # capability question, not general knowledge.  Anything that also carries
    # a task is left to the ordinary rules.
    if _is_capability_inquiry(text):
        return IntentResult(frozenset({Intent.CAPABILITY_INQUIRY}), 0.97, "capability_inquiry")

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
