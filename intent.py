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
# Mechanical union of the existing English action literals used below.  The
# residual Chinese conceptual rule must fail open when any of these signals is present.
_ENGLISH_ACTION_SIGNAL = re.compile(
    r"\b(?:"
    r"computer[- _]?use|desktop control|control (?:the )?(?:desktop|computer)|"
    r"capture (?:the )?(?:safari|chrome|browser|desktop|screen|window)|"
    r"(?:safari|chrome) window|desktop window|"
    r"screenshot|photo|picture|image|diagram|"
    r"video|movie|clip|animation|"
    r"log\s*in|sign\s*in|click|submit|fill|form|navigate|interact|checkout|purchase|"
    r"search|research|look\s*up|latest|current|today|news|weather|price|online|"
    r"read|inspect|review|"
    r"save|write|edit|modify|patch|rename|copy|move|delete|"
    r"run|execute|install|build|test|pytest|npm|compile|lint|shell|terminal|command|"
    r"commit|push|pull|checkout|merge|git\s+diff|"
    r"remember|store this|save this to memory|"
    r"previous session|past conversation|session history|where did we leave|"
    r"schedule|cron|remind me|every day|every week|"
    r"delegate|subagent|sub-agent"
    r")\b|https?://",
    re.IGNORECASE,
)
_CONCEPTUAL_FRESHNESS_OR_URL = re.compile(
    r"\b(current|latest|today|now|my|this\s+(?:file|repo|repository|image|screenshot))\b"
    r"|最新|当前|今天|现在|我的|https?://",
    re.IGNORECASE,
)
_CHINESE_LEADING_CONCEPTUAL = re.compile(
    r"^\s*(?:请\s*(?:问\s*)?[，,:：]?\s*)?(?:什么是|为什么|为啥|怎么(?!样)|怎样|如何|比较|定义|谁是|"
    r"讲讲|说说|请教|解释(?=一下|这个|该|原理|原因|[\s，。！？：?]|$))"
)
# Keep the …和别的方式有什么区别 family deferred: this sentence-final
# conceptual rule runs before the Chinese action merge, and its guards cannot
# distinguish 上网查一下这两个模型有什么区别 (must stay {web}) from
# 运行测试和别的方式有什么区别. A Chinese imperative-lead exemption
# mirroring _english_imperative_material_precedes_sentence_final would not
# discriminate either, because both carry Chinese action verbs.
_CHINESE_SENTENCE_FINAL_CONCEPTUAL = re.compile(
    r"(?:是什么|什么意思|是干什么的|有什么用|是怎么回事)\s*[。！？!?]*$"
)
_ENGLISH_IMPERATIVE_LEAD = re.compile(
    r"^\s*(?:"
    r"control|capture|take|generate|create|draw|design|make|analy[sz]e|describe|animate|"
    r"log\s*in|sign\s*in|click|submit|fill|navigate|interact|checkout|purchase|open|"
    r"find|search|research|look\s*up|read|inspect|review|"
    r"save|write|edit|modify|patch|rename|copy|move|delete|"
    r"run|execute|install|build|test|compile|lint|"
    r"commit|push|pull|merge|"
    r"remember|store|schedule|remind|delegate|"
    r"use|load"
    r")\b",
    re.IGNORECASE,
)
_CHINESE_EXPLICIT_REQUEST_LEAD = re.compile(r"^\s*(?:帮我|请|麻烦|给我|帮忙)")
_CHINESE_FILESYSTEM_PATH = re.compile(
    r"(?:^|[\s：:]|(?<=[\u4e00-\u9fff]))(?:/|\\)[A-Za-z0-9_./\\-]+"
)
_CHINESE_RESIDUAL_CONCEPTUAL = re.compile(
    r"(?:什么是|是什么|为什么|有什么区别|区别|原理|"
    r"解释(?=一下|这个|该|原理|原因|[\s，。！？：?]|$)|"
    r"介绍(?=一下|这个|该|[\s，。！？：?]|$))"
)
_CHINESE_READ_ACTION = re.compile(
    r"(?:帮我|请|麻烦)?(?:看看|看一下|查看|读取|读一下|打开|显示)"
    r".{0,48}(?:文件|文档|目录|路径|内容|文本)"
    r"|(?:帮我|请|麻烦)?(?:读取|查看|打开|显示)(?:/|\\)[A-Za-z0-9_./\\-]+",
    re.IGNORECASE,
)
_CHINESE_WEB_SEARCH_ACTION = re.compile(
    r"(?:上网|网上|在线)(?:查|搜索|搜|查询)(?:一下)?"
    r"|(?:帮我|请|麻烦)?(?:查一下|查询一下|搜索一下|搜一下|查找一下)"
    r".{0,80}(?:最新|当前|信息|资料|新闻|天气|跑分|基准|benchmark|网址|网页|网站|https?://)",
    re.IGNORECASE,
)
_CHINESE_EXECUTE_ACTION = re.compile(
    r"(?:把|将).{0,8}(?:目录|文件夹|项目).{0,32}"
    r"(?:测试|pytest|test|命令|脚本).{0,12}(?:跑一遍|跑一下|运行|执行)"
    r"|(?:请|帮我)?(?:运行|执行|跑一下|跑一遍)"
    r".{0,80}(?:测试|pytest|test|命令|脚本|程序|项目|代码)"
    r"|(?:把|将).{0,24}(?:测试|pytest|test|命令|脚本)"
    r".{0,12}(?:跑一遍|跑一下|运行|执行)",
    re.IGNORECASE,
)
_CHINESE_ACTION_RULES = (
    (_CHINESE_READ_ACTION, Intent.READ_LOCAL),
    (_CHINESE_WEB_SEARCH_ACTION, Intent.RESEARCH_WEB),
    (_CHINESE_EXECUTE_ACTION, Intent.EXECUTE_LOCAL),
)


CHINESE_ABSTENTION_REASON = "zh_abstain_no_classifier"
_CHINESE_ABSTENTION_EXACT = frozenset(
    {
        "只回复两个字：成功",
        "检查hermes是否已升级",
        "评估 Hermes tool router这个插件",
        "Hermes内置了调用 codex的能力吗",
        "帮我看看这个",
        "什么是 glm-5.3 的最新跑分",
    }
)
_CHINESE_ABSTENTION_NEGATION_PREFIX = re.compile(
    r"(?:^|[，,。！？!?；;]\s*)(?:请\s*)?(?:不要|别|无需|不必)\s*"
)
_CHINESE_ABSTENTION_REPORT_PREFIX = re.compile(
    r"(?:^|[，,。！？!?；;]\s*)(?:这句话是|他说|日志显示|文档写着)\s*[:：]\s*"
)
_CHINESE_ABSTENTION_QUOTED_CONTENT = re.compile(
    r"“(?P<curly>[^”]*)”|\"(?P<double>[^\"]*)\"|"
    r"'(?P<single>[^']*)'|`(?P<backtick>[^`]*)`"
)


def _chinese_action_material(text: str) -> bool:
    """Return whether text contains one of the bounded Chinese action forms."""
    return any(pattern.search(text) for pattern, _intent in _CHINESE_ACTION_RULES)


def _chinese_framed_action(
    text: str,
    frame: re.Pattern[str],
) -> bool:
    """Check action material in a negated/reported clause only."""
    for match in frame.finditer(text):
        remainder = text[match.end() :]
        boundary = re.search(r"[。！？!?；;]", remainder)
        clause = remainder if boundary is None else remainder[: boundary.start()]
        if _chinese_action_material(clause):
            return True
    return False


def chinese_abstention_reason(message: str) -> str | None:
    """Return the internal no-classifier reason for bounded Chinese frames."""
    text = (message or "").strip()
    if text in _CHINESE_ABSTENTION_EXACT:
        return CHINESE_ABSTENTION_REASON
    if _chinese_framed_action(text, _CHINESE_ABSTENTION_NEGATION_PREFIX):
        return CHINESE_ABSTENTION_REASON
    if _chinese_framed_action(text, _CHINESE_ABSTENTION_REPORT_PREFIX):
        return CHINESE_ABSTENTION_REASON
    for match in _CHINESE_ABSTENTION_QUOTED_CONTENT.finditer(text):
        content = next((value for value in match.groups() if value is not None), "")
        if _chinese_action_material(content):
            return CHINESE_ABSTENTION_REASON
    return None


def _english_imperative_material_precedes_sentence_final(text: str) -> bool:
    """Return whether an English action lead has a substantive clause before the suffix."""
    lead_match = _ENGLISH_IMPERATIVE_LEAD.match(text)
    question_match = _CHINESE_SENTENCE_FINAL_CONCEPTUAL.search(text)
    if not lead_match or not question_match:
        return False

    between = text[lead_match.end() : question_match.start()]
    return bool(re.search(r"[^\W_]", between, re.UNICODE))


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
    abstention_reason = chinese_abstention_reason(text)
    if abstention_reason is not None:
        return IntentResult(
            frozenset({Intent.FULL_SURFACE}),
            0.0,
            abstention_reason,
        )
    lower = text.lower()
    if not text:
        return IntentResult(frozenset({Intent.FULL_SURFACE}), 0.0, "empty")

    # Checked before conceptual phrasing: "what are your capabilities?" is a
    # capability question, not general knowledge.  Anything that also carries
    # a task is left to the ordinary rules.
    if _is_capability_inquiry(text):
        return IntentResult(frozenset({Intent.CAPABILITY_INQUIRY}), 0.97, "capability_inquiry")

    if _CONCEPTUAL.search(text) and not _CONCEPTUAL_FRESHNESS_OR_URL.search(lower):
        return IntentResult(frozenset({Intent.ANSWER_ONLY}), 0.99, "conceptual")
    if _CHINESE_LEADING_CONCEPTUAL.search(text) and not _CONCEPTUAL_FRESHNESS_OR_URL.search(lower):
        return IntentResult(frozenset({Intent.ANSWER_ONLY}), 0.99, "conceptual")
    if (
        _CHINESE_SENTENCE_FINAL_CONCEPTUAL.search(text)
        and not _CONCEPTUAL_FRESHNESS_OR_URL.search(lower)
        and not _CHINESE_EXPLICIT_REQUEST_LEAD.search(text)
        and not _CHINESE_FILESYSTEM_PATH.search(text)
        and not _english_imperative_material_precedes_sentence_final(text)
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

    chinese_intents = {
        intent for pattern, intent in _CHINESE_ACTION_RULES if pattern.search(text)
    }

    def _merge(found, confidence, reason):
        if chinese_intents:
            return IntentResult(frozenset(set(found) | chinese_intents), 0.95, "deterministic_actions_zh")
        return IntentResult(frozenset(found), confidence, reason)

    if re.search(
        r"\b(computer[- _]?use|desktop control|control (?:the )?(?:desktop|computer)|"
        r"capture (?:the )?(?:safari|chrome|browser|desktop|screen|window)|"
        r"(?:safari|chrome) window|desktop window)\b",
        lower,
    ):
        return _merge({Intent.CONTROL_DESKTOP}, 0.99, "desktop_control")

    if re.search(r"\b(screenshot|photo|picture|image|diagram)\b", lower):
        if re.search(r"\b(generate|create|draw|design|make)\b", lower):
            return _merge({Intent.GENERATE_IMAGE}, 0.98, "generate_image")
        if re.search(r"\b(review|inspect|analy[sz]e|describe|read|look at)\b", lower):
            return _merge({Intent.ANALYZE_IMAGE}, 0.98, "analyze_image")

    if re.search(r"\b(video|movie|clip|animation)\b", lower):
        if re.search(r"\b(generate|create|make|animate)\b", lower):
            return _merge({Intent.GENERATE_VIDEO}, 0.96, "generate_video")
        if re.search(r"\b(analy[sz]e|review|describe|inspect)\b", lower):
            return _merge({Intent.ANALYZE_VIDEO}, 0.96, "analyze_video")

    if _URL.search(text) and _BROWSER_ACTION.search(text):
        return _merge({Intent.INTERACT_BROWSER}, 0.99, "interactive_url")

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

    if intents or chinese_intents:
        return _merge(intents, 0.95, "deterministic_actions")

    if (
        _CHINESE_RESIDUAL_CONCEPTUAL.search(text)
        and not _CONCEPTUAL_FRESHNESS_OR_URL.search(lower)
        and not _ENGLISH_ACTION_SIGNAL.search(lower)
    ):
        return IntentResult(frozenset({Intent.ANSWER_ONLY}), 0.99, "conceptual")

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
