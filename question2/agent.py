from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import compact_order, parse_resolution

# --- Cost notes -------------------------------------------------------------
# Baseline: 4 LLM calls, two of them stuffed with 80-120k characters of help
# center, ~480 output tokens each -> ~73k points per ticket.
#
# Now only ONE node needs a language model. The other three are deterministic:
#   triage   -> rules: turn ticket text + order facts into search terms (no LLM)
#   retrieve -> lexical (BM25-style) index over the help center, top-k (no LLM)
#   compose  -> the single LLM call, sees only the k retrieved articles
#   audit    -> schema / allow-list / safety checks in code (no LLM)
# The help-center index is built once and never enters a prompt in full.

ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
SAFE_FALLBACK_ACTION = "ESCALATE_TIER2"
TOP_K = 3

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from had has have i in is it me my of on or "
    "our please that the this to was were with you your can get ago".split()
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in _tokens(text)]


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass
class Triage:
    query_terms: list[str]


class HelpCenterIndex:
    """Tiny BM25 index. In production this is built offline / by a vector DB."""

    def __init__(self, article_ids_: list[str]):
        self.docs: dict[str, Counter] = {}
        for aid in article_ids_:
            # The id/title is highly informative (e.g. "custom-items-no-replace").
            text = f"{aid.replace('-', ' ')} {aid.replace('-', ' ')} {load_article(aid)}"
            self.docs[aid] = Counter(_terms(text))
        self.avg_len = sum(sum(c.values()) for c in self.docs.values()) / max(len(self.docs), 1)
        df: Counter = Counter()
        for c in self.docs.values():
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, terms: list[str], k: int) -> list[str]:
        k1, b = 1.5, 0.75
        scores: dict[str, float] = {}
        for aid, tf in self.docs.items():
            dl = sum(tf.values())
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avg_len))
            scores[aid] = s
        ranked = sorted(scores, key=lambda a: (-scores[a], a))
        return ranked[:k]


_SYSTEM = (
    "You are ShopAssist, a Tier-1 support copilot. Pick one action using ONLY the "
    "articles and order facts below.\n"
    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.\n"
    "kb_article must be one of the article ids shown.\n"
    'Reply with JSON only: {"action":...,"kb_article":...,"customer_reply":<max 15 words>}.'
)


class SupportPipeline:
    """triage -> retrieve -> compose -> audit, with a single LLM call (compose)."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.index = HelpCenterIndex(article_ids())

    # -- orchestration -------------------------------------------------------
    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        triage = self._triage(thread, order)
        article_ids_ = self._retrieve(triage)
        draft = self._compose(thread, order, article_ids_)
        return self._audit(draft, article_ids_)

    # -- nodes ---------------------------------------------------------------
    def _triage(self, thread: str, order: dict) -> Triage:
        """Deterministic: customer words + order facts -> retrieval terms."""
        customer_text = " ".join(m for m in thread.splitlines() if m.startswith("[CUSTOMER]"))
        extra: list[str] = []
        ptype = order.get("product_type")
        if ptype:
            extra.append(str(ptype))
        if ptype == "subscription":
            extra += ["subscription", "cancel", "annual"]
        if order.get("license_revealed"):
            extra += ["digital", "license", "refundable"]
        if order.get("customized"):
            extra += ["custom", "personalized", "replacement"]
        if order.get("damage_reported"):
            extra += ["damaged", "replacement"]
        if ptype == "physical" and not order.get("opened"):
            extra += ["physical", "refund", "return"]
        return Triage(_terms(customer_text) + _terms(" ".join(extra)))

    def _retrieve(self, triage: Triage) -> list[str]:
        """Deterministic: top-k help-center articles (ids only)."""
        return self.index.search(triage.query_terms, TOP_K)

    def _compose(self, thread: str, order: dict, article_ids_: list[str]) -> str:
        """The only LLM call: static instructions, then per-ticket content."""
        articles = "\n".join(f"--- {a} ---\n{load_article(a)}" for a in article_ids_)
        messages = [
            ChatMessage("system", _SYSTEM),
            ChatMessage(
                "user",
                f"ARTICLES:\n{articles}\n\nTHREAD:\n{thread}\n"
                f"ORDER_SNAPSHOT: {compact_order(order)} END_SNAPSHOT",
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(self, draft: str, retrieved: list[str]) -> TicketResolution:
        """Deterministic guardrails instead of a second LLM pass."""
        try:
            parsed = parse_resolution(draft)
        except (ValueError, KeyError, json.JSONDecodeError):
            return TicketResolution(SAFE_FALLBACK_ACTION, retrieved[0] if retrieved else "")
        action, article = parsed.action, parsed.kb_article
        valid_article = article in set(article_ids())
        if action not in ACTIONS or not valid_article:
            # Unknown action or hallucinated article id -> hand to a human.
            return TicketResolution(SAFE_FALLBACK_ACTION, retrieved[0] if retrieved else article)
        return TicketResolution(action, article)
