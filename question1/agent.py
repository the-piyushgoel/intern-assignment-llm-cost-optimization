from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import all_articles_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# --- Cost notes -------------------------------------------------------------
# Output tokens cost 5x input tokens, so the biggest wins were:
#   1. No chain-of-thought / "quote the article" / "double-check" instructions
#      (the model answered with a ~480-token narrative nobody reads).
#   2. A short, STATIC system prompt: no per-run uuid, no few-shot examples
#      (the examples duplicated what the help-center articles already say).
#   3. Compact order snapshot: only fields the routing policy depends on,
#      minified JSON instead of indent=2.
# The static part comes first and the per-ticket part last, so the prefix is
# cache-friendly if the help center ever grows past the 1024-token threshold.

_SYSTEM = (
    "You are ShopAssist, a Tier-1 support copilot. Pick one action for the ticket "
    "using the help center and order facts.\n"
    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.\n"
    "kb_article must be one article id from the help center.\n"
    'Reply with JSON only: {"action":...,"kb_article":...,"customer_reply":<max 15 words>}.\n'
    "HELP CENTER:\n"
)

# Fields the routing policy depends on. Keep this list in sync with the policy
# articles: dropping a field silently changes decisions (e.g. subscriptions need
# subscription_age_days). Anything missing from the order is simply omitted.
_ORDER_FIELDS = (
    "product_type",
    "delivered_days_ago",
    "subscription_age_days",
    "opened",
    "license_revealed",
    "customized",
    "damage_reported",
)


def compact_order(order: dict) -> str:
    slim = {k: order[k] for k in _ORDER_FIELDS if k in order}
    return json.dumps(slim, separators=(",", ":"))


class SupportCopilot:
    """ShopAssist Tier-1 copilot: one cheap LLM call per ticket."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        system = _SYSTEM + all_articles_blob()
        user = f"THREAD:\n{thread}\nORDER_SNAPSHOT: {compact_order(order)} END_SNAPSHOT"
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
