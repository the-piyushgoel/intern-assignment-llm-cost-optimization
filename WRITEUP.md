# Write-up — ShopAssist cost optimization

Billing: `points = uncached_in × 3 + cached_in × 0.3 + out × 15`. Output is 5× the price of input, so I attacked output first, then input.

## Results (public tickets T-1001..T-1004, tiktoken cl100k_base)

| | Baseline / ticket | Now / ticket | Target |
|---|---:|---:|---:|
| Q1 | ~8,915 (35,658 / 4) | **~1,680** | ≤ 3,000 |
| Q2 | ~73,500 (T-1001) | **~1,430** | ≤ 2,500 |

Holdout `T-1091` (subscription, day 38 → SEND_ARTICLE) still passes on both tracks. `pytest -q` → 7 passed.

## Q1 — what I changed
1. **Removed the verbose-output triggers.** The prompt said "step by step", "full reasoning", "quote any article". Each call produced a ~480-token narrative (480 × 15 = 7,200 pts, ~80% of cost). Output is now ~40 tokens.
2. **Static, short system prompt.** Dropped the per-run `uuid` (it also breaks prefix caching), and the few-shot block (it repeated the articles and leaked answers).
3. **Compact order snapshot.** Whitelisted only the fields the policy uses, minified JSON instead of `indent=2`, dropped `order_id`/`sku`/`title`.
4. Static content first, per-ticket content last (cache-friendly layout).

### Per-call breakdown, T-1001 (Q1)
| input tok | output tok | input pts | output pts | total |
|---:|---:|---:|---:|---:|
| 372 | 46 | 1,116 | 690 | **1,806** |

## Q2 — what I changed
Only **one** node needs an LLM; the other three are deterministic code.
- **triage** → rules turn customer text + order facts into search terms.
- **retrieve** → small BM25 index over the help center, top-3 article ids. The 210k-token help center never goes into a prompt.
- **compose** → the single LLM call, sees only 3 articles + thread + compact snapshot.
- **audit** → code checks: parseable JSON, action in allowed set, `kb_article` exists; otherwise fall back to `ESCALATE_TIER2` (safe default). Replaces a 6.7k-token LLM re-check.

### Per-call breakdown
| ticket | path | node | input | output | in pts | out pts | total |
|---|---|---|---:|---:|---:|---:|---:|
| T-1001 | physical refund | triage/retrieve/audit | 0 | 0 | 0 | 0 | 0 |
| T-1001 | | compose | 296 | 46 | 888 | 690 | **1,578** |
| T-1003 | escalation | triage/retrieve/audit | 0 | 0 | 0 | 0 | 0 |
| T-1003 | | compose | 262 | 39 | 786 | 585 | **1,371** |

(Baseline T-1001: triage 32,670 + retrieve 2,475 + compose 10,908 + audit 27,477 = 73,530.)

## Trade-offs / honest caveats
- Lexical retrieval is cheap but weaker than embeddings on paraphrased tickets; top-3 + order-derived terms covers the policy set here, but with a real 400-article center I would add embeddings or a reranker and measure recall@k.
- The deterministic audit checks format and validity, not policy correctness. If a second opinion is needed, run an LLM audit only on low-confidence or high-value (refund) cases.
- I did not pad prompts to reach the 1,024-token cache threshold: that would be gaming the simulator (real APIs charge cache writes and the first call). The layout is cache-ready if the static prefix grows.
- `LlmClient` is a simulator whose decision comes from `ORDER_SNAPSHOT`; with a real model, I'd add an eval set (incl. holdouts like T-1091) to confirm accuracy didn't regress after trimming the prompt.
- Tests `test_baseline_cost_is_high` were inverted into `test_cost_under_target`.
