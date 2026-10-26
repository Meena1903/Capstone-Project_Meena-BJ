"""Shared ticket guardrail, redaction, retry and classification helpers."""
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES, BASE_DELAY = 3, 0.05
TOKEN_BUDGET = 6000
ALLOWED = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
             'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
            'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.\n'
            'EXAMPLES:\nTicket: I was charged twice and want my money back.\n{"category": "billing", "reason": "double charge"}\n'
            'Ticket: My rider never arrived with the order.\n{"category": "delivery", "reason": "order not received"}\n'
            'Ticket: I cannot open the app, it shows an error.\n{"category": "technical", "reason": "app problem"}\n'
            'Ticket: Someone hacked my account and made a payment I did not authorise.\n'
            '{"category": "fraud", "reason": "unauthorised access"}')
random.seed(42)


def inputs():
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv").set_index("customer_id")
    return tickets, segments


def redact(text):
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def validate(reply):
    try:
        result = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply or "").lower()))
        found = [category for category in ALLOWED if category in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(result, dict) and result.get("category") in ALLOWED:
        return "classified_json", result["category"]
    return "flagged_invalid_output", ""


class PermanentLLMError(Exception):
    pass


def call_with_retry(ticket_id, messages, label, stats, verbose=False):
    failures = 0
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            stats["recovered"] += failures
            return response
        except TransientLLMError:
            failures += 1
            stats["transient"] += 1
            if attempt == MAX_RETRIES:
                raise PermanentLLMError(ticket_id)
            delay = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            stats["log"].append((label, ticket_id, attempt, delay))
            if verbose:
                print(f"RETRY ticket={ticket_id} retry={attempt} wait={delay:.4f}s")
            time.sleep(delay)


def run_pipeline(label, prompt, token_budget=None, verbose=False):
    tickets, segments = inputs()
    redacted = {row.ticket_id: redact(row.text)[0] for row in tickets.itertuples()}
    mock_llm.reset()
    stats = {"transient": 0, "recovered": 0, "tokens": 0, "calls_ok": 0, "verbose": verbose, "log": []}
    rows, sent = [], []
    for row in tickets.itertuples():
        tid, ticket_text = row.ticket_id, row.text
        status, category, used = None, "", 0
        if ticket_text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(ticket_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and stats["tokens"] >= token_budget:
            status = "budget_exhausted"
        else:
            user_text = redacted[tid]
            assert not any(pattern.search(user_text) for _, pattern, _ in PATTERNS)
            sent.append(tid)
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}]
            try:
                response = call_with_retry(tid, messages, label, stats, verbose)
                used = response.usage.total_tokens
                stats["tokens"] += used
                stats["calls_ok"] += 1
                status, category = validate(response.choices[0].message.content)
            except PermanentLLMError:
                status = "failed_permanent"
        if status is None:
            status = "failed_permanent"
        premium = bool(segments.loc[row.customer_id, "is_premium"]) if row.customer_id in segments.index else False
        if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "failed_permanent"}:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        rows.append({"ticket_id": tid, "customer_id": row.customer_id, "status": status,
                     "category": category, "segment_name": segments.loc[row.customer_id, "segment_name"]
                     if row.customer_id in segments.index else "", "priority": priority, "route": route,
                     "tokens_used": used, "true_category": row.true_category})
    frame = pd.DataFrame(rows)
    assert len(frame) == len(tickets)
    assert not (set(frame.loc[frame.status.isin(["skipped_empty", "flagged_long"]), "ticket_id"]) & set(sent))
    return frame, stats


def summarize(label, frame, stats):
    accepted = frame[frame.status.isin(["classified_json", "classified_keyword"])]
    correct = int((accepted.category == accepted.true_category).sum())
    accuracy = correct / len(accepted) if len(accepted) else 0
    print(f"{label}: counts={frame.status.value_counts().to_dict()}")
    print(f"classified={len(accepted)} accuracy={accuracy * 100:.1f}% ({correct}/{len(accepted)}) "
          f"tokens={stats['tokens']} tokens/classified={stats['tokens'] / max(len(accepted), 1):.1f}")
    print(f"transient={stats['transient']} recovered={stats['recovered']}")
    return accepted, accuracy


def write_retry_lines(path, groups, mode="w"):
    with open(path, mode, encoding="utf-8") as output:
        for stats in groups:
            for label, tid, attempt, delay in stats["log"]:
                output.write(f"{label}\tticket={tid}\tretry={attempt}\twait={delay:.4f}s\n")
