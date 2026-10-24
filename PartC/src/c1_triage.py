"""C1-C7 - Support-ticket triage with guardrails, redaction, retries, validation, budget and routing.
Run AFTER b1_segments.py, from repo root:  python c1_triage.py"""
import json, os, random, re, sys, time
import pandas as pd

# Resolve paths relative to this script's location so the data/ and outputs/
# directories, and the repo-root mock_llm module, are found regardless of the
# current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "raw")
OUTPUTS_DIR = os.path.join(_REPO_ROOT, "outputs")
sys.path.insert(0, _REPO_ROOT)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
pd.set_option("display.width", 140)

import mock_llm
from mock_llm import TransientLLMError, est_tokens

# ---------------- settings ----------------
MAX_INPUT_TOKENS = 400
MAX_RETRIES, BASE_DELAY = 3, 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),      # order matters: email, phone, order id
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
random.seed(42)

# ---------------- C3. the two prompts ----------------
ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
             'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
            'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.\n'
            'EXAMPLES:\n'
            'Ticket: I was charged twice and want my money back.\n{"category": "billing", "reason": "double charge"}\n'
            'Ticket: My rider never arrived with the order.\n{"category": "delivery", "reason": "order not received"}\n'
            'Ticket: I cannot open the app, it shows an error.\n{"category": "technical", "reason": "app problem"}\n'
            'Ticket: Someone hacked my account and made a payment I did not authorise.\n{"category": "fraud", "reason": "unauthorised access"}')
for p in (ZERO_SHOT, FEW_SHOT):
    assert "CLASSIFY" in p and all(c in p for c in ALLOWED) and '{"category": "...", "reason": "..."}' in p
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
print(f"Prompt length (est. tokens): zero-shot {est_tokens(ZERO_SHOT)}, few-shot {est_tokens(FEW_SHOT)}")

# ---------------- data ----------------
tickets = pd.read_csv(os.path.join(DATA_DIR, "tickets.csv"), keep_default_na=False)
seg = pd.read_csv(os.path.join(OUTPUTS_DIR, "customer_segments.csv")).set_index("customer_id")

# ---------------- C2. redaction ----------------
def redact(text):
    counts = {}
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[name] = n
    return text, counts

red_counts = {n: 0 for n, _, _ in PATTERNS}
redacted = {}
for _, r in tickets.iterrows():
    redacted[r["ticket_id"]], c = redact(r["text"])
    for k, v in c.items():
        red_counts[k] += v
print("\n=== C2. Redaction over all 164 tickets ===")
print(red_counts, "total", sum(red_counts.values()), "(ref 55 / 91 / 132 / 278)")
left = sum(len(p.findall(t)) for t in redacted.values() for _, p, _ in PATTERNS)
print("PII matches left after re-applying the 3 patterns to the redacted texts:", left, "(ref 0)")
assert left == 0

# ---------------- C4. resilient call ----------------
class PermanentLLMError(Exception):
    pass

def call_with_retry(ticket_id, messages, stats, log):
    failures = 0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = mock_llm.chat(messages)
            stats["recovered"] += failures
            return resp
        except TransientLLMError:
            failures += 1; stats["transient"] += 1
            if attempt == MAX_RETRIES:
                raise PermanentLLMError(ticket_id)
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            log.append((ticket_id, attempt, wait))
            print(f"  RETRY ticket={ticket_id} retry={attempt} wait={wait:.4f}s") if stats["verbose"] else None
            time.sleep(wait)

# ---------------- C5. validation ----------------
def validate(reply):
    """Return (status, category). Never guesses."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply or "").lower()))
        found = [c for c in ALLOWED if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(data, dict) and data.get("category") in ALLOWED:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""                     # valid JSON, wrong category -> human review

# ---------------- C7b. routing ----------------
def route(status, category, customer_id):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "failed_permanent"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    premium = bool(seg.loc[customer_id, "is_premium"]) if customer_id in seg.index else False
    return ("premium_desk", "HIGH") if premium else ("support_queue", "NORMAL")

# ---------------- pipeline ----------------
all_retry_lines = []

def run_pipeline(label, system_prompt, token_budget=None, verbose=False):
    mock_llm.reset()                                         # immediately before each full run
    stats = {"transient": 0, "recovered": 0, "tokens": 0, "calls_ok": 0, "verbose": verbose}
    log, rows, sent = [], [], []
    for _, r in tickets.iterrows():
        tid, text = r["ticket_id"], r["text"]
        status, category, used = None, "", 0
        if text.strip() == "":                               # C1a
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:            # C1b
            status = "flagged_long"
        elif token_budget is not None and stats["tokens"] >= token_budget:   # C7a
            status = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": redacted[tid]}]               # user message = redacted text only
            assert msgs[1]["content"] == redacted[tid]
            assert not any(p.search(msgs[1]["content"]) for _, p, _ in PATTERNS), "PII reached the prompt!"
            sent.append(tid)
            try:
                resp = call_with_retry(tid, msgs, stats, log)
                used = resp.usage.total_tokens
                stats["tokens"] += used; stats["calls_ok"] += 1
                status, category = validate(resp.choices[0].message.content)
            except PermanentLLMError:
                status = "failed_permanent"
        rt, pr = route(status, category, r["customer_id"])
        rows.append({"ticket_id": tid, "customer_id": r["customer_id"], "status": status, "category": category,
                     "segment_name": seg.loc[r["customer_id"], "segment_name"] if r["customer_id"] in seg.index else "",
                     "priority": pr, "route": rt, "tokens_used": used, "true_category": r["true_category"]})
    df = pd.DataFrame(rows)
    # guardrail proof: nothing empty/long was ever sent to the model
    bad = set(df.loc[df.status.isin(["skipped_empty", "flagged_long"]), "ticket_id"])
    assert not (bad & set(sent)), "empty/long ticket reached the model"
    all_retry_lines.extend((label, *x) for x in log)
    for _, attempt, wait in log:                             # jitter range check
        lo, hi = BASE_DELAY * 2 ** attempt * 0.9, BASE_DELAY * 2 ** attempt * 1.15
        assert lo <= wait <= hi
    stats["log"] = log
    return df, stats

def summarise(label, df, stats):
    cls = df[df.status.isin(["classified_json", "classified_keyword"])]
    acc = (cls.category == cls.true_category).mean() if len(cls) else float("nan")
    print(f"\n--- {label} ---")
    print("status counts:", df.status.value_counts().to_dict())
    print(f"TransientLLMErrors: {stats['transient']} | recovered by retry: {stats['recovered']} | retry lines: {len(stats['log'])}")
    print(f"classified: {len(cls)} | accuracy: {acc*100:.1f}% ({int((cls.category == cls.true_category).sum())} of {len(cls)}) | "
          f"total tokens: {stats['tokens']} | tokens per classified ticket: {stats['tokens']/max(len(cls),1):.1f}")
    return cls, acc

# ---------------- C6. zero-shot vs few-shot ----------------
print("\n=== C4/C6. Full runs (no token budget) ===")
print("(retry log of the zero-shot run)")
zs_df, zs_stats = run_pipeline("zero-shot", ZERO_SHOT, verbose=True)
zs_cls, zs_acc = summarise("ZERO-SHOT", zs_df, zs_stats)
fs_df, fs_stats = run_pipeline("few-shot", FEW_SHOT)
fs_cls, fs_acc = summarise("FEW-SHOT", fs_df, fs_stats)
print("(refs: 135 json / 12 keyword / 9 invalid; 19 transient errors all recovered; 147 classified; zero-shot 85.7%, few-shot 100%)")
print("Eligible tickets sent to the model:", len(zs_df[~zs_df.status.isin(['skipped_empty', 'flagged_long'])]),
      "| empty:", int((zs_df.status == 'skipped_empty').sum()), "| long:", int((zs_df.status == 'flagged_long').sum()))

print("\nConfusion table, zero-shot (rows = true, columns = predicted):")
print(pd.crosstab(zs_cls.true_category, zs_cls.category))
print("\nComparison:")
print(pd.DataFrame({"run": ["zero-shot", "few-shot"],
                    "classified": [len(zs_cls), len(fs_cls)],
                    "accuracy_%": [round(zs_acc * 100, 1), round(fs_acc * 100, 1)],
                    "total_tokens": [zs_stats["tokens"], fs_stats["tokens"]],
                    "tokens_per_classified": [round(zs_stats["tokens"] / len(zs_cls), 1), round(fs_stats["tokens"] / len(fs_cls), 1)]}).to_string(index=False))

# ---------------- C7a. budget demo ----------------
DEMO_BUDGET = 6000
demo_df, demo_stats = run_pipeline("budget-demo", ZERO_SHOT, token_budget=DEMO_BUDGET)
summarise(f"BUDGET DEMO (zero-shot, TOKEN_BUDGET={DEMO_BUDGET})", demo_df, demo_stats)
n_ex = int((demo_df.status == "budget_exhausted").sum())
print(f"budget_exhausted tickets: {n_ex} (must be >= 1)")
assert n_ex >= 1
COLS = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
demo_df[COLS].to_csv(os.path.join(OUTPUTS_DIR, "triage_budget_demo.csv"), index=False)

# ---------------- C7b. final results ----------------
final_df, final_stats = run_pipeline("final", FEW_SHOT, token_budget=100000)
summarise("FINAL (few-shot, TOKEN_BUDGET=100000)", final_df, final_stats)
final_df[COLS].to_csv(os.path.join(OUTPUTS_DIR, "triage_results.csv"), index=False)
print("\nFinal routing:")
print(final_df.groupby(["route", "priority"]).size().to_string())
print("rows in triage_results.csv:", len(final_df), "(must be 164)")
validated = final_df.status.isin(["classified_json", "classified_keyword"]).sum()
print(f"rule-validated (classified_json + classified_keyword): {validated} | sent to humans: {int((final_df.route == 'human_review').sum())}")
print(final_df[final_df.route == "human_review"].status.value_counts().to_dict())

with open(os.path.join(OUTPUTS_DIR, "retry_log.txt"), "w") as f:
    for label, tid, attempt, wait in all_retry_lines:
        f.write(f"{label}\tticket={tid}\tretry={attempt}\twait={wait:.4f}s\n")
print(f"\nretry log written to outputs/retry_log.txt ({len(all_retry_lines)} lines across all runs)")
