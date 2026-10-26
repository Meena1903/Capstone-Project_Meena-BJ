"""Shared tool, ReAct, safe-stop and checkpoint functions for D5-D9 scripts."""
import json, os, sqlite3, sys, time
from datetime import datetime

# Resolve paths relative to this script's location so data/, outputs/, state/
# and the repo-root mock_llm / d_rag modules are found regardless of the
# current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUTPUTS_DIR = os.path.join(_REPO_ROOT, "outputs")
STATE_DIR = os.path.join(_REPO_ROOT, "state")
sys.path.insert(0, _REPO_ROOT)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

import mock_llm
from d_rag import retrieve

DB, LEDGER, STATE_DIR = os.path.join(DATA_DIR, "instamart.db"), \
                        os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl"), \
                        STATE_DIR
RUNS, DEMOS = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl"), \
              os.path.join(OUTPUTS_DIR, "agent_demos.jsonl")
_ctx = {"run_id": None}                      # lets ledger lines name the run that wrote them


# ====================== D5. tools ======================
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    hits = retrieve(query, k=3)
    return {"results": [{"source": h["source"], "score": round(h["score"], 3), "text": h["text"]} for h in hits]}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}       # late: 100% of the order amount
    return {"refund_inr": 0.0}                                       # 45 minutes or less: nothing


def _ledger(tool, order_id, refund_inr):
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool, "run_id": _ctx["run_id"],
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")


def issue_refund(order_id, refund_inr):
    if refund_inr > 1000:                                            # safety guard: never auto-issue above INR 1000
        return {"error": "refund above INR 1000 must be escalated to a human"}
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database and return its status, amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the three most relevant passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes using the refund rules.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, calculate_refund, "
          "issue_refund and escalate_to_human. Finish with one line starting REFUND:, ESCALATE:, DENIED: or NOT_FOUND:.")


# ====================== D6-D8. ReAct loop, stops, checkpoints ======================
class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w") as f:
        json.dump(state, f, indent=1)


def _refund_of(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return s["observation"]["refund_inr"]
    return 0.0


def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": _refund_of(state["steps"])}


def _fallback(state, status, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    msg = (f"{status}: stopped because {reason}. Steps completed: {done}. "
           f"Please hand this case ({state['scenario_id']}) to a human agent with this summary.")
    return _result(state, status, msg)


def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    _ctx["run_id"] = state["run_id"]
    while True:
        if time.monotonic() - start >= time_limit_seconds:                        # (c) timeout, top of iteration
            return _fallback(state, "FALLBACK_TIMEOUT", f"the time limit of {time_limit_seconds}s was reached")
        if state["llm_calls"] >= max_iterations:                                  # (a) too many model calls
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"{max_iterations} model calls gave no final answer")
        resp = mock_llm.chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:                                    # (b) token limit, tool NOT executed
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"{state['tokens_used']} tokens exceeded the limit of {token_limit}")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                    # final answer
            answer = msg.content
            return _result(state, answer.split(":")[0].strip(), answer)
        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        observation = TOOL_MAP[tc.name](**args)                                   # look up by name, call with arguments
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": tc.name, "arguments": args, "observation": observation})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        save_checkpoint(state)                                                    # after EVERY tool observation
        if crash_after_tools and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": []}
    mock_llm.reset()
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json")) as f:
        state = json.load(f)
    return _loop(state, max_iterations, token_limit, time_limit_seconds)


def log_run(path, res):
    line = {"scenario_id": res["scenario_id"], "run_id": res["run_id"], "status": res["status"],
            "final_answer": res["final_answer"], "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
            "steps": [{"tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]} for s in res["steps"]]}
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
