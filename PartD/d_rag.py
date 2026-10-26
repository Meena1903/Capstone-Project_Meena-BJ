"""Shared policy retrieval helpers used by the independently runnable D1-D4 scripts and D5 agent."""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve paths relative to this script's location so the data/ directory and
# the repo-root mock_llm module are found regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUTPUTS_DIR = os.path.join(_REPO_ROOT, "outputs")
sys.path.insert(0, _REPO_ROOT)
pd.set_option("display.width", 140)

import mock_llm

POLICY_DIR = os.path.join(DATA_DIR, "policies")
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for Instamart support. Answer the QUESTION using only the supplied CONTEXT. "
                 "If the context does not contain the answer, say that you could not find the information.")


# ---------------- D1. chunking ----------------
def chunk_words(text, size=40, overlap=10):
    words = text.split()
    step, chunks, start = size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches or passes the last word
            break
        start += step
    return chunks


def load_chunks(size=40, overlap=10):
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        name = os.path.basename(path)
        for c in chunk_words(open(path, encoding="utf-8").read(), size, overlap):
            out.append({"source": name, "text": c})
    return out


# ---------------- D2. TF-IDF retrieval ----------------
class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])   # fit on chunk texts only

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])                                          # same fitted vectorizer
        sims = cosine_similarity(q, self.matrix).ravel()
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]


_default = None
def retrieve(query, k=3):
    global _default
    if _default is None:
        _default = Retriever()
    return _default.retrieve(query, k)


# ---------------- D3. augment and answer ----------------
def answer(question, retriever=None, k=3):
    hits = (retriever.retrieve(question, k) if retriever else retrieve(question, k))
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]


# ---------------- D4. evaluation ----------------
def evaluate(retriever, label):
    ev = pd.read_csv(os.path.join(DATA_DIR, "raw", "policy_eval.csv"), keep_default_na=False)
    rows = []
    for _, r in ev.iterrows():
        text, sources = answer(r["question"], retriever)
        if r["expected_doc"] == "NONE":
            rows.append({"q": r["question"], "kind": "unanswerable", "doc_ok": None, "fact_ok": None,
                         "fallback_ok": text.startswith("I could not find"), "answer": text})
        else:
            in_chunks = any(r["expected_fact"] in c["text"] for c in retriever.chunks)
            rows.append({"q": r["question"], "kind": "answerable", "doc_ok": r["expected_doc"] in sources,
                         "fact_ok": r["expected_fact"] in text, "fallback_ok": None,
                         "fact_in_any_chunk": in_chunks, "answer": text})
    df = pd.DataFrame(rows)
    a, u = df[df.kind == "answerable"].copy(), df[df.kind == "unanswerable"].copy()
    a[["doc_ok", "fact_ok", "fact_in_any_chunk"]] = a[["doc_ok", "fact_ok", "fact_in_any_chunk"]].astype(bool)
    u["fallback_ok"] = u["fallback_ok"].astype(bool)
    print(f"\n=== {label} ===")
    print(pd.DataFrame({"check": ["answerable: expected_doc in top 3", "answerable: answer contains expected_fact",
                                  "unanswerable: fallback 'I could not find'"],
                        "result": [f"{int(a.doc_ok.sum())}/{len(a)}", f"{int(a.fact_ok.sum())}/{len(a)}",
                                   f"{int(u.fallback_ok.sum())}/{len(u)}"]}).to_string(index=False))
    broken = a[~(a.doc_ok & a.fact_ok)]
    for _, b in broken.iterrows():
        why = ("the expected fact is split across a chunk boundary (no single chunk contains it)"
               if not b.fact_in_any_chunk else "the sentence that carries the fact is cut by a chunk boundary, so the answerer only sees a fragment of it")
        print(f"  BREAKS: {b.q!r} -> doc_ok={b.doc_ok}, fact_ok={b.fact_ok}; reason: {why}")
        print(f"          answer given: {b.answer!r}")
    return df
