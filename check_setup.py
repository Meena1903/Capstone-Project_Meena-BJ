"""check_setup.py -- run after `python generate_data.py` to confirm your environment is ready for Part A."""
import hashlib, os, sys

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

problems = []
if sys.version_info < (3, 11):
    problems.append(f"Python 3.11+ needed, found {sys.version.split()[0]}")
try:
    import sqlite3
    v = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
    if v < (3, 25, 0):
        problems.append(f"SQLite 3.25+ needed, found {sqlite3.sqlite_version}")
except Exception as e:
    problems.append(f"sqlite3 problem: {e}")
for lib in ("pandas", "numpy", "sklearn", "joblib"):
    try:
        __import__(lib)
    except ImportError:
        problems.append(f"missing library: {lib}")
for f in ("generate_data.py", "mock_llm.py", ".gitignore", "requirements.txt"):
    if not os.path.exists(f):
        problems.append(f"missing file: {f}")

if not problems:
    import pandas as pd
    o = pd.read_csv("data/raw/orders.csv")
    c = pd.read_csv("data/raw/customers.csv")
    t = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    checks = {"orders rows == 13084": len(o) == 13084,
              "customers rows == 1200": len(c) == 1200,
              "tickets rows == 164": len(t) == 164,
              "policy files == 5": len([x for x in os.listdir("data/policies") if x.endswith(".txt")]) == 5}
    for k, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + k)
        if not ok:
            problems.append(k)
print("SETUP OK" if not problems else "SETUP PROBLEMS:\n - " + "\n - ".join(problems))
