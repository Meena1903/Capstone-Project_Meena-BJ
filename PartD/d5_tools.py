"""D5 - Show the five tool definitions and dispatch map."""
from d_agent import TOOL_MAP, TOOLS_SPEC

for spec in TOOLS_SPEC:
    print(spec["name"], "-", spec["description"], spec["parameters"])
print("TOOL_MAP:", sorted(TOOL_MAP))
