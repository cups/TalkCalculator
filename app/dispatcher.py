# Takes model output JSON → calls calculator methods.

import json
from app.calculator_interface import call_tool


def dispatch(model_output: str):
    """
    Accepts:
      - single call: {"name":"add","arguments":{"number":27}}
      - dict with tool_calls: {"tool_calls":[{...}, {...}]}
      - a JSON array: [{...}, {...}]
    """
    payload = json.loads(model_output)
    # Normalize to a list of calls
    if isinstance(payload, dict):
        calls = payload.get("tool_calls") or ([payload] if "name" in payload else [])
    elif isinstance(payload, list):
        calls = payload
    else:
        raise ValueError("Unsupported model output")
    results = []
    for call in calls:
        results.append(call_tool(call["name"], call.get("arguments", {})))
    # return the most recent (useful for get_total-like calls)
    return results[-1] if results else None
