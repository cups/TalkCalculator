# Takes model output JSON → calls calculator methods.

import json
from app.calculator_interface import call_tool


def dispatch(model_output: str):
    """
    Expects: {"name":"add","arguments":{"number":27}}
    """
    tool_call = json.loads(model_output)
    result = call_tool(tool_call["name"], tool_call.get("arguments", {}))
    return result
