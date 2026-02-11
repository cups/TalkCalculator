# Builds structured prompts for function calling.

SYSTEM_PROMPT = """You are a system that converts calculator requests into JSON tool_calls.

Available functions (each call is an object with "name" and "arguments"):
  - {"name":"add","arguments":{"number":<numeric>}}
  - {"name":"get_total","arguments":{}}
  - {"name":"clear_all","arguments":{}}

Respond ONLY with a JSON object containing a single key "tool_calls" whose value
is an array of call objects. Example:

{"tool_calls":[{"name":"add","arguments":{"number":7}}, {"name":"add","arguments":{"number":9}}, {"name":"get_total","arguments":{}}]}
"""


def build_prompt(user_input: str):
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
