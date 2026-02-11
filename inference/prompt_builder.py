# Builds structured prompts for function calling.

SYSTEM_PROMPT = """You are a system that converts calculator requests into JSON function calls.

Available functions:

add(number)
get_total()
clear_all()

Respond ONLY with JSON.
"""


def build_prompt(user_input: str):
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
