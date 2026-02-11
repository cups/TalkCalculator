# Wraps the existing calculator so the rest of the system doesn’t care about internals.

try:
    from calculator import Calculator  # your existing class
except Exception:
    Calculator = None

calculator = Calculator() if Calculator is not None else None


def call_tool(name, args):
    if calculator is None:
        raise RuntimeError("Calculator backend not available")
    if name == "add":
        # Expect a single argument named "number". Callers should invoke add
        # multiple times for multiple numbers rather than passing lists.
        if "number" not in args:
            raise ValueError("Missing required argument: 'number'")
        calculator.add(args["number"])
    elif name == "get_total":
        return calculator.get_total()
    elif name == "clear_all":
        calculator.clear_all()
    else:
        raise ValueError(f"Unknown tool: {name}")
