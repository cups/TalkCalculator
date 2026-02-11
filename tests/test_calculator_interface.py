from decimal import Decimal
import pytest
from app.calculator_interface import call_tool


def setup_function():
    # ensure calculator is reset before each test
    call_tool("clear_all", {})


def test_add_and_get_total():
    call_tool("clear_all", {})
    call_tool("add", {"number": 7})
    call_tool("add", {"number": 9})
    res = call_tool("get_total", {})
    assert float(res) == 16.0


def test_add_numbers_list():
    call_tool("clear_all", {})
    # Call add twice with single 'number' arg to simulate multiple inputs
    call_tool("add", {"number": 2})
    call_tool("add", {"number": 3})
    res = call_tool("get_total", {})
    assert float(res) == 5.0


def test_unknown_tool_raises():
    call_tool("clear_all", {})
    with pytest.raises(ValueError):
        call_tool("unknown", {})
