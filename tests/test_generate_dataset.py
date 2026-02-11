def test_make_add_example():
    from data.generate_dataset import make_add_example
    ex = make_add_example()
    assert "user" in ex and "tool_calls" in ex
    assert isinstance(ex["tool_calls"], list)
    assert ex["tool_calls"][0]["name"] == "add"
