def test_make_add_example():
    from data.generate_dataset import make_add_example
    ex = make_add_example()
    assert "user" in ex and "tool_call" in ex
    assert ex["tool_call"]["name"] == "add"
