import examples

@examples.is_irony.test_prompt_examples
def test_correct_function_name(function_name, call_args, call_kwargs, return_value):
    assert function_name == "is_irony"

@examples.is_irony.test_prompt_examples
def test_correct_return_type(function_name, call_args, call_kwargs, return_value):
    assert isinstance(return_value, bool)
