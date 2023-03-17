# Overview of the GPTRun Project

This document provides a high-level overview of the GPTRun project's structure, which is designed to make it easy for developers to understand and contribute to the project.

## Structure

The GPTRun project consists of two main components:

1. **LLM Runners (gptrun/llm)**: Responsible for managing the state of a connection to various Language Model (LLM) services and executing the models.
2. **Few-shot Functions (gptrun/function)**: Define functions using few-shot prompting techniques, which may include examples, custom code, or any other methods.

### LLM Runners

The LLM runners are organized into the following modules:

- **gptrun/llm/_abstract.py**: Contains the AbstractLLMRunner class, which serves as a base class for implementing different LLM runners.
- **gptrun/llm/openai.py**: Contains the ChatAPI and CompletionAPI classes for OpenAI's Chat and Completion APIs, respectively.
- **gptrun/llm/facebook.py**: Contains the LLaMAcpp class for Facebook's LLaMA C++ local model.
- **gptrun/llm/nlpcloud.py**: Contains the GPTJAPI class for the GPT-J API provided by NLP Cloud.

### Few-shot Functions

The few-shot functions are organized into the following modules:

- **gptrun/function/_abstract.py**: Contains the AbstractFewShotFunction class, which serves as a base class for implementing different types of few-shot functions.
- **gptrun/function/doctest_based.py**: Contains the DoctestBasedFewShotFunction class, which implements a few-shot function using doctests to provide examples.
- **gptrun/function/parameter_based.py**: Contains the ParameterBasedFewShotFunction class, which implements a few-shot function using dynamic, on-demand examples based on the function parameters.

## Testing

The project includes a `tests/` directory with a similar structure to the main `gptrun/` directory. This allows for easier organization and discovery of test cases related to each module.

### Test Structure

- **tests/llm/test_openai.py**: Contains test cases for OpenAI runners.
- **tests/llm/test_facebook.py**: Contains test cases for Facebook's LLaMAcpp runner.
- **tests/llm/test_nlpcloud.py**: Contains test cases for NLP Cloud's GPTJAPI runner.
- **tests/function/test_doctest_based.py**: Contains test cases for DoctestBasedFewShotFunction.
- **tests/function/test_parameter_based.py**: Contains test cases for ParameterBasedFewShotFunction.

## Contributing

To contribute to the GPTRun project, please follow the project's structure and add new LLM runners, few-shot functions, or tests as needed. Update the relevant documentation to reflect the changes made, and ensure that all tests pass before submitting a pull request.

