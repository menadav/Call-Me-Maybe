*This project has been created as part of the 42 curriculum by dmena-li*
# Call-Me-Maybe


## Description

*Call Me Maybe* is a project designed to explore and implement function calling within Large Language Models (LLMs). The goal is to translate natural language prompts into structured, machine-executable function calls with typed arguments. By using a small 0.6B parameter model (Qwen/Qwen3-0.6B), the project demonstrates how **constrained decoding** can achieve 100% reliability in generating valid JSON, bridging the gap between human language and computer instructions

## Instructions

#### Installation
This project requires Python 3.10 or later. It uses uv for dependency management. To install the necessary packages (numpy, pydantic, and the llm_sdk dependencies):

```
make install
```
You can run the default pipeline using the Makefile:
```
make run 
```
Or manually specify custom input/output files using the following command:
```
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```
*Note on 42 Environment Compatibility: This Makefile is optimized for the 42 school environment to address local storage limitations. Because the standard home directory often lacks sufficient space for large models and environments, the configuration explicitly redirects the virtual environment and cache directories to ***/goinfre***. By exporting **UV_CACHE_DIR** and **HF_HOME** to this partition, the project ensures stable execution and prevents disk quota issues during model downloading and dependency installation.*


## Example usage

1. Input Prompt
We provide the AI with a specific task in natural language:
```
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks"
  }
```

2. Function Definitions (Tool Selection)
The AI is provided with a catalog of available functions. Its job is to select the most appropriate tool based on the user's request. In this case, it identifies fn_substitute_string_with_regex as the correct function:
```
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },  
      "regex": {
        "type": "string"
      },  
      "replacement": {
        "type": "string"
      }   
    },  
    "returns": {
      "type": "string"
    }   
 }
```
3. Output Parsing and Filtering
The final step involves parsing the AI's raw generation and filtering the relevant data. We extract the selected function and the mapped arguments to generate a structured JSON output that our system can execute:
```
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "function": "fn_substitute_string_with_regex",
  "parameters":{ "source_string": "Programming is fun", "regex": "aeiou", "replacement": "asterisk" }
}
```
## Resources
* https://poloclub.github.io/transformer-explainer - Understand transformers.
* https://docs.pydantic.dev/latest/ - Data validation
* https://www.youtube.com/watch?v=VkWlLSTdHs8 - Understand decoding
* Google Gemini 
