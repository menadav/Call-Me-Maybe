import sys
import os
import argparse
import json
from typing import TYPE_CHECKING

try:
    from pydantic import ValidationError
except ImportError:
    sys.stderr.write("[ERROR] Run: make install\n")
    sys.exit(1)

from src.config_json import read_json, check_prompt, check_definitions
from src.tokenizer import LlmManager
from src.output import output_json
if TYPE_CHECKING:
    from src.llm_sdk import Small_LLM_Model
else:
    try:
        from src.llm_sdk import Small_LLM_Model
    except ModuleNotFoundError:
        from llm_sdk import Small_LLM_Model


def main() -> None:
    """
    Parses arguments, validates input files, and executes
    the LLM function calling workflow.

    This function performs the following steps:
    1. Sets up command-line argument parsing.
    2. Validates the existence of input and definition files.
    3. Reads and sanitizes JSON data into Pydantic models.
    4. Initializes the LLM SDK and the LlmManager.
    5. Generates structured JSON output and saves it to a file.

    Raises:
        FileNotFoundError: If input paths do not exist.
        ValidationError: If JSON data doesn't match expected schemas.
        JSONDecodeError: If input files are not valid JSON.
        SystemExit: On any handled error to return a non-zero exit code.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling Tool"
        )
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()
    try:
        if not os.path.exists(args.input):
            raise FileNotFoundError(
                f"[ERROR] Path incorrect input: {args.input}")
        if not os.path.exists(args.functions_definition):
            raise FileNotFoundError(
                "[ERROR] Path incorrect definitions"
                f"{args.functions_definition}"
                )
        data_calling = read_json(args.input)
        calling = check_prompt(data_calling)
        data_definition = read_json(args.functions_definition)
        definitions = check_definitions(data_definition)
        sdk = Small_LLM_Model()
        llm_manager = LlmManager(definitions, calling, sdk)
        validate_json = llm_manager.output_json()
        output_json(args.output, validate_json)
    except (AttributeError, FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"{str(e)}\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write("[ERROR] Load JSON")
        sys.exit(1)
    except ValidationError as e:
        sys.stderr.write(f"{str(e)}\n")
        sys.exit(1)
    except Exception:
        sys.stderr.write("[ERROR]")
        sys.exit(1)


if __name__ == "__main__":
    main()
