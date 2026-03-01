import sys
import os
import argparse
import json

try:
    from pydantic import BaseModel, ValidationError
    import numpy as np
except ImportError:
    sys.stderr.write("[ERROR] Run: make install\n")
    sys.exit(1)

from src.config_json import read_json, check_prompt, check_definitions
from src.tokenizer import LlmManager
from src.output import output_json
from llm_sdk import Small_LLM_Model
from src.filter import FunctionCallResult

def main() -> None:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Tool")
    parser.add_argument("--functions_definition", type=str, 
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, 
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, 
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()
    try:
        if not os.path.exists(args.input):
           raise FileNotFoundError(f"[ERROR] Path incorrect input: {args.input}")
        if not os.path.exists(args.functions_definition):
           raise FileNotFoundError(f"[ERROR] Path incorrect definitions {args.functions_definition}")
        data_calling = read_json(args.input)
        calling = check_prompt(data_calling)
        data_definition = read_json(args.functions_definition)
        definitions = check_definitions(data_definition)
        sdk = Small_LLM_Model()
        llm_manager = LlmManager(definitions, calling, sdk)
        llm_manager.output_json()
        
#        dict_json = json.loads(file)
#        validate_json = [FunctionCallResult(**item).model_dump() for item in dict_json]
#        output_json(args.output, validate_json)
    except (AttributeError, FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"{str(e)}\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write("[ERROR] Load JSON")
        sys.exit(1)
    except ValidationError as e:
        sys.stderr.write(f"{str(e)}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        sys.stderr.write(f"Error real: {type(e).__name__}: {e}\n")
        traceback.print_exc() # Esto te dirá la línea exacta
        sys.exit(1)

if __name__ == "__main__":
    main() 
