import sys
import json
from typing import Dict, Any, Literal, List
from src.filter import JsonCalling, JsonDefinition

def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path} not found\n")
    except Exception:
        raise ValueError("[ERROR]\n")

def check_prompt(data: dict) -> list[dict]:
    try:
        return [JsonCalling(**item) for item in data]
    except ValidationError as e:
        print("Expected validation error: calling_test.json")
        for error in e.errors():
             print(f"- {error['msg']}")
        sys.exit(1)

def check_definitions(data: dict) -> list[dict]:
    try:
        return [JsonDefinition(**item) for item in data]
    except ValidationError as e:
        print("Expected validation error: Definition.json")
        for error in e.errors():
             print(f"- {error['msg']}")
        sys.exit(1)
