import sys
import json
from pydantic import ValidationError
from typing import Dict, Any, List
from src.filter import JsonCalling, JsonDefinition


def read_json(path: str) -> Any:
    """Loads a JSON file from the specified path.

    Args:
        path: File system path.
    Returns:
        Parsed JSON content.
    """
    try:
        with open(path, 'r',  encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path} not found\n")
    except Exception:
        raise ValueError("[ERROR]\n")


def check_prompt(data: List[Dict[str, Any]]) -> List[JsonCalling]:
    """Validates data against JsonCalling schema.

    Args:
        data: List of raw dictionaries.
    Returns:
        List of validated JsonCalling objects.
    """
    try:
        return [JsonCalling(**item) for item in data]
    except ValidationError as e:
        print("Expected error: calling_test.json")
        for error in e.errors():
            print(f"- {error['msg']}")
        sys.exit(1)


def check_definitions(data: List[Dict[str, Any]]) -> List[JsonDefinition]:
    """Validates data against JsonDefinition schema.

    Args:
        data: List of raw dictionaries.
    Returns:
        List of validated JsonDefinition objects.
    """
    try:
        return [JsonDefinition(**item) for item in data]
    except ValidationError as e:
        print("Expected validation error: Definition.json")
        for error in e.errors():
            print(f"- {error['msg']}")
        sys.exit(1)
