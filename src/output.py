from pathlib import Path
import json
from typing import Dict, Any, List


def output_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Writes the generated function calling results to a JSON file.

    This function ensures the target directory exists before
    attempting to write.
    It uses UTF-8 encoding and pretty-printing (indent=4) to ensure the
    output is both machine-readable and human-friendly.

    Args:
        file_path (str): The destination path where the JSON file
        will be saved.
        data (List[Dict[str, Any]]): A list of dictionaries containing the
            processed function calls and their parameters.

    Raises:
        ValueError: If the file cannot be written due to invalid paths,
            permission issues, or other I/O errors.
    """
    try:
        dir_path = Path(file_path)
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dir_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError:
        raise ValueError("[ERROR] Path incorrect Output")
