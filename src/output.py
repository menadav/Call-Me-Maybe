import sys
import json
from typing import List, Dict, Any

def output_json(file_path: str, data: dict[str, Any]) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(dict_json, f)
    except IOError as e:
        raise ValueError("[ERROR] Path incorrect")
