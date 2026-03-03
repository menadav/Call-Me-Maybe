import json
from typing import Dict, Any, List


def output_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError:
        raise ValueError("[ERROR] Path incorrect")
