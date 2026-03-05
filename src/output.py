from pathlib import Path
import json
from typing import Dict, Any, List


def output_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    try:
        dir_path = Path(file_path)
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dir_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError:
        raise ValueError("[ERROR] Path incorrect Output")
