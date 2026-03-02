from typing import List, Any, Dict
from src.filter import JsonDefinition, JsonCalling
import re
import json
import numpy as np

class LlmManager:
    def __init__(
        self,
        definitions: list[JsonDefinition],
        calling: List[JsonCalling],
        sdk_instance: Any
    ) -> None:
        self.definitions = definitions
        self.calling = calling
        self.sdk = sdk_instance
        try:
            vocab_path = self.sdk.get_path_to_vocab_file()
            with open(vocab_path, 'r') as f:
                self.vocabulary: Dict[str, int] = json.load(f)
            self.id_to_token = {v: k for k, v in self.vocabulary.items()}
        except Exception:
            raise ValueError("Error")


    def _interact_with_llm(self) -> list[list[int]]:
        defs_text: str = ""
        for func in self.definitions:
            params = ", ".join(func.parameters.keys())
            defs_text += f"\n- {func.name}({params}): {func.description}"
        results = []
        for prompt in self.calling:
            json_prefill = f'{{\n  "prompt": "{prompt.prompt}",\n  "name": "'
            full_prompt = (
                f"Task: Select the correct function name and extract argument in JSON. \n"
                f"Functions: {defs_text}\n"
                f"User request: {prompt.prompt}\n"
                f"Response (JSON only):"
            )
            tokens = self.sdk.encode(full_prompt)
            results.append((tokens, json_prefill))
        return results

    def _decode_function(self, token_id) -> str:
        word = self.sdk.decode([token_id])
        return word

    def _get_next_token(self, logits_np, allowed_ids):
        valid_ids = [i for i in allowed_ids if i is not None]
        if not valid_ids:
            return int(np.argmax(logits_np))
        max_idx = logits_np.shape[0]
        valid_ids = [i for i in valid_ids if i < max_idx]
        constrained_logits = np.full_like(logits_np, -np.inf)
        for idx in valid_ids:
            constrained_logits[idx] = logits_np[idx]
        return int(np.argmax(constrained_logits))

    def _steps_output(self, logi_np, current_text):
        def safe_get(key):
            res = self.vocabulary.get(key)
            if res is not None:
                return self._get_next_token(logi_np, [res])
            return int(np.argmax(logi_np))
        if current_text.endswith('"function": "'):
            return int(np.argmax(logi_np))
        if re.search(r'"function": "[^"]+"$', current_text):
            res = self.vocabulary.get('",\n  "parameters": {')
            if res is not None:
                return self._get_next_token(logi_np, [res])
            return safe_get('",')
        if current_text.endswith('",'):
            return safe_get('\n  "parameters": {')
        return int(np.argmax(logi_np))

    def output_json(self) -> None:
        tensors = self._interact_with_llm()
        final_results = []
        for t , prefix in tensors:
            generated_json = prefix
            prefix_tokens_raw = self.sdk.encode(prefix)
            if hasattr(prefix_tokens_raw, "tolist"):
                prefix_tokens = prefix_tokens_raw.flatten().tolist()
            elif isinstance(prefix_tokens_raw, np.ndarray):
                prefix_tokens = prefix_tokens_raw.tolist()
            else:
                prefix_tokens = list(prefix_tokens_raw)
            input_ids = t.flatten().tolist() + prefix_tokens
            limit = 0
            while limit < 100:
                limit += 1
                logi = self.sdk.get_logits_from_input_ids(input_ids)
                if hasattr(logi, "detach"):
                    logi_np = logi.detach().cpu().numpy().flatten()
                else:
                    logi_np = np.array(logi).flatten()
                next_token_id = self._steps_output(logi_np, generated_json)
                input_ids.append(next_token_id)
                word = self._decode_function(next_token_id)
                generated_json += word
                if generated_json.count('{') == generated_json.count('}') and "function" in generated_json:
                    break
            try:
                start_idx = generated_json.find('{')
                end_idx = generated_json.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    clean_json = generated_json[start_idx:end_idx+1]
                    obj = json.loads(clean_json)
                    final_results.append(obj)
                else:
                    print(f"Error parseando: {generated_json}")
                    pass
            except Exception as e:
                print(f"Error parseando: {e}")
        print(json.dumps(final_results, indent=2, ensure_ascii=False))
