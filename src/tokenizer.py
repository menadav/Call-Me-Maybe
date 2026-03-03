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
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary: Dict[str, int] = json.load(f)
            self.id_to_token = {v: k for k, v in self.vocabulary.items()}
        except Exception as e:
            raise ValueError(e)

    def _interact_with_llm(self) -> list[tuple]:
        defs_text = ""
        for func in self.definitions:
            params = ", ".join(func.parameters.keys())
            defs_text += f"\n- {func.name}({params}): {func.description}"
        results = []
        for prompt in self.calling:
            json_prefill = f'{{\n  "prompt": "{prompt.prompt}",\n  "name": "'
            full_prompt = (
                "SYSTEM: You are a professional API assistant. "
                "You must ONLY use the function names provided in the list. "
                "Do not invent new function names.\n"
                f"FUNCTIONS LIST:{defs_text}\n\n"
                "EXAMPLE:\n"
                "User: Greet Batman\n"
                "Assistant: {\"prompt\": \"Greet Batman\", \"name\": \"fn_greet_person\", \"parameters\": {\"name\": \"Batman\"}}\n\n"
                f"USER REQUEST: {prompt.prompt}\n"
                "ASSISTANT (JSON):"
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
        if current_text.endswith('"name": "'):
            return int(np.argmax(logi_np))
        if re.search(r'"name": "[^"]+"$', current_text):
            res = self.vocabulary.get('",\n  "parameters": {') or self.vocabulary.get('", "parameters": {')
            if res is not None:
                return self._get_next_token(logi_np, [res])
            return safe_get('",')
        if current_text.endswith('",'):
            return safe_get('\n  "parameters": {') or safe_get(' "parameters": {')
        return int(np.argmax(logi_np))


    def _parse_generated_json(self, text: str) -> dict | None:
        try:
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            count = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    count += 1
                elif text[i] == '}':
                    count -= 1
                    if count == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                return None
            clean_str = text[start_idx:end_idx+1]
            obj = json.loads(clean_str)
            prompt = obj.get("prompt")
            name = obj.get("name") or obj.get("function")
            params = obj.get("parameters") or obj.get("arguments", {})
            if isinstance(params, list):
                params = {
                    item['name']: item.get('value')
                    for item in params
                    if isinstance(item, dict) and 'name' in item
                }
            ordered_obj = {}
            if prompt:
                ordered_obj["prompt"] = prompt
            ordered_obj["name"] = name
            ordered_obj["parameters"] = params
            return ordered_obj
        except Exception:
            return None

    def output_json(self) -> None:
        tensors = self._interact_with_llm()
        final_results = []
        for t, prefix in tensors:
            generated_json = prefix
            prefix_tokens_raw = self.sdk.encode(prefix)
            if hasattr(prefix_tokens_raw, "tolist"):
                prefix_tokens = prefix_tokens_raw.flatten().tolist()
            elif isinstance(prefix_tokens_raw, np.ndarray):
                prefix_tokens = prefix_tokens_raw.tolist()
            else:
                prefix_tokens = list(prefix_tokens_raw)
            input_ids = t.flatten().tolist() + prefix_tokens
            for _ in range(150):
                logi = self.sdk.get_logits_from_input_ids(input_ids)
                if hasattr(logi, "detach"):
                    logi_np = logi.detach().cpu().numpy().flatten()
                else:
                    logi_np = np.array(logi).flatten()
                next_token_id = self._steps_output(logi_np, generated_json)
                input_ids.append(next_token_id)
                generated_json += self._decode_function(next_token_id)
                print(generated_json)
                if generated_json.count('{') == generated_json.count('}') and "parameters" in generated_json:
                    break
                result_obj = self._parse_generated_json(generated_json)
                if result_obj:
                    print(result_obj)
                    final_results.append(result_obj)
                    break
        print(final_results)
