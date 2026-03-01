from typing import List, Any, Dict
from src.filter import JsonDefinition, JsonCalling
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
            full_prompt = (
                f"Functions: {defs_text}\n"
                f"User request: {prompt.prompt}\n"
                f"Response in JSON format:"
            )
            tokens = self.sdk.encode(full_prompt)
            results.append(tokens)
        return results

    def _decode_function(self, token_id) -> str:
        word = self.sdk.decode([token_id])
        return word

    def _get_next_token(self, logits_np, allowed_ids):
        valid_ids = [i for i in allowed_ids if i is not None]
        if not valid_ids:
            return int(np.argmax(logits_np))
        constrained_logits = np.full_like(logits_np, -np.inf)
        constrained_logits[allowed_ids] = logits_np[allowed_ids]
        return int(np.argmax(constrained_logits))

    def _steps_output(self, logi_np, current_text):
        def safe_get(key, ids=None):
            if ids:
                return self._get_next_token(logi_np, ids)
            res = self.vocabulary.get(key)
            if res is not None:
                return self._get_next_token(logi_np, [res])
            return int(np.argmax(logi_np))
        num_quotes = current_text.count('"')
        if not current_text:
            return safe_get('{')
        if current_text.endswith('{') or current_text.endswith(', ') or current_text.endswith(': '):
            return safe_get('"')
        if current_text.endswith('"'):
            if num_quotes == 1:
                return safe_get('prompt')
            if num_quotes == 5:
                return safe_get('function')
            if num_quotes == 9:
                return safe_get('parameters')
        if current_text.endswith('prompt') or current_text.endswith('function') or current_text.endswith('parameters'):
            return safe_get('"')
        if current_text.endswith('":'):
            return safe_get(' ')

        if current_text.endswith('"'):
            if num_quotes in [2, 6, 10]:
                return safe_get(':')
            if num_quotes in [4, 8]:
                return safe_get(',')
        if current_text.endswith(','):
            return safe_get(' ')
        if num_quotes == 12 and "parameters" in current_text:
            if current_text.strip().endswith('}'):
                return self.vocabulary.get(self.sdk.tokenizer.eos_token, int(np.argmax(logi_np)))
            if current_text.count('{') > current_text.count('}'):
                return safe_get('}')
        return int(np.argmax(logi_np)
)

    def output_json(self) -> str:
        tensors = self._interact_with_llm()
        for t in tensors:
            generated_json = ""
            input_ids = t.flatten().tolist()
            while(1):
                logi = self.sdk.get_logits_from_input_ids(input_ids)
                if hasattr(logi, "detach"):
                    logi_np = logi.detach().cpu().numpy().flatten()
                else:
                    logi_np = np.array(logi).flatten()
                next_token_id = self._steps_output(logi_np, generated_json)
                input_ids.append(next_token_id)
                word = self._decode_function(next_token_id)
                generated_json += word
                if "}" in word and "parameters" in generated_json:
                    break
        return generated_json
