try:
    import numpy as np
except ImportError:
    raise ValueError("[ERROR] Run: make install\n")
import re
import json
from typing import List, Any, Dict, Optional
from src.filter import JsonDefinition, JsonCalling


class LlmManager:
    """ Manages interaction with a Language Model for function
        calling tasks.
        This class handles prompt construction,
        token-level constraints for JSON
        generation, and parsing the LLM output into
        structured data.

        Attributes:
        definitions (list[JsonDefinition]): List of available
        function definitions.
        calling (List[JsonCalling]): List of user requests/prompts to process.
        sdk (Any): Instance of the SDK used for inference and tokenization.
        vocabulary (Dict[str, int]): Token-to-ID mapping from the SDK.
        id_to_token (Dict[int, str]): ID-to-token mapping for decoding.
    """
    def __init__(
        self,
        definitions: list[JsonDefinition],
        calling: List[JsonCalling],
        sdk_instance: Any
    ) -> None:
        self.definitions = definitions
        self.calling = calling
        """ Initializes the LlmManager with tools and SDK. """
        self.sdk = sdk_instance
        try:
            vocab_path = self.sdk.get_path_to_vocab_file()
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary: Dict[str, int] = json.load(f)
            self.id_to_token = {v: k for k, v in self.vocabulary.items()}
        except Exception as e:
            raise ValueError(e)

    def _interact_with_llm(self) -> list[tuple]:
        """ Prepares prompts and initial JSON structure for the LLM.

        Returns:
            list[tuple]: A list of tuples containing
            (encoded_tokens, json_prefix).
        """
        defs_text = ""
        for func in self.definitions:
            params = ", ".join(func.parameters.keys())
            defs_text += f"\n- {func.name}({params}),: {func.description},"
        results = []
        for prompt in self.calling:
            final_text = prompt.prompt
            safe_prompt = final_text.replace('\\', '\\\\').replace('"', '\\"')
            json_prefill = (
                f'{{\n  "prompt": "{safe_prompt}",'
                '\n  "function": "'
            )
            full_prompt = (
                "Task: Convert request to JSON.\n"
                f"Functions: {defs_text},\n"
                f"Request: {safe_prompt},\n"
                "JSON Output:"
            )
            tokens = self.sdk.encode(full_prompt)
            results.append((tokens, json_prefill))
        return results

    def _decode_function(self, token_id: int) -> Any:
        """ Decodes a single token ID back into a string.

        Args:
            token_id (int): The ID of the token to decode.

        Returns:
            Any: The decoded string representation of the token.
        """
        word = self.sdk.decode([token_id])
        return word

    def _get_next_token(
        self, logits_np: np.ndarray,
        allowed_ids: List[Optional[int]]
    ) -> int:
        """ Selects the best next token from logits, restricted to allowed IDs.
        Args:
            logits_np (np.ndarray): Array of logits from the model output.
            allowed_ids (List[Optional[int]]): IDs that the model is
            allowed to pick.
        Returns:
            int: The index (ID) of the selected token.
        """
        valid_ids = [i for i in allowed_ids if i is not None]
        if not valid_ids:
            return int(np.argmax(logits_np))
        max_idx = logits_np.shape[0]
        valid_ids = [i for i in valid_ids if i < max_idx]
        constrained_logits = np.full_like(logits_np, -np.inf)
        for idx in valid_ids:
            constrained_logits[idx] = logits_np[idx]
        return int(np.argmax(constrained_logits))

    def _steps_output(
        self, logi_np: np.ndarray,
        current_text: str
    ) -> Optional[int | None]:
        """ Determines the next token based on current JSON generation state.
        Forces specific tokens (like keys or structural characters) based on
        regex patterns to ensure valid JSON output.

        Args:
            logi_np (np.ndarray): Current logits from the model.
            current_text (str): The text generated so far.

        Returns:
            Optional[int]: The forced or most likely next token ID.
        """

        def get_id(text: str) -> Optional[int | None]:
            return self.vocabulary.get(text)

        def force(token_str: str) -> Optional[int | None]:
            tid = get_id(token_str)
            if tid is not None:
                return self._get_next_token(logi_np, [tid])
            return None
        if re.search(r'"function":\s*"[^"]+"$', current_text):
            return force(', "parameters":') or force(',')
        if current_text.strip().endswith('"parameters":'):
            return force(' {') or force('{') or int(np.argmax(logi_np))
        if current_text.strip().endswith('"parameters'):
            return force('":') or force('": ')
        if current_text.endswith(' "') or current_text.endswith('\n"'):
            if '"parameters"' not in current_text\
                    and re.search(r'"function":\s*"[^"]+"', current_text):
                return force('parameters')
        if re.search(r'"function":\s*"[^"]+"$', current_text.strip()):
            return force('", "parameters":') or \
                   force(',') or \
                   force(', "parameters":') or \
                   int(np.argmax(logi_np))
        return int(np.argmax(logi_np))

    def _parse_generated_json(self, text: str) -> dict | None:
        """ Cleans and parses the raw generated string into a
            Python dictionary.
        Args:
            text (str): The raw text containing a JSON block.
        Returns:
            Optional[dict]: The parsed and formatted dictionary,
            or None if invalid.
        """
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
            if isinstance(params, str):
                params = params.replace('\\', '\\\\')
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

    def output_json(self) -> List[Dict[str, Any]]:
        """ Executes the full generation loop to produce
            structured JSON outputs.

        Returns:
            List[Dict[str, Any]]: List of results extracted from the model.
        """
        tensors = self._interact_with_llm()
        final_results = []
        for t, prefix in tensors:
            generated_json = prefix
            prefix_tokens_raw = self.sdk.encode(prefix)
            prefix_tokens = prefix_tokens_raw.flatten().tolist()
            input_ids = t.flatten().tolist() + prefix_tokens
            for _ in range(120):
                logi = self.sdk.get_logits_from_input_ids(input_ids)
                logi_np = np.array(logi).flatten()
                next_token_id = self._steps_output(logi_np, generated_json)
                input_ids.append(next_token_id)
                if next_token_id is None:
                    break
                generated_json += self._decode_function(next_token_id)
                if generated_json.count('{') == generated_json.count('}') \
                        and '"parameters"' in generated_json:
                    result_obj = self._parse_generated_json(generated_json)
                    if result_obj:
                        final_results.append(result_obj)
                    break
        return final_results
