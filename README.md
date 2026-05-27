# 📞 Call Me Maybe


## Description

**Call Me Maybe** is a project that explores **function calling with Large Language Models** through constrained decoding (logit masking). Given a natural language prompt and a catalog of available functions, the system reliably translates user intent into a structured, machine-executable JSON call with typed arguments.

It uses `Qwen/Qwen3-0.6B` — a tiny, fully local model — and demonstrates that with proper **constrained decoding**, even a sub-1B parameter model can achieve **100% reliability** in producing valid, schema-compliant JSON outputs.


## ✨ Features

- **Constrained decoding (logit masking)** — tokens that would produce invalid JSON are masked at inference time, guaranteeing structural correctness
- **Schema-driven generation** — function signatures defined via JSON schemas guide argument type inference
- **Fully local** — no OpenAI API, no cloud calls; everything runs on-device with `Qwen/Qwen3-0.6B`
- **Custom `llm_sdk`** — an internal workspace package that wraps the model and decoding logic
- **Pydantic validation** — outputs are validated against the function schema before being returned
- **42 school compatible** — Makefile redirects caches to `/goinfre` to avoid home directory quota issues

---

## 🏗️ How it works

```
Natural Language Prompt + Function Definitions
             │
             ▼
    [1] Prompt Construction    ← Formats the task and tool catalog for the LLM
             │
             ▼
    [2] Constrained Decoding   ← Logit masking ensures only valid JSON tokens are sampled
             │
             ▼
    [3] Output Parsing         ← Extracts function name and typed arguments
             │
             ▼
    [4] Pydantic Validation    ← Validates output against the function's parameter schema
             │
             ▼
    Structured JSON Function Call
```

The key insight: instead of hoping the model generates valid JSON, the system **enforces** it at the token level. At each decoding step, any token that would break JSON structure or violate the target schema is assigned `-inf` logit probability and cannot be sampled.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
make install
```

Uses [`uv`](https://github.com/astral-sh/uv) with workspace support to install `numpy`, `pydantic`, and the local `llm_sdk` package.

### 2. Run the default pipeline

```bash
make run
```

### 3. Custom input/output files

```bash
uv run python -m src \
  [--functions_definition <file>] \
  [--input <file>] \
  [--output <file>]
```

---

## 📋 Example

**Input prompt** (`data/input/`):
```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks"
}
```

**Function catalog** (provided as a JSON schema):
```json
{
  "name": "fn_substitute_string_with_regex",
  "description": "Replace all occurrences matching a regex pattern in a string.",
  "parameters": {
    "source_string": { "type": "string" },
    "regex":         { "type": "string" },
    "replacement":   { "type": "string" }
  },
  "returns": { "type": "string" }
}
```

**Structured output**:
```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "function": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "[aeiouAEIOU]",
    "replacement": "*"
  }
}
```

---

## 📁 Project Structure

```
IA_LLM_Logit_Masking/
├── src/                    # Main pipeline: prompt construction, parsing, CLI
├── llm_sdk/                # Internal workspace package: model loading, constrained decoding
├── data/
│   └── input/              # Input prompts and function definitions
├── Makefile                # Install, run, and environment setup
└── pyproject.toml          # Project metadata and workspace config
```

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Language model | `Qwen/Qwen3-0.6B` via HuggingFace |
| Constrained decoding | Custom `llm_sdk` (workspace package) |
| Output validation | `pydantic >= 2.12` |
| Numerical ops | `numpy >= 2.2` |
| Package manager | `uv` (workspace mode) |
| Linting / typing | `flake8`, `mypy` |

---

## 📋 Requirements

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) installed
- ~1–2 GB disk space for model weights

> **42 School note:** The Makefile exports `UV_CACHE_DIR` and `HF_HOME` to `/goinfre` to avoid home directory quota issues during model download and environment setup.

---

## 💡 Why Constrained Decoding?

Standard LLM prompting for structured outputs is unreliable: models can produce malformed JSON, hallucinate argument names, or use incorrect types. **Logit masking** solves this at the source — by modifying the model's probability distribution at each token step, only tokens consistent with the target schema can ever be generated. The result is 100% structurally valid output, regardless of model size.

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.
