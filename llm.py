import os
import json
import logging
from urllib import request, error, parse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Provider configuration
#
# ollama   → Cloud API  (requires API key, hosted remotely)
# lm_studio → Local LAN server (no key, runs on 192.168.x.x)
# ──────────────────────────────────────────────
LLM_PROVIDERS = {
    "ollama": {
        "label": "gpt-oss:120b",
        "model": os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
        "base_url": os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api"),
        "api_key": os.environ.get("OLLAMA_API_KEY", ""),   # Real API key required for Ollama Cloud
        "type": "cloud",
        "description": "Ollama Cloud API"
    },
    "lm_studio": {
        "label": "Gemma 4 (Local)",
        "model": os.environ.get("LM_MODEL_NAME", "google/gemma-4-e4b"),
        "base_url": os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.0.100:1234/v1"),
        "api_key": "lm-studio",                            # Dummy — no auth on local server
        "type": "local",
        "description": "LM Studio — local LAN server"
    }
}


def _ollama_chat_url(base_url: str) -> str:
    """Return the correct Ollama /api/chat endpoint for cloud or local hosts."""
    base = (base_url or "https://ollama.com/api").strip().rstrip("/")

    # Older/incorrect cloud host used by some OpenAI-compatible examples.
    parsed = parse.urlparse(base)
    if parsed.netloc == "api.ollama.ai":
        return "https://ollama.com/api/chat"

    if base.endswith("/api/chat"):
        return base
    if base.endswith("/api"):
        return f"{base}/chat"
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")

    return f"{base}/api/chat"


def _is_ollama_cloud_url(base_url: str) -> bool:
    parsed = parse.urlparse(base_url or "https://ollama.com/api")
    return parsed.netloc in {"ollama.com", "www.ollama.com", "api.ollama.ai"}


def _get_ollama_response(
    config: dict,
    messages: list,
    temperature: float,
    max_tokens: int
) -> str:
    """Call Ollama's native chat API. Ollama Cloud is not OpenAI-compatible."""
    endpoint = _ollama_chat_url(config.get("base_url", "https://ollama.com/api"))
    api_key = (config.get("api_key") or "").strip()

    if _is_ollama_cloud_url(endpoint) and not api_key:
        raise RuntimeError("OLLAMA_API_KEY is missing. Please add your Ollama Cloud API key in the .env file.")

    payload = {
        "model": config["model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=120) as res:
            raw = res.read().decode("utf-8")
            data = json.loads(raw)
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API returned HTTP {e.code}: {err_body}")
    except error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama API at {endpoint}: {e.reason}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama API returned invalid JSON: {e}")

    message = data.get("message") or {}
    answer = (message.get("content") or "").strip()

    if not answer:
        raise RuntimeError(f"Ollama API returned no assistant message content: {data}")

    return answer


def get_llm_response(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    history: list = None,
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> str:
    """
    Call the selected LLM provider.

    Args:
        provider:      "ollama" or "lm_studio"
        system_prompt: System instruction for the AI
        user_prompt:   The user's message (with context embedded)
        history:       Prior conversation turns [{role, content}, ...]
        temperature:   Sampling temperature
        max_tokens:    Max response tokens

    Returns:
        AI reply as a string
    """
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(LLM_PROVIDERS.keys())}")

    config = LLM_PROVIDERS[provider]

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]

    # Append conversation history (last 6 turns to keep context manageable)
    if history:
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_prompt})

    logger.info(f"Calling {provider} ({config['model']}) with {len(messages)} messages")

    try:
        if provider == "ollama":
            answer = _get_ollama_response(config, messages, temperature, max_tokens)
        else:
            client = OpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"]
            )
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message.content.strip()

        logger.info(f"LLM response received: {len(answer)} chars")
        return answer

    except Exception as e:
        logger.error(f"LLM call failed for provider '{provider}': {e}")
        raise RuntimeError(f"Could not get a response from {config['label']}. Error: {str(e)}")
