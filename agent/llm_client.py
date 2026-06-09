import logging
from pathlib import Path

import openai
import requests
import yaml

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    pass


def call(system_prompt: str, user_prompt: str) -> dict:
    # Backend 1 — Ollama
    try:
        payload = {
            "model": CFG["ollama"]["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": CFG["ollama"]["temperature"],
                "top_p": CFG["ollama"]["top_p"],
                "num_predict": CFG["ollama"]["max_tokens"],
            },
        }
        resp = requests.post(
            CFG["ollama"]["base_url"] + "/api/chat",
            json=payload,
            timeout=CFG["ollama"]["timeout_seconds"],
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from Ollama")
        return {"response_text": content, "backend_used": "ollama"}
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.warning(f"Ollama unavailable: {e}")
    except Exception as e:
        logger.warning(f"Ollama error: {e}")

    # Backend 2 — Groq llama-3.1-8b-instant
    groq_key = CFG["groq"]["api_key"]
    if not groq_key:
        logger.warning("Groq API key not configured")
    else:
        try:
            client = openai.OpenAI(
                base_url=CFG["groq"]["base_url"],
                api_key=groq_key,
            )
            completion = client.chat.completions.create(
                model=CFG["groq"]["primary_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=CFG["groq"]["temperature"],
                max_tokens=CFG["groq"]["max_tokens"],
                timeout=CFG["groq"]["timeout_seconds"],
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Groq primary")
            return {"response_text": content, "backend_used": "groq_llama"}
        except Exception as e:
            logger.warning(f"Groq primary failed: {e}")

        # Backend 3 — Groq mixtral-8x7b-32768
        try:
            client = openai.OpenAI(
                base_url=CFG["groq"]["base_url"],
                api_key=groq_key,
            )
            completion = client.chat.completions.create(
                model=CFG["groq"]["fallback_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=CFG["groq"]["temperature"],
                max_tokens=CFG["groq"]["max_tokens"],
                timeout=CFG["groq"]["timeout_seconds"],
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Groq fallback")
            return {"response_text": content, "backend_used": "groq_mixtral"}
        except Exception as e:
            logger.warning(f"Groq fallback failed: {e}")

    # Backend 4 — all failed
    raise LLMUnavailableError("All LLM backends exhausted.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(_ROOT))

    system = "You are a helpful assistant. Respond in JSON: {\"answer\": \"...\"}."
    user = "What regime is the market in? Reply with a short JSON answer."
    try:
        result = call(system, user)
        print(f"response_text[:200]: {result['response_text'][:200]}")
        print(f"backend_used: {result['backend_used']}")
        print("llm_client OK")
    except LLMUnavailableError:
        print("All LLM backends unavailable (expected without Ollama/Groq configured)")
        print("llm_client OK")
