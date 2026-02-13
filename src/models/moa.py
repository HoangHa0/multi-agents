import os
import time
import random
import threading
from typing import Optional, List, Dict, Any

from mistralai import Mistral, SDKError

from openai import OpenAI

from google import genai  
from google.genai import types


# ============================
# Prompts
# ============================

SYNTHESIZE_PROMPT = (
    "You have been provided with a set of responses from various models to the latest user query. "
    "Your task is to synthesize these responses into a single, high-quality response. "
    "Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. "
    "Do not simply replicate the given answers; produce a refined, accurate, and comprehensive reply. "
    "Keep the final answer well-structured and coherent.\n\n"
    "Responses from models:\n"
)

# ============================
# Round-robin Mistral client
# ============================

# --- Mistral key pool (round robin) ---------------------------------

_MISTRAL_API_KEYS = [
    os.environ.get("MISTRAL_API_KEY"),
    os.environ.get("MISTRAL_API_KEY_1"),
    os.environ.get("MISTRAL_API_KEY_2"),
    os.environ.get("MISTRAL_API_KEY_3"),
    os.environ.get("MISTRAL_API_KEY_4"),
    os.environ.get("MISTRAL_API_KEY_5"),
    os.environ.get("MISTRAL_API_KEY_6"),
    os.environ.get("MISTRAL_API_KEY_7"),
    os.environ.get("MISTRAL_API_KEY_8"),
    os.environ.get("MISTRAL_API_KEY_9"),
    os.environ.get("MISTRAL_API_KEY_10"),
]
_MISTRAL_API_KEYS = [k for k in _MISTRAL_API_KEYS if k]

_mistral_rr_counter = 0
_mistral_rr_lock = threading.Lock()


def _mistral_rr_start_index() -> int:
    global _mistral_rr_counter
    with _mistral_rr_lock:
        if not _MISTRAL_API_KEYS:
            return 0
        idx = _mistral_rr_counter % len(_MISTRAL_API_KEYS)
        _mistral_rr_counter += 1
        return idx


def _is_rate_limited_mistral(e: Exception) -> bool:
    # Mistral SDKError includes status text; robust string checks work across versions.
    s = str(e)
    return ("Status 429" in s) or ('"code":"1300"' in s) or ("rate limit" in s.lower())


class _MistralRRChatProxy:
    def __init__(self, parent):
        self._parent = parent

    def complete(self, **kwargs):
        return self._parent._complete(**kwargs)


class _MistralRoundRobinClient:
    """Round-robin across multiple Mistral keys; on 429, try next key immediately."""

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("No Mistral API keys found in env.")
        self._clients = [Mistral(api_key=k) for k in api_keys]
        self.chat = _MistralRRChatProxy(self)

    def _complete(self, **kwargs):
        last_exc = None
        n = len(self._clients)
        start = _mistral_rr_start_index()

        # One pass over all keys per call()
        for i in range(n):
            idx = (start + i) % n
            try:
                return self._clients[idx].chat.complete(**kwargs)
            except SDKError as e:
                last_exc = e
                # if rate-limited on this key, try next key
                if _is_rate_limited_mistral(e):
                    continue
                # other SDK errors are usually "real" (bad request, etc.)
                raise
            except Exception as e:
                last_exc = e
                # transient network-ish errors: try next key
                continue

        # all keys failed (often all 429)
        if last_exc:
            raise last_exc
        raise RuntimeError("No Mistral clients available.")


# ============================
# Provider clients 
# ============================
_MISTRAL_CLIENT = None; _MISTRAL_CLIENT_LOCK = threading.Lock()
_OPENAI_CLIENT = None
_GEMINI_CLIENT = None; _GEMINI_TYPES = None


def _get_mistral_client():
    """Return Mistral client; uses round-robin if multiple keys are provided."""
    global _MISTRAL_CLIENT
    with _MISTRAL_CLIENT_LOCK:
        if _MISTRAL_CLIENT is None:
            if not _MISTRAL_API_KEYS:
                raise ValueError(
                    "No Mistral API keys found. Set MISTRAL_API_KEY (and optionally MISTRAL_API_KEY_1..10)."
                )
            if len(_MISTRAL_API_KEYS) == 1:
                _MISTRAL_CLIENT = Mistral(api_key=_MISTRAL_API_KEYS[0])
            else:
                _MISTRAL_CLIENT = _MistralRoundRobinClient(_MISTRAL_API_KEYS)
    return _MISTRAL_CLIENT


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _OPENAI_CLIENT


def _get_gemini_client():
    global _GEMINI_CLIENT, _GEMINI_TYPES
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        _GEMINI_TYPES = types
    return _GEMINI_CLIENT, _GEMINI_TYPES


# ============================
# Logging helper
# ============================

# Default no-op logger
def _noop_log(msg):
    pass 

# ============================
# Prompt helpers
# ============================

def _pack(prev_responses: List[str]) -> str:
    """Pack previous responses into a single string."""
    return "\n".join([f"{i+1}. {r}" for i, r in enumerate(prev_responses)])


def _synthesize_prompt(prefix: str, prev_responses: List[str]) -> str:
    """Synthesize previous responses into a single system prompt for the next layer."""
    return prefix + _pack(prev_responses)


# ==========================
# Provider calls
# ==========================

def _call_mistral(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client = _get_mistral_client()
    resp = client.chat.complete(
        model=agent["model"],
        messages=[
            {"role": "system", "content": system_prompt or agent.get("system", "")},
            {"role": "user", "content": user_prompt},
        ],
        temperature=agent.get("temperature", 0.7),
        max_tokens=agent.get("max_tokens", 512),
    )
    return resp.choices[0].message.content


def _call_openai(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=agent["model"],
        messages=[
            {"role": "system", "content": system_prompt or agent.get("system", "")},
            {"role": "user", "content": user_prompt},
        ],
        temperature=agent.get("temperature", 0.7),
        max_tokens=agent.get("max_tokens", 512),
    )
    return resp.choices[0].message.content


def _call_gemini(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client, types = _get_gemini_client()

    config = types.GenerateContentConfig(
        system_instruction=(system_prompt or agent.get("system", "")) or None,
        temperature=agent.get("temperature", 0.7),
        max_output_tokens=agent.get("max_tokens", 512),
    )

    resp = client.models.generate_content(
        model=agent["model"],
        contents=user_prompt,
        config=config,
    )
    return resp.text

def call(agent: Dict[str, Any], system_prompt: str, user_prompt: str) -> str:
    """Single call with exponential-backoff retry.

    Agent fields:
      - provider: "mistral" | "openai" | "gemini" (default "mistral")
      - model, temperature, max_tokens
      - retries (default 6), backoff (default 1.0)
    """
    provider = (agent.get("provider") or "mistral").lower()
    max_retries = 6
    base_sleep = 1.0

    for attempt in range(max_retries):
        try:
            if provider == "mistral":
                return _call_mistral(agent, user_prompt, system_prompt)
            if provider == "openai":
                return _call_openai(agent, user_prompt, system_prompt)
            if provider == "gemini":
                return _call_gemini(agent, user_prompt, system_prompt)
            raise ValueError(f"Unknown provider={provider!r}")

        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_s = min(30.0, base_sleep * (2 ** attempt) + random.uniform(0, 0.5))
            time.sleep(sleep_s)

    raise RuntimeError(f"Retry exhausted for provider={provider} model={agent.get('model')}")


# ============================
# MoA Implementation
# ============================

def run_layer(
    agents: List[Dict[str, Any]],
    layer_system: Optional[str],
    user_prompt: str,
) -> List[str]:
    return [call(a, layer_system or a.get("system", ""), user_prompt) for a in agents]

def run_moa(
    user_prompt: str,
    proposer_layers: List[List[Dict[str, Any]]],
    aggregator: Dict[str, Any],
    synthesize_prompt: str = SYNTHESIZE_PROMPT,
    return_intermediate: bool = False,
    log: Any = _noop_log,
) -> str | tuple[str, List[List[str]]]:
    """MoA with arbitrary per-layer, per-agent providers/models."""
    all_layer_results: List[List[str]] = []

    # Layer 1
    results = run_layer(proposer_layers[0], None, user_prompt)
    all_layer_results.append(results)
    
    log(f"\n[INFO] Layer 1 outputs:")
    for agent_index, output in enumerate(results, start=1):
        log(f"Agent {agent_index}: {output}\n")

    # Optional proposer layers 2..k
    for layer_index, layer_agents in enumerate(proposer_layers[1:], start=2):
        log(f"\n[INFO] Layer {layer_index} outputs:")
        layer_system = _synthesize_prompt(synthesize_prompt, results)
        results = run_layer(layer_agents, layer_system, user_prompt)
        all_layer_results.append(results)
        
        for agent_index, output in enumerate(results, start=1):
            log(f"Agent {agent_index}: {output}\n")

    # Final aggregation
    final_system = _synthesize_prompt(synthesize_prompt, results)
    final_decision = call(aggregator, final_system, user_prompt)
    
    log(f"\n[INFO] Final answer:\n{final_decision}\n")
    
    if return_intermediate:
        return final_decision, all_layer_results
    
    return final_decision


__all__ = ["SYNTHESIZE_PROMPT", "run_moa"]


if __name__ == "__main__":
    PROPOSER_LAYERS = [
        [   # Layer 1 
            {"provider": "mistral", "model": "mistral-small-2506", "temperature": 0.7},
            {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.7},
            {"provider": "gemini", "model": "gemini-2.5-flash-lite", "temperature": 0.7},
            {"provider": "mistral", "model": "ministral-14b-2512", "temperature": 0.7},
        ],
    ]
    
    # Layer 2
    AGGREGATOR = {"provider": "mistral", "model": "mistral-large-2512", "temperature": 0.0, "max_tokens": 2048}