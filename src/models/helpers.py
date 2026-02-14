import os
import threading
import time
from dotenv import load_dotenv
from typing import Any, List

from mistralai import Mistral
from openai import OpenAI
from google import genai
from google.genai import types


# -----------------------------------------
# Mistral: Round-robin API key rotation 
# -----------------------------------------

load_dotenv()

# Mistral key pool 
_MISTRAL_API_KEYS = [
	os.getenv("MISTRAL_API_KEY"),
	os.getenv("MISTRAL_API_KEY_1"),
	os.getenv("MISTRAL_API_KEY_2"),
	os.getenv("MISTRAL_API_KEY_3"),
	os.getenv("MISTRAL_API_KEY_4"),
	os.getenv("MISTRAL_API_KEY_5"),
	os.getenv("MISTRAL_API_KEY_6"),
	os.getenv("MISTRAL_API_KEY_7"),
	os.getenv("MISTRAL_API_KEY_8"),
	os.getenv("MISTRAL_API_KEY_9"),
	os.getenv("MISTRAL_API_KEY_10"),
]
_MISTRAL_API_KEYS = [k for k in _MISTRAL_API_KEYS if k]


_mistral_rr_lock = threading.Lock()
_mistral_rr_counter = 0


def _mistral_rr_start_index() -> int:
    """
    Thread-safe round-robin start index.
    """
    global _mistral_rr_counter
    with _mistral_rr_lock:
        if not _MISTRAL_API_KEYS:
            return 0
        idx = _mistral_rr_counter % len(_MISTRAL_API_KEYS)
        _mistral_rr_counter += 1
        return idx


def _is_rate_limited_mistral(e: Exception) -> bool:
    """
    Best-effort 429 detection across Mistral SDK versions.
    """
    s = str(e)
    return ("Status 429" in s) or ('"code":"1300"' in s) or ("rate limit" in s.lower())


class _MistralRRChatProxy:
	"""
	Proxy so callers can keep using client.chat.complete(...).
	"""
	def __init__(self, parent):
		self._parent = parent

	def complete(self, **kwargs):
		return self._parent._complete(**kwargs)


_MISTRAL_MAX_ROTATION_CYCLES = 10
_MISTRAL_COOLDOWN_S = 60

class MistralRoundRobinClient:
	"""
	Round-robin across multiple Mistral API keys; on failure, try next key.
	"""

	def __init__(self, api_keys: List[str]):
		if not api_keys:
			raise ValueError("No Mistral API keys found in env.")
		self._clients = [Mistral(api_key=k) for k in api_keys]
		self.chat = _MistralRRChatProxy(self)

	def _complete(self, **kwargs):
		last_exc = None
		n = len(self._clients)

		for cycle in range(_MISTRAL_MAX_ROTATION_CYCLES):
			start = _mistral_rr_start_index()
			
			all_rate_limited = True
			for i in range(n):
				idx = (start + i) % n
				try:
					return self._clients[idx].chat.complete(**kwargs)
				except Exception as e:
					last_exc = e
					if not _is_rate_limited_mistral(e):
						all_rate_limited = False
			
			if all_rate_limited:
				time.sleep(_MISTRAL_COOLDOWN_S)
				continue
			
			break
				
		if last_exc:
			raise last_exc
		raise RuntimeError("No Mistral clients available.")


# ----------------------------
# Provider Clients 
# ----------------------------

_MISTRAL_CLIENT_SINGLETON = None; _MISTRAL_CLIENT_LOCK = threading.Lock()
_OPENAI_CLIENT = None
_GEMINI_CLIENT = None; _GEMINI_TYPES = None


def get_mistral_client():
    """
    Return a Mistral client; uses round-robin if multiple keys are provided.
    """
    global _MISTRAL_CLIENT_SINGLETON
    with _MISTRAL_CLIENT_LOCK:
        if _MISTRAL_CLIENT_SINGLETON is None:
            if not _MISTRAL_API_KEYS:
                raise ValueError(
                    "No Mistral API keys found. Set MISTRAL_API_KEY (and optionally MISTRAL_API_KEY_1..10)."
                )
            if len(_MISTRAL_API_KEYS) == 1:
                _MISTRAL_CLIENT_SINGLETON = Mistral(api_key=_MISTRAL_API_KEYS[0])
            else:
                _MISTRAL_CLIENT_SINGLETON = MistralRoundRobinClient(_MISTRAL_API_KEYS)
    return _MISTRAL_CLIENT_SINGLETON


def get_openai_client():
	global _OPENAI_CLIENT
	if _OPENAI_CLIENT is None:
		_OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
	return _OPENAI_CLIENT


def get_gemini_client():
	global _GEMINI_CLIENT, _GEMINI_TYPES
	if _GEMINI_CLIENT is None:
		_GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
		_GEMINI_TYPES = types
	return _GEMINI_CLIENT, _GEMINI_TYPES


# ----------------------------
# Logging Placeholder
# ----------------------------

def noop_log(msg: str | Any) -> None:
	return None


# ------------------------------------------
# Remove "thinking" content from responses
# ------------------------------------------

def remove_mistral_thinking(content: Any) -> str:
	if content is None:
		return ""
	if isinstance(content, str):
		return content

	if isinstance(content, list):
		parts = []
		for item in content:
			if isinstance(item, dict):
				if item.get("type") == "thinking":
					continue
				text = item.get("text") or item.get("content") or ""
				if text:
					parts.append(text)
				continue
			if isinstance(item, str):
				parts.append(item)
				continue
			if getattr(item, "type", None) == "thinking":
				continue
			text = getattr(item, "text", None)
			if text is not None:
				parts.append(text)
			else:
				parts.append(str(item))
		return "".join(parts)

	text = getattr(content, "text", None)
	if text is not None:
		return text
	return str(content)


# -------------------------------
# API Call Tracker
# -------------------------------

class SampleAPICallTracker:
    """
    Track API calls per sample by registering Agent instances created and
    raw calls for that sample and summing their per-instance counters. 
    This avoids cross-thread contamination from a global total.
    """
    def __init__(self):
        self._agents = []
        self._lock = threading.Lock()
        self._raw_calls = 0

    def register_agent(self, agent):
        if agent is None:
            return
        with self._lock:
            self._agents.append(agent)

    register = register_agent

    def total_calls(self):
        with self._lock:
            return sum(getattr(a, 'api_calls', 0) for a in self._agents)

    def breakdown(self):
        with self._lock:
            return [(getattr(a, 'role', 'unknown'), getattr(a, 'api_calls', 0)) for a in self._agents]

    def register_call(self, n: int = 1):
        """Register raw API calls that are not associated with Agent instances (e.g., investigation layer)."""
        with self._lock:
            try:
                self._raw_calls += int(n)
            except Exception:
                self._raw_calls += 1

    def raw_calls(self) -> int:
        with self._lock:
            return int(self._raw_calls)

    def total_calls(self):
        """Total calls including registered Agent instances and raw calls."""
        with self._lock:
            return sum(getattr(a, 'api_calls', 0) for a in self._agents) + int(self._raw_calls)


__all__ = [
    "get_mistral_client",
    "get_openai_client",
    "get_gemini_client",
    "noop_log",
    "remove_mistral_thinking",
    "SampleAPICallTracker",    
]

