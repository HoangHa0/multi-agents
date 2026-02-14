import time
import random
from typing import Optional, List, Dict, Any

from src.models.helpers import (
    get_mistral_client,
    get_openai_client,
    get_gemini_client,
    noop_log,
    remove_mistral_thinking,
)
from src.prompts.system_prompts import (
    ELIMINATION_STRATEGIST,
    FORWARD_CHAINING_CLINICIAN, 
    MECHANISM_AUDITOR, 
    GUIDELINE_CLINICIAN, 
    SKEPTIC_CONTRARIAN
)
from src.models.debate import process_query


# ============================
# Prompts
# ============================

DEBATE_REQUIREMENT_PROMPT = f"""
You are about to enter a multidisciplinary medical debate with other expert agents.

You are given:
1) The medical question.
2) INVESTIGATION OUTPUTS from 5 investigator agents.

How to use INVESTIGATION OUTPUTS:
- Treat them as a reference only (not ground truth) to come up with your own opinion and build your own reasonings. 

Core objectives:
- Actively surface and resolve disagreements.
- Stress-test the leading option against at least one strong alternative.
- Converge on the most defensible option by evidence and mechanism, not by authority.

Strict interaction rules (to match the debate workflow):
A) When asked whether you want to talk to any expert:
- Output EXACTLY one token on the first line: YES or NO

B) When asked to pick which agent(s) to talk to:
- Output ONLY agent numbers separated by commas (e.g., 1 or 1,3). No other words.

C) When you send a message to another agent:
- Keep it <= 4 lines.
- Include: (1) your current option letter, (2) key supports, (3) key refutes of their likely option.

D) When asked for “your answer” / “your final answer” at any time:
- Use the Answer Card format exactly.

Answer Card format (required):
Answer: <A-E>
Confidence: <0.00-1.00>
Key rationales: <your evidence-based rationales for your answer; why not other options?>
If updated (only when you change):
- Previous answer: <your previous option letter>
- Update reason: <why your opinion changed>

Quality guardrails:
- No invented patient facts.
- Prefer discriminators: timing, population, contraindication, mechanism, “missing hallmark feature”.
- If you are unsure, lower confidence and explicitly name what would disambiguate.
"""

# ==========================
# Provider calls
# ==========================

def _call_mistral(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client = get_mistral_client()
    resp = client.chat.complete(
        model=agent["model"],
        messages=[
            {"role": "system", "content": system_prompt or agent.get("system", "")},
            {"role": "user", "content": user_prompt},
        ],
        temperature=agent.get("temperature", 0.7),
    )
    
    return remove_mistral_thinking(resp.choices[0].message.content)

def _call_openai(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=agent["model"],
        messages=[
            {"role": "system", "content": system_prompt or agent.get("system", "")},
            {"role": "user", "content": user_prompt},
        ],
        temperature=agent.get("temperature", 0.7),
    )
    return resp.choices[0].message.content


def _call_gemini(agent: Dict[str, Any], user_prompt: str, system_prompt: str) -> str:
    client, types = get_gemini_client()

    config = types.GenerateContentConfig(
        system_instruction=(system_prompt or agent.get("system", "")) or None,
        temperature=agent.get("temperature", 0.7),
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
      - model, temperature
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
# Minor helper
# ============================
def _pack(prev_responses: List[str]) -> str:
    """
    Pack previous responses into a single string.
    """
    return "\n".join([f"Agent {i+1}. {r}" for i, r in enumerate(prev_responses)])


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
    question: str,
    proposer_layers: List[List[Dict[str, Any]]],
    aggregators: List[Dict[str, Any]],
    debate_prompt: str = DEBATE_REQUIREMENT_PROMPT,
    return_intermediate: bool = False,
    log: Any = noop_log,
) -> str | tuple[str, List[List[str]]]:
    """
    MoA with customized per-layer, per-agent providers/models.
    """
    all_layer_results: List[List[str]] = []

    # Layer 1 - Investigation
    results = run_layer(proposer_layers[0], None, question)
    all_layer_results.append(results)
    
    log(f"\n[INFO] LAYER 1: INVESTIGATION")
    for agent_index, output in enumerate(results, start=1):
        log(f"\nAgent {agent_index}: {output}")

    # Layer 2 - Debate and Synthesis
    log(f"\n\n[INFO] LAYER 2: DEBATE & AGGREGATION")
    investigation_output = _pack(results)
    user_query = f"[DEBATE REQUIREMENTS]\n{debate_prompt}\n[INVESTIGATION OUTPUTS]:\n{investigation_output}"
    final_decision = process_query(
        question=question,
        aggregators=aggregators, 
        user_query=user_query,
        log=log
    )
    
    log(f"\n[INFO] Final answer:\n{final_decision}\n")
    
    if return_intermediate:
        return final_decision, all_layer_results
    
    return final_decision


__all__ = ["DEBATE_REQUIREMENT_PROMPT", "run_moa"]


if __name__ == "__main__":
    PROPOSER_LAYERS = [
        [   # Layer 1 
            {"provider": "mistral", "model": "mistral-small-2506", "temperature": 0.7, "system": ELIMINATION_STRATEGIST},
            {"provider": "mistral", "model": "mistral-medium-2508", "temperature": 0.7, "system": FORWARD_CHAINING_CLINICIAN},
            {"provider": "mistral", "model": "ministral-3b-2512", "temperature": 0.7, "system": MECHANISM_AUDITOR},
            {"provider": "mistral", "model": "ministral-8b-2512", "temperature": 0.7, "system": GUIDELINE_CLINICIAN},
            {"provider": "mistral", "model": "magistral-small-2509", "temperature": 0.7, "system": SKEPTIC_CONTRARIAN},
        ],
    ]
    
    # Layer 2
    AGGREGATORS = [
        {"provider": "mistral", "model": "mistral-large-2512"},
        {"provider": "mistral", "model": "ministral-14b-2512"},
        {"provider": "mistral", "model": "magistral-medium-2509"},
    ]