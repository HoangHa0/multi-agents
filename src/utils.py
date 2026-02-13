import os
import re
import time
import json
import random
import threading
from dotenv import load_dotenv
from prettytable import PrettyTable 
from google import genai
from google.genai import types
from openai import OpenAI
from mistralai import Mistral, SystemMessage, UserMessage, AssistantMessage

# -----------------------------------------
# Mistral: round-robin API key rotation 
# -----------------------------------------

load_dotenv()

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

_mistral_rr_counter = [0]
_mistral_rr_lock = threading.Lock()
_MISTRAL_MAX_RETRIES = 0
_MISTRAL_RETRY_SLEEP_S = 1.0


def _mistral_rr_start_index() -> int:
    """Thread-safe round-robin start index."""
    with _mistral_rr_lock:
        if not _MISTRAL_API_KEYS:
            return 0
        idx = _mistral_rr_counter[0] % len(_MISTRAL_API_KEYS)
        _mistral_rr_counter[0] += 1
        return idx


class _MistralRRChatProxy:
    """Proxy so callers can keep using client.chat.complete(...)."""

    def __init__(self, parent):
        self._parent = parent

    def complete(self, **kwargs):
        return self._parent._complete(**kwargs)


class _MistralRoundRobinClient:
    """Round-robin across multiple Mistral API keys; on failure, try next key."""

    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError(
                "No Mistral API keys found. Set MISTRAL_API_KEY (and optionally MISTRAL_API_KEY_1..MISTRAL_API_KEY_10)."
            )
        self._clients = [Mistral(api_key=k) for k in api_keys]
        self.chat = _MistralRRChatProxy(self)

    def _complete(self, **kwargs):
        last_exc = None
        attempts = 0
        n = len(self._clients)

        while True:
            start = _mistral_rr_start_index()
            for i in range(n):
                idx = (start + i) % n
                try:
                    return self._clients[idx].chat.complete(**kwargs)
                except Exception as e:
                    last_exc = e
                    continue

            attempts += 1
            if _MISTRAL_MAX_RETRIES > 0 and attempts >= _MISTRAL_MAX_RETRIES:
                break
            time.sleep(_MISTRAL_RETRY_SLEEP_S)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Mistral round-robin client has no available clients.")


_MISTRAL_RR_CLIENT_SINGLETON = None


def _get_mistral_client():
    """Return a Mistral client (demux to round-robin if multiple keys exist)."""
    global _MISTRAL_RR_CLIENT_SINGLETON
    if _MISTRAL_RR_CLIENT_SINGLETON is None:
        if not _MISTRAL_API_KEYS:
            raise ValueError(
                "No Mistral API keys found. Set MISTRAL_API_KEY (and optionally MISTRAL_API_KEY_1..MISTRAL_API_KEY_10)."
            )
        # Preserve original behavior when only one key is provided.
        if len(_MISTRAL_API_KEYS) == 1:
            _MISTRAL_RR_CLIENT_SINGLETON = Mistral(api_key=_MISTRAL_API_KEYS[0])
        else:
            _MISTRAL_RR_CLIENT_SINGLETON = _MistralRoundRobinClient(_MISTRAL_API_KEYS)
    return _MISTRAL_RR_CLIENT_SINGLETON


# -----------------------
# Robustness helpers
# -----------------------

# Default no-op logger
def _noop_log(msg):
    pass 

# Model distribution helper
def distribute_models(model_list, num_agents):
    """
    Distribute models equally among agents.
    
    Args:
        model_list: List of model names (e.g., ['gpt-4o-mini', 'gemini-2.5-flash-lite', 'gemini-2.5-pro'])
        num_agents: Number of agents to distribute models to
    
    Returns:
        List of model names for each agent, distributed as evenly as possible
        
    Examples:
        - If models=['A', 'B', 'C'] and num_agents=3, returns ['A', 'B', 'C']
        - If models=['A', 'B', 'C'] and num_agents=5, returns ['A', 'B', 'C', 'A', 'B']
        - If models=['A'] and num_agents=3, returns ['A', 'A', 'A']
        - If models=['A', 'B'] and num_agents=4, returns ['A', 'B', 'A', 'B']
    """
    if not model_list:
        return ['gpt-4o-mini'] * num_agents  # Fallback to default
    
    if len(model_list) == 1:
        # If only one model, use it for all agents
        return [model_list[0]] * num_agents
    
    # Distribute models in round-robin fashion for balanced distribution
    distributed_models = []
    for i in range(num_agents):
        distributed_models.append(model_list[i % len(model_list)])
    
    return distributed_models

# Robust regex helpers 
_EXPERT_LINE_RE = re.compile(r'^\s*(?P<expert>.+?)\s*$', re.IGNORECASE)
_EXPERT_ROLE_DESC_RE = re.compile(r'^\s*(?:\d+\.\s*)?(?P<role>.+?)(?:\s*-\s*(?P<desc>.+))?\s*$')

# Retry helper
DEFAULT_LLM_RETRIES = 5

def _retry_call(name, fn, max_tries=None, retry_exceptions=(IndexError,), sleep_s=None, log=None):
    if log is None:
        log = _noop_log
    """
    Retry a callable as the last resort when the LLM output (or downstream parsing) is brittle.
    This is intentionally lightweight so we don't change any parsing logic; we just re-ask.
    """
    if max_tries is None:
        max_tries = DEFAULT_LLM_RETRIES
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            last_err = e
            log(f"[WARN] {name} failed (attempt {attempt}/{max_tries}): {type(e).__name__}: {e}. Retrying...")
            if sleep_s:
                try:
                    import time
                    time.sleep(sleep_s)
                except Exception:
                    pass
        except Exception as e:
            # Also retry on generic transient failures (API hiccups, etc.)
            last_err = e
            log(f"[WARN] {name} failed (attempt {attempt}/{max_tries}): {type(e).__name__}: {e}. Retrying...")
            if sleep_s:
                try: 
                    import time
                    time.sleep(sleep_s)
                except Exception:
                    pass
    # If still failing, raise the last error so caller can persist progress.
    raise last_err

# API Call Tracker
class SampleAPICallTracker:
    """
    Track API calls per sample by registering Agent instances created for
    that sample and summing their per-instance counters. This avoids
    cross-thread contamination from a global total.
    """
    def __init__(self):
        self._agents = []
        self._lock = threading.Lock()

    def register_agent(self, agent):
        if agent is None:
            return
        with self._lock:
            self._agents.append(agent)

    # Back-compat alias
    register = register_agent

    def total_calls(self):
        with self._lock:
            return sum(getattr(a, 'api_calls', 0) for a in self._agents)

    def breakdown(self):
        with self._lock:
            return [(getattr(a, 'role', 'unknown'), getattr(a, 'api_calls', 0)) for a in self._agents]

class Agent:
    # Class-level counter for total API calls across all agents
    total_api_calls = 0
    _api_calls_lock = threading.Lock()  # Thread-safe lock for API call counting
    
    def __init__(self, instruction, role, examplers=None, model_info='mistral-large-2512', img_path=None, tracker=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.api_calls = 0  # Instance-level counter
        self._tracker = tracker
        if self._tracker is not None:
            try:
                self._tracker.register_agent(self)
            except Exception:
                pass

        if self.model_info in ['gemini-2.5-flash-lite', 'gemini-2.5-pro']:
            self.client = genai.Client(api_key=os.environ['GENAI_API_KEY'])
            self.messages = []
            
            # Map examplers to Gemini history format
            if examplers:
                for exampler in examplers:
                    self.messages.append(types.Content(role="user", parts=[types.Part(text=exampler['question'])])) 
                    reason_prefix = f"Let's think step by step. {exampler['reason']} " if 'reason' in exampler else ""
                    self.messages.append(types.Content(role="model", parts=[types.Part(text=reason_prefix + exampler['answer'])]))
        
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content":  ("Let's think step by step. " + exampler['reason'] + " "  if 'reason' in exampler else '') + exampler['answer']})
                    
        elif self.model_info in ['mistral-large-2512', 'mistral-small-2506', 'ministral-14b-2512', 'ministral-8b-2512', 'ministral-3b-2512']:
            # Uses round-robin across multiple keys if configured.
            self.client = _get_mistral_client()
            self.messages = [
                SystemMessage(content=instruction)
            ]
            if examplers:
                for exampler in examplers:
                    self.messages.append(UserMessage(content=exampler['question']))
                    self.messages.append(AssistantMessage(content=("Let's think step by step. " + exampler['reason'] + " "  if 'reason' in exampler else '') + exampler['answer']))

        # log(f"[DEBUG] Print out the messages for Agent {self.messages}")

    def chat(self, message, img_path=None):
        self.messages.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if self.model_info in ['gemini-2.5-flash-lite', 'gemini-2.5-pro']:
            for attempt in range(10):
                try:
                    # Initialize persistent chat session
                    self._chat = self.client.chats.create(
                        model=self.model_info,
                        history=self.messages[:-1],  # Exclude latest user message for history
                        config=types.GenerateContentConfig(system_instruction=self.instruction)
                    )
                    response = self._chat.send_message(message=message)
                    
                    # Track API call (thread-safe)
                    with Agent._api_calls_lock:
                        self.api_calls += 1
                        Agent.total_api_calls += 1
                                                
                    self.messages.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
                    return response.text
                except Exception as e:
                    print(f"Retrying due to: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )
            
            # Track API call (thread-safe)
            with Agent._api_calls_lock:
                self.api_calls += 1
                Agent.total_api_calls += 1

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content
        
        elif self.model_info in ['mistral-large-2512', 'mistral-small-2506', 'ministral-14b-2512', 'ministral-8b-2512', 'ministral-3b-2512']:            
            self.messages.append(UserMessage(content=message))
            for attempt in range(3):
                try:                    
                    response = self.client.chat.complete(
                        model=self.model_info,
                        messages=self.messages
                    )
                    
                    # Track API call (thread-safe)
                    with Agent._api_calls_lock:
                        self.api_calls += 1
                        Agent.total_api_calls += 1
                    
                    self.messages.append(AssistantMessage(content=response.choices[0].message.content))
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Retrying due to: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                        
    def temp_responses(self, message, temperatures=[0.0], img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = list(set(temperatures))
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                # Track API call (thread-safe)
                with Agent._api_calls_lock:
                    self.api_calls += 1
                    Agent.total_api_calls += 1
                
                responses[temperature] = response.choices[0].message.content
                
            # self.messages.append({"role": "assistant", "content": responses})
                
            return responses
        
        elif self.model_info in ['gemini-2.5-flash-lite', 'gemini-2.5-pro']:
            self.messages.append(types.Content(role="user", parts=[types.Part(text=message)]))
            temperatures = list(set(temperatures))
            responses = {}
                        
            for temperature in temperatures:
                for attempt in range(3):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_info,
                            contents=self.messages,
                            config=types.GenerateContentConfig(temperature=temperature)
                        )
                        
                        with Agent._api_calls_lock:
                            self.api_calls += 1
                            Agent.total_api_calls += 1
                        
                        responses[temperature] = response.text
                        break
                    except Exception as e:
                        print(f"Retrying due to: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                                
            # OPTIONAL: If you want to "pick" one to actually save to history, 
            # you would call self._chat.send_message(message) once at the end.
            # self.messages.append(types.Content(role="model", parts=[types.Part(text=responses)]))
            return responses
        
        elif self.model_info in ['mistral-large-2512', 'mistral-small-2506', 'ministral-14b-2512', 'ministral-8b-2512', 'ministral-3b-2512']:
            self.messages.append(UserMessage(content=message))
            temperatures = list(set(temperatures))
            responses = {}
                        
            for temperature in temperatures:
                for attempt in range(3):
                    try:
                        response = self.client.chat.complete(
                            model=self.model_info,
                            messages=self.messages,
                            temperature=temperature,
                        )
                        
                        with Agent._api_calls_lock:
                            self.api_calls += 1
                            Agent.total_api_calls += 1
                        
                        responses[temperature] = response.choices[0].message.content
                        break
                    except Exception as e:
                        print(f"Retrying due to: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                                          
            # OPTIONAL: If you want to "pick" one to actually save to history, 
            # you would call self._chat.send_message(message) once at the end.
            # self.messages.append(AssistantMessage(content=responses))
            return responses
        
    def agent_talk(self, message, recipient, img_path=None):
        """
        Generates a message from this agent (self) and injects it into the recipient's context.
        """
        content = self.chat(message, img_path=img_path)

        incoming_msg = f"Message from {self.role}: {content}"

        if recipient.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            recipient.messages.append({"role": "user", "content": incoming_msg})
        
        elif recipient.model_info in ['gemini-2.5-flash-lite', 'gemini-2.5-pro']:
            recipient.messages.append(types.Content(role="user", parts=[types.Part(text=incoming_msg)]))
            
        elif recipient.model_info in ['mistral-large-2512', 'mistral-small-2506', 'ministral-14b-2512', 'ministral-8b-2512', 'ministral-3b-2512']:
            recipient.messages.append(UserMessage(content=incoming_msg))
        
        return content

    @classmethod
    def get_total_api_calls(cls):
        """Get total API calls across all agents (thread-safe)."""
        with cls._api_calls_lock:
            return cls.total_api_calls
    
    @classmethod
    def reset_total_api_calls(cls):
        """Reset total API call counter (thread-safe)."""
        with cls._api_calls_lock:
            cls.total_api_calls = 0
    
    def get_api_calls(self):
        """Get API calls for this agent instance."""
        return self.api_calls
    
    def reset_api_calls(self):
        """Reset API call counter for this agent instance."""
        self.api_calls = 0
    

def parse_group_info(group_info):
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    group_goal_match = re.search(r'(Group\s+\d+\s*-?\s*|###\s*Group\s+\d+\s*-?\s*)([^\n]+)', group_info, re.IGNORECASE)
    if group_goal_match:
        goal_full = group_goal_match.group(2).strip()
        goal_full = re.sub(r'Group\s+\d+\s*-\s*', '', goal_full, flags=re.IGNORECASE).strip()
        parsed_info['group_goal'] = goal_full
    
    # Parse members line by line - more robust for various LLM output formats
    # Handles: "Member N: Role - Description", "Member N: **Role** - **Description**", etc.
    lines = group_info.split('\n')
    for line in lines:
        # Match "Member N:" at the start of line (with optional markdown bullets/dashes)
        member_match = re.match(r'^\s*[-*]*\s*\*?\*?Member\s+\d+:\s*(.+)', line, re.IGNORECASE)
        if member_match:
            content = member_match.group(1).strip()
            
            # Remove all markdown asterisks first
            content = re.sub(r'\*+', '', content)
            
            # Split by dash to separate role from description
            if ' - ' in content:
                parts = content.split(' - ', 1)
                role_part = parts[0].strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ''
            else:
                role_part = content.strip()
                desc_part = ''
            
            # Remove parenthetical notes like "(Lead)" from role
            role_clean = re.sub(r'\s*\([^)]*\)\s*', ' ', role_part).strip()
            
            if role_clean:
                parsed_info['members'].append({
                    'role': role_clean,
                    'expertise_description': desc_part
                })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        client = genai.Client(api_key=os.environ['genai_api_key'])
        return client, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None


def process_query(question, args, log=None, tracker=None):
    """
    Intermediate (MDT) setting:
      - Recruit N experts 
      - Collect initial opinions
      - For each round:
          * participatory debate (agents optionally message each other)
          * agents update final answers for the round
          * consensus check; if not reached, continue to next round
      - Final decision maker reviews all agent answers and produces the final answer
    """
    if log is None:
        log = _noop_log
    
    created_tracker = False
    if tracker is None:
        tracker = SampleAPICallTracker()
        created_tracker = True
    
    # Create moderator if not provided (when difficulty is not 'adaptive')

    moderator = Agent(
        instruction='You are a medical expert who conducts initial assessment and moderates the discussion.',
        role='moderator',
        # model_info='gpt-4o-mini',
        tracker=tracker
    )
    
    log("\n[INFO] Step 1. Expert Recruitment")

    num_agents = 3 # You can adjust this number as needed

    def _recruit_and_parse_intermediate():
        recruiter = Agent(instruction="You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query.", 
                role='recruiter', 
                # model_info='gpt-4o-mini',
                tracker=tracker)
        recruited = recruiter.chat(
            f"Question: {question}\n"
            f"You can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, "
            "what kind of experts will you recruit to better make an accurate answer?\n"
            "For example, if you want to recruit five experts, you answer can be like:\n"
            "1. Pediatrician - Specializes in the medical care of infants, children, and adolescents.\n"
            "2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions.\n"
            "3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders.\n"
            "4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth.\n"
            "5. Medical Geneticist - Specializes in the study of genes and heredity.\n\n"
            "Please answer in above format, and do not include your reason."
        )
        # log(f"[DEBUG] Recruited Experts:\n{recruited}")

        agents_data = []
        for _line in re.findall(r'[^\n]+', recruited or ''):
            _line = _line.strip()
            if not _line:
                continue
            m = _EXPERT_LINE_RE.match(_line)
            expert_txt = (m.group('expert') if m else _line).strip()
            agents_data.append(expert_txt)

        if not agents_data:
            raise IndexError('No experts parsed from recruitment output')

        # Keep only the requested number of experts (LLM may output extra lines)
        agents_data = agents_data[:num_agents]
        return agents_data

    agents_data = _retry_call('intermediate_recruitment', _recruit_and_parse_intermediate, log=log)

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent or '')
        agent_role = (m.group('role') if m else (agent or '')).strip().lower()
        description = ((m.group('desc') or '') if m else '').strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    # Distribute models equally among agents
    agent_models = distribute_models(args.model, len(agents_data))

    agent_dict = {}
    medical_agents = []
    for idx, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent or '')
        agent_role = (m.group('role') if m else (agent or '')).strip().lower()
        description = ((m.group('desc') or '') if m else '').strip().lower()

        inst_prompt = f"You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=agent_models[idx], tracker=tracker)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent or '')
        role_txt = (m.group('role') if m else (agent or '')).strip()
        desc_txt = ((m.group('desc') or '') if m else '').strip()
        if desc_txt:
            log(f"Agent {idx+1} ({agent_emoji[idx]} {role_txt}): {desc_txt}")
        else:
            log(f"Agent {idx+1} ({agent_emoji[idx]}): {role_txt}")
            

    # Moderator (consensus checking)
    moderator_prompt = (
        "You are now a moderator in a multidisciplinary medical team discussion. "
        "Your job is to moderate the discussion, check whether the team has reached consensus, "
        "toward a correct and consistent final answer."
    )
    moderator.chat(moderator_prompt)

    num_rounds = 5
    num_turns = 5

    # Logs
    interaction_log = {} # interaction_log[round_name][turn_name]['Agent i']['Agent j'] = message
    feedback_log = {} # feedback_log[round_name][role] = feedback

    # Helper function to print summary table
    def _print_summary_table(turn_data, n_agents):
        """Print the interaction summary table showing which agents communicated.
        Args:
            turn_data: Dictionary with structure {'Agent i': {'Agent j': msg, ...}, ...}
            n_agents: Number of agents
        """
        header = [""] + [f"Agent {i+1}" for i in range(n_agents)]
        myTable = PrettyTable(header)

        matrix = [[False] * n_agents for _ in range(n_agents)]
        # Process the turn data directly
        for src, dsts in turn_data.items():
            msrc = re.search(r'Agent\s+(\d+)', src)
            if not msrc:
                continue
            i = int(msrc.group(1)) - 1
            for dst in dsts.keys():
                mdst = re.search(r'Agent\s+(\d+)', dst)
                if not mdst:
                    continue
                j = int(mdst.group(1)) - 1
                if 0 <= i < n_agents and 0 <= j < n_agents and i != j:
                    matrix[i][j] = True

        for i in range(1, n_agents + 1):
            row = [f"Agent {i}"]
            for j in range(1, n_agents + 1):
                if i == j:
                    row.append(" ")
                    continue
                i2j = matrix[i-1][j-1]
                j2i = matrix[j-1][i-1]
                if not i2j and not j2i:
                    row.append(" ")
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                else:
                    row.append(f'\u270B ({i}<->{j})')
            myTable.add_row(row)
            if i != n_agents:
                myTable.add_row(['' for _ in range(n_agents + 1)])

        log(f'{myTable}\n')

    # Initial opinions
    log("[INFO] Step 2. Initial Opinions")
    opinions = {}
    for idx, agent in enumerate(medical_agents):
        prompt = (
            f"Please return your answer to the medical query among the option provided.\n\n"
            f"Question: {question}\n\n"
        )
        resp = agent.chat(prompt, img_path=None)
        opinions[agent.role] = resp
        # Print like the original: include agent number + emoji + role
        log(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}) : {resp}")

    # Collaborative decision making rounds
    log("\n[INFO] Step 3. Collaborative Decision Making")
    final_answers = dict(opinions)

    for r in range(1, num_rounds + 1):
        round_name = f"Round {r}"
        interaction_log.setdefault(round_name, {})
        feedback_log.setdefault(round_name, {})

        log(f"== {round_name} ==")

        assessment = "".join(f"({k}): {v}\n" for k, v in opinions.items())

        # Participatory debate (T turns)
        log("[INFO] Participatory Debate")
        
        num_yes_total = 0
        for t in range(1, num_turns + 1):
            turn_name = f"Turn {t}"
            interaction_log[round_name].setdefault(turn_name, {})
            log(f"|_{turn_name}")

            num_yes = 0
            for idx, agent in enumerate(medical_agents):
                participate = agent.chat(
                    "Given the opinions from other medical agents, indicate whether you want to talk to any expert (yes/no). "
                    "If not, provide your opinion.\n\n"
                    f"Opinions:\n{assessment}",
                    img_path=None
                )

                if re.search(r'(?i)\byes\b', (participate or "").strip()):                    
                    # Build filtered agent list (not including self)
                    filtered_agent_list = ""
                    for i, agent_data in enumerate(agents_data):
                        if i == idx:  # Skip the current agent
                            continue
                        m = _EXPERT_ROLE_DESC_RE.match(agent_data or '')
                        agent_role = (m.group('role') if m else (agent_data or '')).strip().lower()
                        description = ((m.group('desc') or '') if m else '').strip().lower()
                        filtered_agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
                    
                    chosen_expert = agent.chat(
                        "Next, indicate the agent(s) you want to talk to.\n"
                        f"{filtered_agent_list}\n"
                        "Return ONLY the number(s), e.g., 1 or 1,2. Do not include reasons.",
                        img_path=None
                    )
                    chosen_experts = [int(ce) for ce in re.split(r'[^0-9]+', chosen_expert or '') if ce.strip().isdigit()]
                    chosen_experts = [ce for ce in chosen_experts if 1 <= ce <= len(medical_agents)-1 and ce != (idx + 1)]  # valid and not self
                    chosen_experts = list(dict.fromkeys(chosen_experts))  # unique, preserve order

                    for ce in chosen_experts:
                        recipient_agent = medical_agents[ce-1]
                        msg = agent.agent_talk(
                            "Remind your medical expertise and leave your opinion to the expert you chose. "
                            "Deliver your opinion once you are confident and in a way to convince the other expert with a short reason.\n\n"
                            f"Question:\n{question}",
                            recipient=recipient_agent,
                            img_path=None
                        )

                        # Print + log 
                        log(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {msg}")
                        interaction_log[round_name][turn_name].setdefault(f"Agent {idx+1}", {})
                        interaction_log[round_name][turn_name][f"Agent {idx+1}"][f"Agent {ce}"] = msg

                    num_yes += 1
                    num_yes_total += 1
                else:
                    # "no" path: store updated opinion for the next turn/round
                    if participate and participate.strip():
                        opinions[agent.role] = participate
                        assessment = "".join(f"({k}): {v}\n" for k, v in opinions.items())
                    log(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): \U0001F910")
                
            log(f"\n[DEBUG] Current agent chat history for {round_name}, {turn_name}:\n" + 
                "\n".join([f"{idx}. {agent.role} history:\n{agent.messages}" for idx, agent in enumerate(medical_agents)]))
                
            if num_yes == 0:
                log(" No agents chose to participate in this turn. End this turn.")
                break

            # Print summary table for this turn only
            log("\n[INFO] Summary Table")
            _print_summary_table(interaction_log[round_name][turn_name], len(medical_agents))

        if num_yes_total == 0:
            log(" No agents chose to participate in this round. End this round.")
            break
        
        # Agents update final answers for this round
        tmp_final_answer = {}
        for agent in medical_agents:
            response = agent.chat(
                "Now that you've interacted with other medical experts (and received moderator feedback if any), "
                "remind your expertise and the comments from other experts and make your final answer to the given question.\n"
                f"Question: {question}\n\n",
                img_path=None
            )
            tmp_final_answer[agent.role] = response

        final_answers = tmp_final_answer

        # Moderator consensus check (moderator decides if another round is needed)
        log("\n[INFO] Moderator Consensus Check")
        answers_text = "".join(f"[{role}] {ans}\n" for role, ans in final_answers.items())
        moderator_consensus = moderator.chat(
            "You are moderating the team. Decide whether the team has reached consensus on the final option.\n"
            "Consensus means the experts' final answers point to the same option letter, or are clearly aligned.\n\n"
            "Return EXACTLY one of the following (and nothing else):\n"
            "Consensus: YES\n"
            "Consensus: NO\n\n"
            f"Question:\n{question}\n\n"
            f"Experts' current answers:\n{answers_text}\n",
            img_path=None
        )
        
        log(f" \U0001F468\u200D\u2696\uFE0F Moderator chat history for {round_name}:\n{moderator.messages}")
                
        consensus_yes = bool(re.search(r'(?im)^\s*Consensus\s*:\s*YES\s*$', moderator_consensus or ''))
        consensus_no = bool(re.search(r'(?im)^\s*Consensus\s*:\s*NO\s*$', moderator_consensus or ''))

        # If the moderator response is malformed, default to NO (continue refining)
        if not (consensus_yes or consensus_no):
            consensus_yes = False
            consensus_no = True

        log(f" \U0001F468\u200D\u2696\uFE0F Moderator consensus check: {'YES' if consensus_yes else 'NO'}")

        if consensus_yes:
            log("\n[INFO] Consensus reached! Ending discussion.")
            break

        # Early stopping mechanism
        # Check if agents agree to continue
        log("\n[INFO] Vote to continue discussion")
        continue_votes = 0
        for agent in medical_agents:
            vote = agent.chat(
                "The moderator has determined that consensus has not been reached yet. "
                "Do you believe further discussion is necessary to reach a conclusion? "
                "Return 'YES' to continue or 'NO' to stop.",
                img_path=None
            )
            # Simple check for YES or NO
            if re.search(r'(?i)\byes\b', vote):
                continue_votes += 1
            log(f" Agent {agent.role}: {'YES' if re.search(r'(?i)\byes\b', vote) else 'NO'}")

        # If majority say NO, then we stop regardless of consensus
        if continue_votes <= len(medical_agents) // 2:
            log("\n[INFO] Agents voted to stop discussion.")
            break
        
        log(f"\n[DEBUG] Current agent chat history for {round_name}:\n" + 
            "\n".join([f"{idx}. {agent.role} history:\n{agent.messages}" for idx, agent in enumerate(medical_agents)]))

        # Moderator provides feedback for next round if not converged
        log("\n[INFO] Disagreement detected")

        # Next round starts from the agents' last answers
        opinions = dict(final_answers)
        
        log(f"\n[DEBUG] End of {round_name} chat opinions:\n" +
            "\n".join([f"{idx}. {agent.role} opinion:\n{opinions[agent.role]}" for idx, agent in enumerate(medical_agents)]))

    # Final decision maker (review all opinions)
    log("\n[INFO] Step 4. Final Decision")

    decision_maker = Agent(
        "You are a final medical decision maker who reviews all opinions from different medical experts and their conversation history to make the final decision.",
        role='decision maker',
        tracker=tracker
    )
    
    answers_text = "".join(f"[{role}] {ans}\n" for role, ans in final_answers.items())
    
    # Build full conversation history for decision maker
    conversation_history = ""
    for round_name, round_data in interaction_log.items():
        conversation_history += f"\n=== {round_name} ===\n"
        for turn_name, turn_data in round_data.items():
            conversation_history += f"\n{turn_name}:\n"
            for src, dsts in turn_data.items():
                for dst, msg in dsts.items():
                    conversation_history += f"  {src} → {dst}:\n    {msg}\n"
    
    log("\n[DEBUG] Full Conversation History for Decision Maker:\n" + conversation_history)
    
    final_decision = decision_maker.temp_responses(
        "You are reviewing the final decision from a multidisciplinary team discussion. "
        "Consider the experts' reasoning, the conversation history showing how they interacted and converged (or disagreed), "
        "and their final answers to make an informed final decision.\n\n"
        f"Question:\n{question}\n\n"
        f"Conversation History:\n{conversation_history if conversation_history.strip() else '(No direct interactions occurred)'}\n\n"
        f"Experts' Final Answers:\n{answers_text}\n"
        "Based on the conversation history and final answers, please make the final answer to the question by considering consensus and reasoning quality:\n"
        "Answer: ",
        temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0],
        img_path=None 
    )
    
    if not final_decision:
        log("[WARN] Final decision contains error or is empty, retrying with reduced context...")
        final_decision = decision_maker.temp_responses(
            "You are reviewing the final decision from a multidisciplinary team discussion. "
            "Consider the experts' reasoning, the conversation history showing how they interacted and converged (or disagreed), "
            "and their final answers to make an informed final decision.\n\n"
            f"Question:\n{question}\n\n"
            f"Conversation History:\n{conversation_history[-6000:] if conversation_history.strip() else '(No direct interactions occurred)'}\n\n"
            f"Experts' Final Answers:\n{answers_text}\n"
            "Based on the conversation history and final answers, please make the final answer to the question by considering consensus and reasoning quality:\n"
            "Answer: ",
            temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0],
            img_path=None 
    )
            
    log(f"\U0001F468\u200D\u2696\uFE0F  Moderator's final decision: {final_decision}")
    
    if created_tracker:
        try:
            log(f"[INFO] API calls (this sample): {tracker.total_calls()}")
        except Exception:
            pass

    return final_decision

