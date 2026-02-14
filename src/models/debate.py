import os
import re
import time
import random
import threading
from prettytable import PrettyTable 

from google import genai
from google.genai import types
from openai import OpenAI
from mistralai import SystemMessage, UserMessage, AssistantMessage

from src.models.helpers import (
    noop_log,
    get_mistral_client,
    remove_mistral_thinking
)


# -----------------------
# Robustness helpers
# -----------------------

# Robust regex helpers 
_EXPERT_LINE_RE = re.compile(r'^\s*(?P<expert>.+?)\s*$', re.IGNORECASE)
_EXPERT_ROLE_DESC_RE = re.compile(r'^\s*(?:\d+\.\s*)?(?P<role>.+?)(?:\s*-\s*(?P<desc>.+))?\s*$')

# Retry helper
DEFAULT_LLM_RETRIES = 5

def _retry_call(name, fn, max_tries=None, retry_exceptions=(IndexError,), sleep_s=None, log=None):
    if log is None:
        log = noop_log
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

    register = register_agent

    def total_calls(self):
        with self._lock:
            return sum(getattr(a, 'api_calls', 0) for a in self._agents)

    def breakdown(self):
        with self._lock:
            return [(getattr(a, 'role', 'unknown'), getattr(a, 'api_calls', 0)) for a in self._agents]


# -----------------------
# Main implementation
# -----------------------

class Agent:
    # Class-level counter for total API calls across all agents
    total_api_calls = 0
    _api_calls_lock = threading.Lock()  # Thread-safe lock for API call counting
    
    def __init__(self, instruction, role, examplers=None, provider='mistral', model_info='mistral-large-2512', img_path=None, tracker=None):
        self.instruction = instruction
        self.role = role
        self.provider = provider
        self.model_info = model_info
        self.img_path = img_path
        self.api_calls = 0  
        self._tracker = tracker
        if self._tracker is not None:
            try:
                self._tracker.register_agent(self)
            except Exception:
                pass

        if self.provider == 'gemini':
            self.client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
            self.messages = []
            
            # Map examplers to Gemini history format
            if examplers:
                for exampler in examplers:
                    self.messages.append(types.Content(role="user", parts=[types.Part(text=exampler['question'])])) 
                    reason_prefix = f"Let's think step by step. {exampler['reason']} " if 'reason' in exampler else ""
                    self.messages.append(types.Content(role="model", parts=[types.Part(text=reason_prefix + exampler['answer'])]))
        
        elif self.provider == 'openai':
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content":  ("Let's think step by step. " + exampler['reason'] + " "  if 'reason' in exampler else '') + exampler['answer']})
                    
        elif self.provider == 'mistral':
            # Uses round-robin across multiple keys if configured.
            self.client = get_mistral_client()
            self.messages = [
                SystemMessage(content=instruction)
            ]
            if examplers:
                for exampler in examplers:
                    self.messages.append(UserMessage(content=exampler['question']))
                    self.messages.append(AssistantMessage(content=("Let's think step by step. " + exampler['reason'] + " "  if 'reason' in exampler else '') + exampler['answer']))

        # log(f"[DEBUG] Print out the messages for Agent {self.messages}")

    def chat(self, message, img_path=None):
        if self.provider == 'gemini':
            self.messages.append(types.Content(role="user", parts=[types.Part(text=message)]))
            for attempt in range(10):
                try:
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

        elif self.provider == 'openai':
            self.messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=self.messages
            )
            
            # Track API call (thread-safe)
            with Agent._api_calls_lock:
                self.api_calls += 1
                Agent.total_api_calls += 1

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content
        
        elif self.provider == 'mistral':
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
                    
                    self.messages.append(AssistantMessage(content=remove_mistral_thinking(response.choices[0].message.content)))
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Retrying due to: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                        
    def temp_responses(self, message, temperatures=[0.0], img_path=None):
        if self.provider == 'openai':
            self.messages.append({"role": "user", "content": message})
            
            temperatures = list(set(temperatures))
            
            responses = {}
            for temperature in temperatures:
                response = self.client.chat.completions.create(
                    model=self.model_info,
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
        
        elif self.provider == 'gemini':
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
                                
            return responses
        
        elif self.provider == 'mistral':
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
                        
                        responses[temperature] = remove_mistral_thinking(response.choices[0].message.content)
                        break
                    except Exception as e:
                        print(f"Retrying due to: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                                          
            return responses
        
    def agent_talk(self, message, recipient, img_path=None):
        """
        Generates a message from this agent (self) and injects it into the recipient's context.
        """
        content = self.chat(message, img_path=img_path)

        incoming_msg = f"Message from {self.role}: {content}"

        if recipient.provider == 'openai':
            recipient.messages.append({"role": "user", "content": incoming_msg})
        
        elif recipient.provider == 'gemini':
            recipient.messages.append(types.Content(role="user", parts=[types.Part(text=incoming_msg)]))
            
        elif recipient.provider == 'mistral':
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
    
    # Parse members line by line. Handles: "Member N: Role - Description", "Member N: **Role** - **Description**", etc.
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

def process_query(question, aggregators, user_query, log=None, tracker=None):
    """
    Intermediate (MDT) setting:
      - Recruit N experts 
      - Collect initial opinions
      - For each round:
          * participatory debate (agents optionally message each other)
          * agents update final answers for the round
          * consensus check; if not reached, continue to next round
      - Final decision maker reviews all agent answers and produces the final answer by majority vote.
    """
    if log is None:
        log = noop_log
    
    created_tracker = False
    if tracker is None:
        tracker = SampleAPICallTracker()
        created_tracker = True
    
    moderator = Agent(
        instruction='You are a medical expert who conducts initial assessment and moderates the discussion.',
        role='moderator',
        # model_info='gpt-4o-mini',
        tracker=tracker
    )
    
    log("\n[INFO] Step 1. Expert Recruitment")

    num_agents = len(aggregators) 

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

    # Agent models
    agent_provider = [aggregator["provider"] for aggregator in aggregators]
    agent_models = [aggregator["model"] for aggregator in aggregators]

    agent_dict = {}
    medical_agents = []
    for idx, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent or '')
        agent_role = (m.group('role') if m else (agent or '')).strip().lower()
        description = ((m.group('desc') or '') if m else '').strip().lower()

        inst_prompt = f"[ROLE]\nYou are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team.\n" + user_query
        _agent = Agent(instruction=inst_prompt, role=agent_role, provider=agent_provider[idx], model_info=agent_models[idx], tracker=tracker)
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
    log("\n[INFO] Step 2. Initial Opinions")
    opinions = {}
    for idx, agent in enumerate(medical_agents):
        prompt = (
            f"Provide your current answer using the Answer Card format (from your system instructions).\n"
            f"Question: {question}"
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

        # Participatory debate (T turns)
        log("\n[INFO] Participatory Debate")
        
        num_yes_total = 0
        for t in range(1, num_turns + 1):
            turn_name = f"Turn {t}"
            interaction_log[round_name].setdefault(turn_name, {})
            log(f"|_{turn_name}")

            num_yes = 0
            for idx, agent in enumerate(medical_agents):
                your_opinion = opinions.get(agent.role, "No opinion yet.")
                other_opinions = {k: v for k, v in opinions.items() if k != agent.role}
                participate = agent.chat(
                    "Given your current opinion and opinions from other medical agents, "
                    "indicate whether you want to talk to any expert. "
                    "Return EXACTLY one token: YES or NO. No other text.\n\n"
                    f"Your current opinion: {your_opinion}\n\n"
                    f"Other agents' opinions:\n{other_opinions}",
                    img_path=None
                )

                if participate.strip().upper() == "YES":
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
                    chosen_experts = [ce for ce in chosen_experts if 1 <= ce <= len(medical_agents) and ce != (idx + 1)]  # valid and not self
                    chosen_experts = list(dict.fromkeys(chosen_experts))  # unique, preserve order

                    for ce in chosen_experts:
                        recipient_agent = medical_agents[ce-1]
                        msg = agent.agent_talk(
                            f"Remind your medical expertise as a {agent.role} and leave your opinion to the expert you chose: {recipient_agent.role}. "
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
                    # "NO" path
                    log(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): \U0001F910")
                
            log(f"\n[DEBUG] Current agent chat history for {round_name}, {turn_name}:\n" + 
                "\n".join([f"{idx}. {agent.role} history:\n{agent.messages}" for idx, agent in enumerate(medical_agents)]))
                
            if num_yes == 0:
                log(" No agents chose to participate in this turn. End this turn.")
                break

            # Print summary table for this turn only
            log("\n[INFO] Summary Table\n")
            _print_summary_table(interaction_log[round_name][turn_name], len(medical_agents))

        if num_yes_total == 0:
            log(" No agents chose to participate in this round. End this round.")
            break
        
        # Agents update final answers for this round
        tmp_final_answer = {}
        for agent in medical_agents:
            response = agent.chat(
                f"Now that you've interacted with other medical experts, "
                f"remind your expertise as a {agent.role} and the comments from other experts, "
                f"and provide your UPDATED answer using the Answer Card format.\n"
                f"If you changed your Answer letter, include \"Previous answer: <letter>\" and \"Update reason: <reason>\"\n"
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
    
    log(f"\n[DEBUG] Full Conversation History:\n{conversation_history}")
    
    final_decision = decision_maker.temp_responses(
        "You are reviewing the final decision from a multidisciplinary team discussion. "
        "You are not allowed to solve the question yourself."
        "Your ONLY job is to aggregate the team's FINAL Answer Cards by majority vote. "
        "If ties occur, choose the tied letter with the highest average Confidence (use the Confidence lines).\n"
        f"Question:\n{question}\n\n"
        f"Experts' Final Answers:\n{answers_text}\n"
        "Answer: ",
        temperatures=[0.0],
        img_path=None 
    )
            
    log(f"\U0001F468\u200D\u2696\uFE0F  Moderator's final decision: {final_decision[0.0]}")
    
    if created_tracker:
        try:
            log(f"[INFO] API calls (this sample): {tracker.total_calls()}")
        except Exception:
            pass

    return final_decision[0.0]

