import asyncio
import json
import os
import re
from datetime import datetime
from google import genai
from google.genai import types as genai_types
from sqlalchemy.orm import Session
from models import Agent, Event

_client = None

# A single stalled Gemini request would otherwise wedge an entire multi-tick run
# (the client has no built-in timeout). Cap every call so a stall degrades to a
# caught exception (→ the agent skips this tick) instead of hanging the simulation.
LLM_TIMEOUT_S = 90

# Gemini intermittently returns 503 UNAVAILABLE / 429 during load spikes. Without a
# retry a single blip kills a whole multi-tick run, so retry transient failures with
# exponential backoff. Non-transient errors (bad request, auth) raise immediately.
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.5
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "overloaded")


def _is_transient(exc: Exception) -> bool:
    return any(m.lower() in str(exc).lower() for m in _TRANSIENT_MARKERS)


async def call_with_retry(make_coro, timeout: float = LLM_TIMEOUT_S):
    """
    Await `make_coro()` with a timeout, retrying transient API failures with
    exponential backoff. `make_coro` must be a zero-arg callable returning a fresh
    awaitable (a coroutine can only be awaited once).
    """
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await asyncio.wait_for(make_coro(), timeout=timeout)
        except asyncio.TimeoutError as e:
            last = e
        except Exception as e:
            if not _is_transient(e):
                raise
            last = e
        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last if last else RuntimeError("call_with_retry exhausted")

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


class BaseAgent:
    def __init__(self, agent_record: Agent):
        self.agent = agent_record
        self.id = agent_record.id
        self.name = agent_record.name
        self.role = agent_record.role
        self.model_id = agent_record.model_id
        self.system_prompt = agent_record.system_prompt
        self.domain = getattr(agent_record, "domain", "")
        self.llm_assist_rate = getattr(agent_record, "llm_assist_rate", 0.7)

    async def call_llm(self, prompt: str, max_tokens: int = 480) -> str:
        response = await call_with_retry(lambda: _get_client().aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.8,
                # Thinking is ON by default and consumes the output budget, truncating
                # JSON mid-string. Agent replies are short structured JSON, so disable it
                # and give the whole budget to the answer (project gotcha, hit 3x).
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        ))
        return response.text.strip()

    def parse_json(self, raw: str) -> dict:
        # Strip markdown fences and surrounding whitespace
        text = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        # If there's still surrounding text, extract the first {...} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)

    def log_event(
        self,
        db: Session,
        session_id: str,
        event_type: str,
        sim_tick: int,
        target_type: str = None,
        target_id: str = None,
        metadata: dict = None,
        is_prelude: bool = False,
    ):
        event = Event(
            agent_id=self.id,
            agent_role=self.role.value if hasattr(self.role, "value") else str(self.role),
            session_id=session_id,
            event_type=event_type,
            target_type=target_type,
            target_id=target_id,
            metadata_=metadata or {},
            is_prelude=is_prelude,
            timestamp=datetime.utcnow(),
            sim_tick=sim_tick,
        )
        db.add(event)
