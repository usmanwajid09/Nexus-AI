from nexus.llm.base import LLMProvider
from nexus.memory.parsing import ExtractedMemory, parse_memory_payload

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

_EXTRACTION_PROMPT = """\
You maintain the long-term memory of an AI assistant. Below is one exchange \
from a conversation. Extract facts worth remembering across future sessions.

Rules:
- semantic: durable facts about the user or their projects ("backend uses FastAPI").
- episodic: notable events ("user finished the auth migration today").
- procedural: how-to knowledge and preferences ("user deploys with `make deploy`").
- Each memory must be a single, self-contained sentence understandable without \
this conversation.
- Extract nothing from small talk or generic questions. An empty list is a \
good answer; do not invent memories.

User message:
{user_message}

Assistant reply:
{assistant_reply}
"""


async def extract_memories(
    llm: LLMProvider, *, user_message: str, assistant_reply: str
) -> list[ExtractedMemory]:
    payload = await llm.complete_json(
        prompt=_EXTRACTION_PROMPT.format(
            user_message=user_message, assistant_reply=assistant_reply
        ),
        schema=MEMORY_SCHEMA,
    )
    return parse_memory_payload(payload)
