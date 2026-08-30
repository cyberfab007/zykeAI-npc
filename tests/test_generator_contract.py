import json
from pathlib import Path

from src.inference import generator


class DummyTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "|".join(str(token) for token in tokens)


def test_generated_token_decode_excludes_prompt_tokens():
    assert generator._decode_generated_tokens(DummyTokenizer(), [10, 11, 12, 13], 2) == "12|13"


def test_parse_fallback_behavior():
    parsed = generator.parse_npc_output("plain text")
    assert parsed["say"] == "plain text"
    assert parsed["action"] == "idle"
    assert parsed["emotion"] == "neutral"


def test_npc_schema_files_exist_and_match_runtime_enums():
    root = Path(__file__).resolve().parents[1]
    actions = json.loads((root / "data/schemas/npc_actions.json").read_text())
    emotions = json.loads((root / "data/schemas/npc_emotions.json").read_text())
    assert set(actions) == set(generator.ALLOWED_ACTIONS)
    assert set(emotions) == set(generator.ALLOWED_EMOTIONS)

    request_schema = json.loads((root / "data/schemas/npc_request.schema.json").read_text())
    response_schema = json.loads((root / "data/schemas/npc_response.schema.json").read_text())
    assert request_schema["required"] == ["persona", "context", "state", "player_input"]
    assert response_schema["required"] == ["say", "action", "emotion"]
