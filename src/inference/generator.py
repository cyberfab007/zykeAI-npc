import json
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.adapter import load_model_with_adapter


ALLOWED_ACTIONS = [
    "idle",
    "nod",
    "point",
    "warn",
    "give_item",
    "take_item",
    "attack",
    "defend",
]

ALLOWED_EMOTIONS = [
    "neutral",
    "happy",
    "angry",
    "afraid",
    "sad",
    "curious",
]

BAD_WORDS = [
    "kill yourself",
    "suicide",
    "self harm",
    "sex",
    "violent",
    "murder",
    "terrorist",
]

REFUSAL = {
    "say": "I cannot help with that.",
    "action": "idle",
    "emotion": "neutral",
    "thoughts": "Refused unsafe content.",
}


def build_npc_prompt(persona: str, context: str, state: str, player_input: str) -> str:
    system = (
        "You are an NPC. Stay in character. Respond ONLY with JSON matching this schema:\n"
        '{\n  "say": "one or two short sentences to the player",\n'
        '  "action": "one of: idle, nod, point, warn, give_item, take_item, attack, defend",\n'
        '  "emotion": "one of: neutral, happy, angry, afraid, sad, curious",\n'
        '  "thoughts": "private reasoning, optional"\n}\n'
        "Do not add extra text before or after the JSON."
    )
    parts = [
        system,
        f"Persona: {persona}",
        f"Context: {context}",
        f"State: {state}",
        f"Player: {player_input}",
        "NPC:",
    ]
    return "\n".join(parts)


def parse_npc_output(raw_text: str) -> Dict[str, str]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    fallback = {"say": raw_text.strip(), "action": "idle", "emotion": "neutral", "thoughts": ""}
    if start == -1 or end == -1 or end <= start:
        return fallback
    try:
        parsed = json.loads(raw_text[start : end + 1])
        say = str(parsed.get("say", fallback["say"]))[:500]
        action = parsed.get("action", "idle")
        emotion = parsed.get("emotion", "neutral")
        thoughts = str(parsed.get("thoughts", ""))
        if action not in ALLOWED_ACTIONS:
            action = "idle"
        if emotion not in ALLOWED_EMOTIONS:
            emotion = "neutral"
        return {"say": say, "action": action, "emotion": emotion, "thoughts": thoughts}
    except Exception:
        return fallback


def generate_text(model_path, tokenizer_path, prompt, max_length=100):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def _bad_words_ids(tokenizer: AutoTokenizer, words: List[str]) -> List[List[int]]:
    ids = []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks)
    return ids


def _enforce_safety(text: str, safe_mode: bool) -> bool:
    if not safe_mode:
        return True
    lower = text.lower()
    return not any(bad in lower for bad in BAD_WORDS)


def _validate_structured(payload: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
    """Return (valid, normalized_payload)."""
    say = str(payload.get("say", "")).strip()
    action = payload.get("action", "idle")
    emotion = payload.get("emotion", "neutral")
    thoughts = str(payload.get("thoughts", "")).strip()
    if action not in ALLOWED_ACTIONS:
        action = "idle"
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"
    normalized = {"say": say[:500] if say else "", "action": action, "emotion": emotion, "thoughts": thoughts[:500]}
    enum_violations = 0
    if action not in ALLOWED_ACTIONS:
        enum_violations += 1
    if emotion not in ALLOWED_EMOTIONS:
        enum_violations += 1
    missing_required = int(not normalized["say"]) + int(not normalized["action"]) + int(not normalized["emotion"])
    valid = bool(normalized["say"] and normalized["action"] and normalized["emotion"])
    return valid, normalized, enum_violations, missing_required


def generate_npc_response(
    base_model: str,
    tokenizer_path: str,
    persona: str,
    context: str,
    state: str,
    player_input: str,
    adapter_path: Optional[str] = None,
    adapter_name: Optional[str] = None,
    manifest_path: str = "data/adapters/manifest.json",
    max_new_tokens: int = 80,
    safe_mode: bool = True,
    quantization: Optional[str] = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    num_beams: Optional[int] = None,
    max_retries: int = 2,
    return_stats: bool = False,
    enforce_schema: bool = True,
    use_flash_attn: bool = False,
    compile_model: bool = False,
) -> Dict[str, str]:
    prompt = build_npc_prompt(persona, context, state, player_input)
    model, tokenizer, device = load_model_with_adapter(
        base_model=base_model,
        tokenizer_path=tokenizer_path,
        adapter_path=adapter_path,
        adapter_name=adapter_name,
        manifest_path=manifest_path,
        quantization=quantization,
        use_flash_attn=use_flash_attn,
        compile_model=compile_model,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    bad_words_ids = _bad_words_ids(tokenizer, BAD_WORDS) if safe_mode else None

    # If schema enforcement is off (adult profile), return the first pass raw parse without retries.
    if not enforce_schema:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if (num_beams is None or num_beams <= 1) else False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams or 1,
                pad_token_id=tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
        raw = tokenizer.decode(output[0], skip_special_tokens=True)
        parsed = parse_npc_output(raw)
        return (parsed, {"final_outcome": "success", "npc_json_valid": True}) if return_stats else parsed

    stats = {
        "npc_json_valid": False,
        "enum_violations": 0,
        "missing_required_fields": 0,
        "retry_count": 0,
        "retry_reason": None,
        "final_outcome": "refusal",
        "safety_filter_triggered": False,
        "initial_temperature": temperature,
        "initial_top_p": top_p,
        "final_temperature": None,
        "final_top_p": None,
    }

    for attempt in range(max_retries + 1):
        temp = max(0.2, temperature * (0.7**attempt))
        top_p_cur = min(0.95, max(0.5, top_p - 0.1 * attempt))
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if (num_beams is None or num_beams <= 1) else False,
                temperature=temp,
                top_p=top_p_cur,
                top_k=top_k,
                num_beams=num_beams or 1,
                pad_token_id=tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
        raw = tokenizer.decode(output[0], skip_special_tokens=True)
        if not _enforce_safety(raw, safe_mode):
            stats["safety_filter_triggered"] = True
            stats["retry_reason"] = "safety_filter"
            continue
        parsed = parse_npc_output(raw)
        ok, normalized, enum_violations, missing_required = _validate_structured(parsed)
        stats["enum_violations"] += enum_violations
        stats["missing_required_fields"] += missing_required
        if ok:
            stats["npc_json_valid"] = True
            stats["final_outcome"] = "success"
            stats["retry_count"] = attempt
            stats["final_temperature"] = temp
            stats["final_top_p"] = top_p_cur
            return (normalized, stats) if return_stats else normalized
        stats["retry_reason"] = "json_parse_fail" if missing_required else "enum_violation"
        stats["retry_count"] = attempt + 1

    # Final fallback if we never produced valid structured output
    stats["final_outcome"] = "refusal"
    stats["final_temperature"] = max(0.2, temperature * (0.7**max_retries))
    stats["final_top_p"] = min(0.95, max(0.5, top_p - 0.1 * max_retries))
    return (REFUSAL.copy(), stats) if return_stats else REFUSAL.copy()


if __name__ == "__main__":
    persona = "Aria, a cautious town guard who values order and warns travelers of danger."
    context = "You stand at the gate of Greyford. Bandits have been sighted nearby."
    state = "Player reputation: neutral. Player carries a rusty sword."
    player = "I'm looking for work. Any trouble around here?"
    result = generate_npc_response(
        base_model="models/latest",
        tokenizer_path="models/latest",
        adapter_path=None,
        persona=persona,
        context=context,
        state=state,
        player_input=player,
    )
    print(json.dumps(result, indent=2))
