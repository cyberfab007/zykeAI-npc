import json
import os
import urllib.request


WORKER_URL = os.getenv("ZYKE_WORKER_URL", "http://127.0.0.1:5000")
API_TOKEN = os.getenv("ZYKE_LOCAL_API_TOKEN")


def generate(payload: dict) -> dict:
    if not API_TOKEN:
        raise RuntimeError("Set ZYKE_LOCAL_API_TOKEN to the local worker bearer token.")

    request = urllib.request.Request(
        f"{WORKER_URL.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    print(
        json.dumps(
            generate(
                {
                    "persona": "A cautious town guard.",
                    "context": "The player is entering Greyford.",
                    "state": "Reputation neutral. Mood alert.",
                    "player_input": "Any trouble nearby?",
                    "npc_type": "guard",
                    "audience": "minor",
                    "safe_mode": True,
                    "enforce_schema": True,
                    "max_new_tokens": 80,
                }
            ),
            indent=2,
        )
    )
