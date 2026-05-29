# GameProfits To ZykeAI Local Protocol

The GameProfits desktop agent owns the local control surface. The ZykeAI worker
performs localhost inference and training only.

## Defaults

- Worker bind host: `127.0.0.1`
- Worker port: `5000`
- Auth: bearer token generated and stored by the desktop agent
- Normal NPC inference stays local.

## Desktop Agent Local API

- `GET /local/status`
- `GET /local/node/status`
- `GET /local/models`
- `GET /local/adapters`
- `GET /local/manifests`
- `POST /local/auth/login`
- `POST /local/auth/refresh`
- `POST /local/npc/generate`
- `POST /local/worker/start`
- `POST /local/worker/stop`
- `POST /local/training/request-job`
- `POST /local/training/start`
- `POST /local/training/stop`
- `POST /local/training/status`
- `POST /local/training/submit-result`

## ZykeAI Worker API

- `GET /health`
- `GET /metrics`
- `POST /generate`

Planned:

- `GET /worker/status`
- `GET /models`
- `GET /adapters`
- `POST /training/start`
- `POST /training/status`
- `POST /training/stop`
- `POST /training/export-delta`
- `POST /memory/query`
- `POST /memory/update`

## Generate Request

```json
{
  "persona": "A cautious town guard.",
  "context": "The player is entering Greyford.",
  "state": "Reputation neutral. Mood alert.",
  "player_input": "Any trouble nearby?",
  "npc_type": "guard",
  "audience": "minor",
  "safe_mode": true,
  "enforce_schema": true,
  "max_new_tokens": 80
}
```

## Generate Response

```json
{
  "result": {
    "say": "Keep your eyes open. Bandits were seen near the north road.",
    "action": "warn",
    "emotion": "neutral",
    "thoughts": "The traveler should be cautious."
  },
  "audience": "minor"
}
```
