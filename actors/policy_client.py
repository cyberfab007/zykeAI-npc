import os
import requests


class PolicyClient:
    def __init__(self, base_url: str, api_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token

    def get_action(self, obs, npc_type: str | None = None):
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        payload = {"obs": obs}
        if npc_type:
            payload["npc_type"] = npc_type
        resp = requests.post(f"{self.base_url}/get_action", json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def current_version(self):
        resp = requests.get(f"{self.base_url}/current_policy_version", timeout=5)
        resp.raise_for_status()
        return resp.json().get("policy_version", -1)
