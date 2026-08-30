import json
from pathlib import Path
from typing import Dict, Optional, Tuple

REQUIRED_FIELDS = {
    "name",
    "adapter_path",
    "base_model",
    "target_modules",
    "r",
    "lora_alpha",
    "lora_dropout",
    "version",
}


class AdapterManifestError(ValueError):
    pass


def load_manifest(manifest_path: str) -> Dict[str, Dict]:
    path = Path(manifest_path)
    if not path.exists():
        raise AdapterManifestError(f"Manifest not found: {manifest_path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AdapterManifestError(f"Invalid manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AdapterManifestError("Manifest root must be an object keyed by adapter name")
    for name, entry in manifest.items():
        if not isinstance(entry, dict):
            raise AdapterManifestError(f"Adapter '{name}' must map to an object")
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise AdapterManifestError(f"Adapter '{name}' missing fields: {', '.join(sorted(missing))}")
    return manifest


def select_adapter(
    adapter_name: str, manifest_path: str = "data/adapters/manifest.json"
) -> Tuple[str, Dict]:
    manifest = load_manifest(manifest_path)
    if adapter_name not in manifest:
        raise AdapterManifestError(f"Adapter '{adapter_name}' not found in manifest")
    entry = manifest[adapter_name]
    adapter_path = entry["adapter_path"]
    return adapter_path, entry
