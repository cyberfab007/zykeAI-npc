from functools import lru_cache
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.models.adapter_manifest import select_adapter


def _quant_config(quantization: Optional[str]) -> Optional[BitsAndBytesConfig]:
    if quantization is None:
        return None
    quantization = quantization.lower()
    if quantization not in {"4bit", "8bit"}:
        raise ValueError("quantization must be one of: None, '4bit', '8bit'")
    if quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    return BitsAndBytesConfig(load_in_8bit=True)


@lru_cache(maxsize=4)
def load_model_with_adapter(
    base_model: str,
    tokenizer_path: str,
    adapter_path: Optional[str] = None,
    adapter_name: Optional[str] = None,
    manifest_path: str = "data/adapters/manifest.json",
    quantization: Optional[str] = None,
    use_flash_attn: bool = False,
    compile_model: bool = False,
    cache_tag: Optional[str] = None,
):
    """
    Load a Hugging Face causal LM and tokenizer, optionally applying a PEFT adapter and quantization.
    Results are cached by (base_model, adapter_path, quantization, tokenizer_path, use_flash_attn, compile_model, cache_tag).
    """
    if adapter_name and not adapter_path:
        adapter_path, _entry = select_adapter(adapter_name, manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = _quant_config(quantization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if quant_cfg else None,
        device_map="auto" if quant_cfg else None,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load adapters") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    # Flash attention requires CUDA + fp16/bf16; fall back quietly otherwise.
    dtype = getattr(model, "dtype", None)
    if use_flash_attn and not (
        device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    ):
        use_flash_attn = False

    if use_flash_attn:
        # Best-effort flash attention toggle (works for flash-compatible builds)
        try:
            if hasattr(model, "set_default_attn_implementation"):
                model.set_default_attn_implementation("flash_attention_2")
            elif hasattr(model, "config"):
                model.config._attn_implementation = "flash_attention_2"
        except Exception:
            pass

    if not quant_cfg:
        model.to(device)

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass
    model.eval()
    return model, tokenizer, device
