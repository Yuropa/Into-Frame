"""The one paid call: a vision chat model reads the photograph into the plan schema.

Providers:
  anthropic   Anthropic Messages API, forced tool use so the reply IS the schema.
  openai      Any OpenAI-compatible /chat/completions endpoint (a local vLLM / llama.cpp
              server on the other machine): json_schema response_format with a plain-JSON
              fallback. Base URL from `base_url` or OPENAI_BASE_URL; key from OPENAI_API_KEY.

Keys never touch the log or the disk. The Anthropic key comes from ANTHROPIC_API_KEY or
the login keyring (`keyring get <service> <user>`, the same entry scripts/c3d-run.sh uses).

Spend guards (docs/hybrid-scene-plan.md section 11): FRAME_NO_API=1 or no_api=True makes a
cache miss raise NoApiError instead of calling out; every paid call appends one line to
<cache_dir>/spend.jsonl; a per-run cap stops calls past it.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# USD per million tokens (input, output); mirrors the harness's pricing table, 2026-10.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


class NoApiError(RuntimeError):
    """A paid call was needed and paid calls are off (FRAME_NO_API / no_api)."""


class SpendCapError(RuntimeError):
    """The per-run spend cap would be exceeded."""


@dataclass
class ModelReply:
    data: dict
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    usd: float
    seconds: float
    raw_text: str = ""


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    key = next((k for k in PRICES if k in model), None)
    if key is None:
        return 0.0
    pin, pout = PRICES[key]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def resolve_anthropic_key(service: str = "anthropic", user: str = "3dcodeverse") -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import keyring
        return keyring.get_password(service, user) or None
    except Exception:
        return None


def image_to_jpeg_b64(pil_image, max_side: int = 1568, quality: int = 90) -> tuple[str, int, int]:
    im = pil_image.convert("RGB")
    w, h = im.size
    s = max(w, h)
    if s > max_side:
        im = im.resize((round(w * max_side / s), round(h * max_side / s)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii"), im.width, im.height


class SpendLedger:
    def __init__(self, cache_dir: Path, run_cap_usd: float = 2.0):
        self.path = Path(cache_dir) / "spend.jsonl"
        self.run_cap_usd = float(run_cap_usd)
        self.run_total = 0.0

    def check(self, expected_usd: float, what: str):
        if self.run_cap_usd > 0 and self.run_total + expected_usd > self.run_cap_usd:
            raise SpendCapError(
                f"{what}: expected ${expected_usd:.2f} would take this run past its ${self.run_cap_usd:.2f} cap "
                f"(spent ${self.run_total:.2f}); raise max_usd_per_run or use the cache")

    def record(self, stage: str, reply: ModelReply, cache_key: str, reason: str):
        self.run_total += reply.usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "provider": reply.provider,
                "model": reply.model, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                "usd": round(reply.usd, 4), "seconds": round(reply.seconds, 1), "cache_key": cache_key,
                "reason": reason}
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")


class PlanModel:
    def __init__(self, model_id: str = "anthropic:claude-opus-5-5", timeout_s: float = 600.0,
                 retries: int = 1, base_url: Optional[str] = None,
                 key_service: str = "anthropic", key_user: str = "3dcodeverse", no_api: bool = False):
        provider, _, model = model_id.partition(":")
        if not model:
            provider, model = "anthropic", provider
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.base_url = base_url
        self.key_service, self.key_user = key_service, key_user
        self.no_api = bool(no_api) or os.environ.get("FRAME_NO_API", "") not in ("", "0", "false")

    # ── public ────────────────────────────────────────────────────────────────
    def read_photo(self, pil_image, system: str, user: str, tool_name: str, schema: dict,
                   max_tokens: int = 8000, extra_images=None, purpose: str = "Record the scene plan read from the photograph.") -> ModelReply:
        """One structured reading of an image (plus optional further images, in order) through
        a tool/schema. `purpose` is the tool's description."""
        if self.no_api:
            raise NoApiError(f"a {self.provider}:{self.model} call is needed and paid calls are off (FRAME_NO_API)")
        b64, w, h = image_to_jpeg_b64(pil_image)
        extra = [image_to_jpeg_b64(im)[0] for im in (extra_images or [])]
        user = user.format(width=w, height=h) if "{width}" in user else user
        started = time.time()
        if self.provider == "anthropic":
            reply = self._anthropic(b64, system, user, tool_name, schema, max_tokens, extra, purpose)
        elif self.provider in ("openai", "local"):
            reply = self._openai(b64, system, user, tool_name, schema, max_tokens, extra)
        else:
            raise ValueError(f"unknown plan model provider {self.provider!r} (anthropic | openai)")
        reply.seconds = time.time() - started
        return reply

    # ── anthropic ─────────────────────────────────────────────────────────────
    def _anthropic(self, b64, system, user, tool_name, schema, max_tokens, extra=(), purpose="") -> ModelReply:
        import anthropic
        key = resolve_anthropic_key(self.key_service, self.key_user)
        if not key:
            raise RuntimeError("no Anthropic key: set ANTHROPIC_API_KEY or `keyring set anthropic 3dcodeverse`")
        client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=self.retries)
        msg = client.messages.create(
            # Readings drift between calls (Paris: river 130 m visible -> 110 m inferred,
            # sun az 140 -> 110 between two readings of the same photograph). The SDK
            # (2.8) takes no temperature for this model family, so repeatability rests on
            # the reading cache: a photograph is read once and the plan kept.
            model=self.model, max_tokens=max_tokens, system=system,
            tools=[{"name": tool_name, "description": purpose or "Record the reading of the photograph.",
                    "input_schema": schema}],
            # Forced tool_choice ("tool"/"any") is rejected by this model family, so the
            # tool is offered and the prompt asks for it; a JSON text reply is the fallback.
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
                for b in (b64, *extra)
            ] + [
                {"type": "text", "text": user + f"\n\nRecord the complete answer by calling the {tool_name} tool."},
            ]}],
        )
        data = None
        text_parts = []
        for block in msg.content:
            if getattr(block, "type", "") == "tool_use" and block.name == tool_name:
                data = block.input
                break
            if getattr(block, "type", "") == "text":
                text_parts.append(block.text)
        if not isinstance(data, dict) and text_parts:
            text = "\n".join(text_parts)
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except ValueError:
                    data = None
        if not isinstance(data, dict):
            raise RuntimeError(f"model returned no {tool_name} tool call (stop_reason {msg.stop_reason})")
        usage = msg.usage
        return ModelReply(data=data, model=self.model, provider="anthropic",
                          input_tokens=int(usage.input_tokens), output_tokens=int(usage.output_tokens),
                          usd=estimate_usd(self.model, usage.input_tokens, usage.output_tokens), seconds=0.0)

    # ── openai-compatible ─────────────────────────────────────────────────────
    def _openai(self, b64, system, user, tool_name, schema, max_tokens, extra=()) -> ModelReply:
        import httpx
        base = (self.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "none")
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in (b64, *extra)]
        content.append({"type": "text", "text": user + "\nReply with the JSON object only."})
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": content}]}
        strict = dict(body, response_format={"type": "json_schema",
                                             "json_schema": {"name": tool_name, "schema": schema, "strict": True}})
        last = None
        for attempt, payload in enumerate((strict, body)):
            try:
                r = httpx.post(f"{base}/chat/completions", json=payload, timeout=self.timeout_s,
                               headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                j = r.json()
                text = j["choices"][0]["message"]["content"]
                text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                data = json.loads(text)
                u = j.get("usage", {})
                return ModelReply(data=data, model=self.model, provider="openai",
                                  input_tokens=int(u.get("prompt_tokens", 0)),
                                  output_tokens=int(u.get("completion_tokens", 0)), usd=0.0, seconds=0.0,
                                  raw_text=text)
            except Exception as exc:  # strict schema unsupported -> plain JSON
                last = exc
        raise RuntimeError(f"OpenAI-compatible plan call failed: {last}")
