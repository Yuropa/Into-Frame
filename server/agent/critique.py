"""Critique: a vision model reads a render against the photograph and returns structured
issues, each targeted at a scene-graph node when one fits (docs/agentic-scene-graph-plan.md
§3 "critique"). Used by scripts/scene-check.py (--judge) and by the agent loop.

Providers, by the model id's prefix (the plan client's convention):
  anthropic:<model>   forced tool use, so the reply IS the schema
  openai:<model>      any OpenAI-compatible /chat/completions (llama.cpp on the Mac Studio,
                      vLLM): json_schema response_format first, plain JSON as the fallback.
                      Base URL from OPENAI_BASE_URL (or base_url=), key from OPENAI_API_KEY.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

ISSUE_KINDS = ["composition", "geometry", "material", "scale", "lighting", "atmosphere", "missing", "extra",
               "motion", "style"]
SEVERITY = ["minor", "major", "severe"]
RUBRIC = ["layout", "geometry", "materials", "lighting", "atmosphere", "realism"]
_PENALTY = {"minor": 0.02, "major": 0.06, "severe": 0.12}
_WEIGHTS = {"layout": 0.25, "geometry": 0.2, "materials": 0.15, "lighting": 0.15, "atmosphere": 0.1, "realism": 0.15}


def judge_schema(node_ids: Optional[list[str]] = None) -> dict:
    node = {"type": "string", "description": "the scene-graph node this issue is about, or '' when none fits"}
    if node_ids:
        node["enum"] = list(node_ids) + [""]
    return {
        "type": "object", "additionalProperties": False,
        "required": ["summary", "issues", "acceptance", "rubric"],
        "properties": {
            "summary": {"type": "string", "description": "two sentences: what the render shows vs the photograph"},
            "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                       "required": ["target", "node", "kind", "severity", "text"],
                       "properties": {"target": {"type": "string", "description": "the thing the issue is about, in words"},
                                      "node": node,
                                      "kind": {"type": "string", "enum": ISSUE_KINDS},
                                      "severity": {"type": "string", "enum": SEVERITY},
                                      "text": {"type": "string"}}}},
            "acceptance": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                           "required": ["id", "pass", "evidence"],
                           "properties": {"id": {"type": "string"}, "pass": {"type": "boolean"},
                                          "evidence": {"type": "string"}}}},
            "rubric": {"type": "object", "additionalProperties": False, "required": RUBRIC,
                       "properties": {k: {"type": "integer", "minimum": 0, "maximum": 5} for k in RUBRIC}},
        },
    }


SYSTEM = """You grade a real-time 3D reconstruction of a photograph. You see the photograph first, then the \
render from the same camera. Observe before you score: say what the render shows, list concrete issues with the \
thing they are about, judge each acceptance item from what is visible, then score the rubric 0-5 (5 = \
indistinguishable from a photograph of the place). The render is a game-engine frame, not a photo: judge whether it \
reads as the same place with believable geometry, surfaces, light and air, not pixel identity. When a list of \
scene-graph nodes is given, name the node each issue is about in `node` (exactly as listed), or '' if none fits. \
Content marked generated (behind the camera, outpainted) is a hypothesis: do not grade it against the photograph."""


def _user_text(acceptance: list[dict], node_lines: Optional[list[str]], style: Optional[str]) -> str:
    acc = "\n".join(f"- {a.get('id')} [{a.get('priority')}]: {a.get('text')}" for a in acceptance) or "- (none)"
    parts = ["The first image is the photograph, the second the render from its camera.",
             f"Acceptance items read from the photograph:\n{acc}"]
    if node_lines:
        parts.append("Scene-graph nodes (id <type/kind> name):\n" + "\n".join(node_lines))
    if style:
        parts.append(f"The input is {style}: grade whether the render belongs in that image, not photorealism.")
    parts.append("Record your verdict with the judge tool / as the JSON object.")
    return "\n\n".join(parts)


def _node_lines(graph) -> list[str]:
    out = []
    for n in graph.nodes.values():
        if n.type in ("World", "Style"):
            continue
        tag = f"{n.id} <{n.type}" + (f"/{n.kind}" if n.kind and n.kind != n.id else "") + ">"
        if n.name and n.name not in (n.kind, n.id):
            tag += f" {n.name}"
        out.append(tag)
    return out


def score_verdict(data: dict, acceptance: list[dict], model: str, usd: float) -> dict:
    rubric = data.get("rubric", {})
    base = sum(_WEIGHTS[k] * float(rubric.get(k, 0)) / 5.0 for k in _WEIGHTS)
    penalty = min(0.3, sum(_PENALTY.get(i.get("severity"), 0.0) for i in data.get("issues", [])))
    musts = {a.get("id") for a in acceptance if a.get("priority") == "must"}
    failed_must = [a["id"] for a in data.get("acceptance", []) if a.get("id") in musts and not a.get("pass")]
    score = max(0.0, base - penalty)
    if failed_must:
        score = min(score, 0.6)
    passed = sum(1 for a in data.get("acceptance", []) if a.get("pass"))
    return {"model": model, "score": round(score, 3), "rubric_score": round(base, 3), "penalty": round(penalty, 3),
            "failed_must": failed_must, "acceptance_pass": f"{passed}/{len(data.get('acceptance', []))}",
            "summary": data.get("summary"), "issues": data.get("issues", []),
            "acceptance": data.get("acceptance", []), "rubric": rubric, "usd": round(usd, 4)}


def judge(photo: Image.Image, render: Image.Image, plan: Optional[dict], model_id: str, out: Path,
          graph=None, style: Optional[str] = None, base_url: Optional[str] = None,
          ledger_cap_usd: float = 2.0) -> Optional[dict]:
    """Vision-model verdict on the photo view; written to out/verdict.json. None when no call
    could be made (paid calls off, no key, server down), with the reason printed."""
    sys.path.insert(0, str(ROOT / "server"))
    from pipeline.scene_plan.client import (PlanModel, SpendLedger, NoApiError, SpendCapError,
                                            image_to_jpeg_b64, resolve_anthropic_key, estimate_usd, ModelReply)
    pm = PlanModel(model_id, timeout_s=300, retries=1, base_url=base_url)
    if pm.no_api:
        print("  judge: paid calls are off (FRAME_NO_API); skipped")
        return None
    acceptance = (plan or {}).get("acceptance") or []
    node_lines = _node_lines(graph) if graph is not None else None
    node_ids = [l.split(" ", 1)[0] for l in node_lines] if node_lines else None
    user = _user_text(acceptance, node_lines, style)
    schema = judge_schema(node_ids)
    b1, _, _ = image_to_jpeg_b64(photo, 1024)
    b2, _, _ = image_to_jpeg_b64(render, 1024)
    ledger = SpendLedger(ROOT / "cache" / "scene_plan", run_cap_usd=ledger_cap_usd)
    t0 = time.time()
    try:
        if pm.provider == "anthropic":
            ledger.check(0.05, "critique")
            data, usage = _anthropic(pm, b1, b2, user, schema)
            usd = estimate_usd(pm.model, usage[0], usage[1])
        else:
            data, usage = _openai(pm, b1, b2, user, schema)
            usd = 0.0
    except (NoApiError, SpendCapError) as exc:
        print(f"  judge: {exc}")
        return None
    except Exception as exc:
        print(f"  judge failed: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(data, dict):
        print("  judge: no structured reply")
        return None
    ledger.record("critique", ModelReply(data, pm.model, pm.provider, usage[0], usage[1], usd, time.time() - t0),
                  out.name, "judge")
    if node_ids:   # a node the model invented is dropped to '' rather than trusted
        for i in data.get("issues", []):
            if i.get("node") not in node_ids:
                i["node"] = ""
    verdict = score_verdict(data, acceptance, f"{pm.provider}:{pm.model}", usd)
    verdict["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def _anthropic(pm, b1: str, b2: str, user: str, schema: dict):
    import anthropic
    from pipeline.scene_plan.client import resolve_anthropic_key
    key = resolve_anthropic_key()
    if not key:
        raise RuntimeError("no Anthropic key: set ANTHROPIC_API_KEY or `keyring set anthropic 3dcodeverse`")
    client = anthropic.Anthropic(api_key=key, timeout=300, max_retries=1)
    msg = client.messages.create(
        model=pm.model, max_tokens=3000, system=SYSTEM,
        tools=[{"name": "judge", "description": "Record the verdict.", "input_schema": schema}],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b1}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b2}},
            {"type": "text", "text": user}]}])
    data = next((b.input for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
    return data, (msg.usage.input_tokens, msg.usage.output_tokens)


def _openai(pm, b1: str, b2: str, user: str, schema: dict):
    import httpx
    base = (pm.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "none")
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b1}"}},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b2}"}},
               {"type": "text", "text": user + "\nReply with the JSON object only."}]
    body: dict[str, Any] = {"model": pm.model, "max_tokens": 3000, "temperature": 0.2,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": content}]}
    strict = dict(body, response_format={"type": "json_schema",
                                         "json_schema": {"name": "judge", "schema": schema, "strict": True}})
    last = None
    for payload in (strict, body):
        try:
            r = httpx.post(f"{base}/chat/completions", json=payload, timeout=pm.timeout_s,
                           headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            j = r.json()
            text = j["choices"][0]["message"]["content"].strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            u = j.get("usage", {})
            return json.loads(text), (int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
        except Exception as exc:
            last = exc
    raise RuntimeError(f"OpenAI-compatible critique failed: {last}")
