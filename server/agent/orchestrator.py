"""The orchestrator call: a vision model reads the graph, the measurements, the critique
and the renders, and proposes graph edits in the knob vocabulary
(docs/agentic-scene-graph-plan.md §4). One call per round; the reply is the schema.
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

PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["diagnosis", "edits", "expect", "stop", "stop_reason"],
    "properties": {
        "diagnosis": {"type": "string", "description": "three sentences: the single most damaging difference between render and photograph, which node owns it, and why the proposed edits should fix it"},
        "edits": {"type": "array", "maxItems": 5, "items": {"type": "object", "additionalProperties": False,
                  "required": ["node", "attr", "value", "why"],
                  "properties": {"node": {"type": "string"}, "attr": {"type": "string"},
                                 "value": {"type": ["number", "string", "boolean"]},
                                 "why": {"type": "string"}}}},
        "expect": {"type": "string", "description": "what should change in the next render if the diagnosis is right, in one sentence"},
        "stop": {"type": "boolean", "description": "true when no edit in the vocabulary would improve the scene, or the last round made it worse and the previous state should be kept"},
        "stop_reason": {"type": "string"},
    },
}

SYSTEM = """You improve a real-time 3D reconstruction of a photograph by editing a semantic scene graph. \
You see the photograph, then the current render from the same camera, then a check sheet of more views. \
You are given the graph (nodes with attributes; each attribute carries its evidence: measured beats fitted beats \
plan-visible beats plan-inferred beats default; generated content is a hypothesis and is never graded), the \
measurements of the current render, the critic's issues targeted at nodes, the history of earlier rounds, and the \
vocabulary of edits you may make (node.attribute with its range, the stage it reruns from and what that costs).

Rules. Edit only attributes in the vocabulary, within their ranges. One diagnosis per round, at most five edits that \
serve it; prefer cheap and medium edits, and spend an expensive one only when the diagnosis cannot be answered \
otherwise. A measured value is not overridden by a plan value without a stated reason in the render. Do not repeat an \
edit the history shows made the score worse; undo it instead. When the scene is as good as the vocabulary allows, or \
the last round made it worse and nothing better is left, set stop. Record the proposal with the propose tool / as the \
JSON object."""


def _history_text(history: list[dict]) -> str:
    if not history:
        return "(no earlier rounds)"
    lines = []
    for h in history:
        s = h.get("scores", {})
        lines.append(f"round {h['round']}: judge {s.get('judge', '?')} dino {s.get('dino_global', '?')}/{s.get('dino_layout', '?')} "
                     f"dE {s.get('colour_de', '?')}; edits " + "; ".join(
                         f"{e['node']}.{e['attr']}={e['value']}" for e in h.get("applied", [])) +
                     (f"; outcome: {h['outcome']}" if h.get("outcome") else ""))
    return "\n".join(lines)


def _scores_text(check: Optional[dict], verdict: Optional[dict]) -> str:
    parts = []
    if check:
        parts.append("measurements: " + ", ".join(f"{k} {check[k]}" for k in
                     ("dino_global", "dino_layout", "colour_de", "black", "flat", "oversize", "treeline_tall") if k in check))
    if verdict:
        parts.append(f"critic: score {verdict.get('score')} (rubric {verdict.get('rubric')}), acceptance {verdict.get('acceptance_pass')}, "
                     f"failed musts {verdict.get('failed_must')}; summary: {verdict.get('summary')}")
        for i in verdict.get("issues", []):
            parts.append(f"  issue [{i.get('severity')}/{i.get('kind')}] node={i.get('node') or '-'} {i.get('target')}: {i.get('text')}")
        for a in verdict.get("acceptance", []):
            parts.append(f"  acceptance {a.get('id')}: {'pass' if a.get('pass') else 'FAIL'}: {a.get('evidence')}")
    return "\n".join(parts) or "(no measurements yet)"


def build_user(graph_text: str, knobs_text: str, check: Optional[dict], verdict: Optional[dict],
               history: list[dict], round_no: int, budget_note: str) -> str:
    return (f"Round {round_no}. {budget_note}\n\n"
            "The first image is the photograph, the second the current render from its camera, the third the check "
            "sheet (photo view, left, right and a step forward; the previous round's views below when present).\n\n"
            f"SCENE GRAPH\n{graph_text}\n\nCURRENT RENDER\n{_scores_text(check, verdict)}\n\n"
            f"HISTORY\n{_history_text(history)}\n\nEDIT VOCABULARY (node.attr [range] (type; rerun stage, cost): meaning)\n{knobs_text}\n\n"
            "Propose this round's edits.")


def propose(model_id: str, user: str, images: list[Image.Image], base_url: Optional[str] = None,
            timeout_s: float = 600.0) -> tuple[dict, dict]:
    """Returns (proposal, usage{input_tokens, output_tokens, usd, seconds, model})."""
    sys.path.insert(0, str(ROOT / "server"))
    from pipeline.scene_plan.client import PlanModel, image_to_jpeg_b64, estimate_usd, resolve_anthropic_key
    pm = PlanModel(model_id, timeout_s=timeout_s, retries=1, base_url=base_url)
    if pm.no_api:
        raise RuntimeError("paid calls are off (FRAME_NO_API)")
    b64s = [image_to_jpeg_b64(im, 1280)[0] for im in images]
    t0 = time.time()
    if pm.provider == "anthropic":
        import anthropic
        key = resolve_anthropic_key()
        if not key:
            raise RuntimeError("no Anthropic key")
        client = anthropic.Anthropic(api_key=key, timeout=timeout_s, max_retries=1)
        content: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
                               for b in b64s]
        content.append({"type": "text", "text": user})
        msg = client.messages.create(
            model=pm.model, max_tokens=4000, system=SYSTEM,
            tools=[{"name": "propose", "description": "Record this round's proposal.", "input_schema": PROPOSAL_SCHEMA}],
            tool_choice={"type": "auto"}, messages=[{"role": "user", "content": content}])
        data = next((b.input for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
        if not isinstance(data, dict):
            text = "".join(getattr(b, "text", "") for b in msg.content)
            raise RuntimeError(f"no propose tool call in the reply: {text[:300]}")
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                 "usd": estimate_usd(pm.model, msg.usage.input_tokens, msg.usage.output_tokens)}
    else:
        import httpx
        base = (pm.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "none")
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in b64s]
        content.append({"type": "text", "text": user + "\nReply with the JSON object only."})
        body: dict[str, Any] = {"model": pm.model, "max_tokens": 4000, "temperature": 0.2,
                                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
        strict = dict(body, response_format={"type": "json_schema", "json_schema": {"name": "propose", "schema": PROPOSAL_SCHEMA, "strict": True}})
        last = None
        data = None
        for payload in (strict, body):
            try:
                r = httpx.post(f"{base}/chat/completions", json=payload, timeout=timeout_s, headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                j = r.json()
                text = j["choices"][0]["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                data = json.loads(text)
                u = j.get("usage", {})
                usage = {"input_tokens": int(u.get("prompt_tokens", 0)), "output_tokens": int(u.get("completion_tokens", 0)), "usd": 0.0}
                break
            except Exception as exc:
                last = exc
        if data is None:
            raise RuntimeError(f"OpenAI-compatible propose failed: {last}")
    usage.update({"seconds": round(time.time() - t0, 1), "model": f"{pm.provider}:{pm.model}"})
    data.setdefault("edits", [])
    data.setdefault("stop", False)
    data.setdefault("stop_reason", "")
    return data, usage
