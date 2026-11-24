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
                       "required": ["target", "node", "kind", "severity", "text", "photo_box"],
                       "properties": {"target": {"type": "string", "description": "the thing the issue is about, in words"},
                                      "node": node,
                                      "photo_box": {"type": "array", "items": {"type": "number"}, "maxItems": 4,
                                                    "description": "where in the PHOTOGRAPH the issue is: [x, y, w, h] as fractions of its width and height (0-1), the region the render should match; [] when it has no place (overall lighting, style)"},
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


def focus_schema(node_ids: Optional[list[str]] = None) -> dict:
    """judge_schema plus a verdict on one targeted issue (the fix loop's question)."""
    s = judge_schema(node_ids)
    s["required"] = list(s["required"]) + ["focus"]
    s["properties"]["focus"] = {"type": "object", "additionalProperties": False,
                                "required": ["resolved", "improved", "evidence", "regressions", "regression_severity", "overall"],
                                "properties": {"resolved": {"type": "boolean", "description": "the targeted issue is fixed in this render (gone, or reduced to minor)"},
                                               "improved": {"type": "boolean", "description": "the target region is clearly closer to the photograph than in the previous render, even if the issue stands (a step worth keeping)"},
                                               "evidence": {"type": "string", "description": "what you see at the target region now, one sentence"},
                                               "regressions": {"type": "string", "description": "anything the change made worse elsewhere, or ''"},
                                               "regression_severity": {"type": "string", "enum": ["none", "minor", "major", "severe"],
                                                                       "description": "how bad the worst thing the edit made worse is, comparing the previous render with this one; none when nothing got worse"},
                                               "overall": {"type": "string", "enum": ["better", "same", "worse"],
                                                           "description": "this render against the previous render as a whole, as a reconstruction of the photograph"}}}
    return s


SYSTEM = """You grade a real-time 3D reconstruction of a photograph. You see the photograph first, then the \
render from the same camera. Observe before you score: say what the render shows, list concrete issues with the \
thing they are about, judge each acceptance item from what is visible, then score the rubric 0-5 (5 = \
indistinguishable from a photograph of the place). The render is a game-engine frame, not a photo: judge whether it \
reads as the same place with believable geometry, surfaces, light and air, not pixel identity. When a list of \
scene-graph nodes is given, name the node each issue is about in `node` (exactly as listed), or '' if none fits. \
Content marked generated (behind the camera, outpainted) is a hypothesis: do not grade it against the photograph. \
Give each issue its place in the photograph as `photo_box` fractions, so the fix can be measured there."""


def _user_text(acceptance: list[dict], node_lines: Optional[list[str]], style: Optional[str],
               focus: Optional[dict] = None, has_previous: bool = False) -> str:
    acc = "\n".join(f"- {a.get('id')} [{a.get('priority')}]: {a.get('text')}" for a in acceptance) or "- (none)"
    parts = ["The first image is the photograph, the second the render from its camera.",
             f"Acceptance items read from the photograph:\n{acc}"]
    if node_lines:
        parts.append("Scene-graph nodes (id <type/kind> name):\n" + "\n".join(node_lines))
    if style:
        parts.append(f"The input is {style}: grade whether the render belongs in that image, not photorealism.")
    if focus:
        box = focus.get("photo_box")
        parts.append("FOCUS. The last edit targeted this issue: "
                     f"[{focus.get('severity')}/{focus.get('kind')}] {focus.get('target')}: {focus.get('text')}"
                     + (f" (photo region x {box[0]:.2f} y {box[1]:.2f} w {box[2]:.2f} h {box[3]:.2f})." if box and len(box) == 4 else ".")
                     + " The third and fourth images are the photograph and the render cropped to that region. "
                     + ("The last images are the PREVIOUS render (before the edit), whole and cropped to the same region: "
                        "compare the two renders directly; a difference you cannot see between them is not a change. "
                        if has_previous else "")
                     + "Say in `focus` whether it is resolved, whether the region is at least clearly closer to the photograph "
                     "than before (improved), what the region shows now, what the edit made worse elsewhere and how bad "
                     "that is (regression_severity), and whether the render as a whole is better, the same or worse than "
                     "the previous one (overall).")
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


def _dicts(items) -> list[dict]:
    """Only the dict entries of a model-written list; a string entry becomes {'text': it}."""
    out = []
    for i in items or []:
        if isinstance(i, dict):
            if "text" not in i and "description" in i:
                i = dict(i, text=i["description"])
            out.append(i)
        elif isinstance(i, str):
            out.append({"text": i})
    return out


def score_verdict(data: dict, acceptance: list[dict], model: str, usd: float) -> dict:
    data = dict(data, issues=_dicts(data.get("issues")), acceptance=_dicts(data.get("acceptance")))
    rubric = data.get("rubric") if isinstance(data.get("rubric"), dict) else {}
    base = sum(_WEIGHTS[k] * float(rubric.get(k) or 0) / 5.0 for k in _WEIGHTS)
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
            "acceptance": data.get("acceptance", []), "rubric": rubric, "usd": round(usd, 4),
            "focus": data.get("focus") if isinstance(data.get("focus"), dict) else None}


def crop_box(im: Image.Image, box: list, margin: float = 0.25, min_side: int = 384) -> Image.Image:
    """The photo region [x, y, w, h] (fractions) with a margin, upscaled to a readable size."""
    W, H = im.size
    x, y, w, h = [float(v) for v in box]
    mx, my = w * margin, h * margin
    x0, y0 = max(0.0, x - mx), max(0.0, y - my)
    x1, y1 = min(1.0, x + w + mx), min(1.0, y + h + my)
    c = im.crop((int(x0 * W), int(y0 * H), max(int(x1 * W), int(x0 * W) + 8), max(int(y1 * H), int(y0 * H) + 8)))
    if min(c.size) < min_side:
        s = min_side / min(c.size)
        c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    return c


def judge(photo: Image.Image, render: Image.Image, plan: Optional[dict], model_id: str, out: Path,
          graph=None, style: Optional[str] = None, base_url: Optional[str] = None,
          ledger_cap_usd: float = 2.0, focus: Optional[dict] = None,
          previous: Optional[Image.Image] = None) -> Optional[dict]:
    """Vision-model verdict on the photo view; written to out/verdict.json. None when no call
    could be made (paid calls off, no key, server down), with the reason printed.
    focus: an issue dict (text, kind, severity, target, photo_box) the last edit targeted; the
    verdict then carries `focus` {resolved, improved, evidence, regressions, regression_severity,
    overall}, judged from crops of the region. previous: the render before that edit; with it the
    judge compares the two renders instead of recalling one it never saw."""
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
    user = _user_text(acceptance, node_lines, style, focus, has_previous=bool(focus) and previous is not None)
    schema = focus_schema(node_ids) if focus else judge_schema(node_ids)
    b1, _, _ = image_to_jpeg_b64(photo, 1024)
    b2, _, _ = image_to_jpeg_b64(render, 1024)
    extra: list[str] = []
    box = (focus or {}).get("photo_box")
    if focus and box and len(box) == 4:
        extra = [image_to_jpeg_b64(crop_box(photo, box), 768)[0], image_to_jpeg_b64(crop_box(render, box), 768)[0]]
    if focus and previous is not None:
        extra.append(image_to_jpeg_b64(previous, 1024)[0])
        if box and len(box) == 4:
            extra.append(image_to_jpeg_b64(crop_box(previous, box), 768)[0])
    ledger = SpendLedger(ROOT / "cache" / "scene_plan", run_cap_usd=ledger_cap_usd)
    t0 = time.time()
    try:
        if pm.provider == "anthropic":
            ledger.check(0.05, "critique")
            data, usage = _anthropic(pm, b1, b2, user, schema, extra)
            usd = estimate_usd(pm.model, usage[0], usage[1])
        else:
            data, usage = _openai(pm, b1, b2, user, schema, raw_path=out / "judge_raw.txt", extra=extra)
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


def _anthropic(pm, b1: str, b2: str, user: str, schema: dict, extra: Optional[list[str]] = None):
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
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in [b1, b2] + list(extra or [])
        ] + [{"type": "text", "text": user}]}])
    data = next((b.input for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
    return data, (msg.usage.input_tokens, msg.usage.output_tokens)


def _openai(pm, b1: str, b2: str, user: str, schema: dict, raw_path=None, extra: Optional[list[str]] = None):
    import httpx
    base = (pm.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "none")
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in [b1, b2] + list(extra or [])]
    content.append({"type": "text", "text": user + "\nReply with the JSON object only."})
    body: dict[str, Any] = {"model": pm.model, "max_tokens": 3000, "temperature": 0.2,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": content}]}
    strict = dict(body, response_format={"type": "json_schema",
                                         "json_schema": {"name": "judge", "schema": schema, "strict": True}})
    last = None
    for mode, payload in (("json_schema", strict), ("plain", body)):
        try:
            r = httpx.post(f"{base}/chat/completions", json=payload, timeout=pm.timeout_s,
                           headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            j = r.json()
            text = j["choices"][0]["message"]["content"].strip()
            if raw_path is not None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(f"[{mode}]\n{text}\n")
            data = parse_json_object(text, required=("rubric", "issues"))
            u = j.get("usage", {})
            return data, (int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
        except Exception as exc:
            last = f"{mode}: {type(exc).__name__}: {str(exc)[:300]}"
            print(f"  judge: {last}")
            if raw_path is not None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("a") as f:
                    f.write(f"[{mode} failed] {last}\n")
    raise RuntimeError(f"OpenAI-compatible critique failed: {last}")


def parse_json_object(text: str, required: tuple[str, ...]) -> dict:
    """The JSON object in a model's reply: fences stripped, a leading/trailing remark cut,
    a single-key wrapper ({"verdict": {...}}) unwrapped. Raises when no object holds any of
    the required keys, so the caller tries its next request mode rather than scoring zeros."""
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise
        data = json.loads(t[a:b + 1])
    if isinstance(data, dict) and not any(k in data for k in required) and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict):
            data = inner
    if not isinstance(data, dict) or not any(k in data for k in required):
        raise ValueError(f"reply has none of {required}: {t[:200]!r}")
    return data
