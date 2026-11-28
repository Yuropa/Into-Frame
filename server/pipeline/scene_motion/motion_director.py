"""The Motion Director: one structured reading of the photograph that says how its
things move -- which boats are underway and which are moored, which way the river
runs and the wind blows, how stiff each plant is -- where the geometry cannot say
and the video's measurements (motion_measurements) are missing or unsure.

It names kinds and directions in the photograph's own terms (left, right, toward or
away from the camera); SceneMotionStage turns them into world directions through the
capture camera, and holds every number to a physical range before using it. Cached
on the photograph and the question, like the Asset Director."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from pipeline.object_completion.asset_director import AssetDirector

SCHEMA_VERSION = 2

DIRECTIONS = ["left", "right", "toward_camera", "away_from_camera", "none", "unknown"]

# Flat at the top: nested objects came back as fragments of the tool call's own markup.
MOTION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["wind_toward", "wind_beaufort", "wind_gustiness", "wind_why",
                 "water_flow_toward", "water_speed_m_s", "swell_period_s", "water_why",
                 "clouds_drift_toward", "clouds_speed", "clouds_why", "boats", "plants"],
    "properties": {
        "wind_toward": {"type": "string", "enum": DIRECTIONS, "description": "the way the wind blows TO, in the photograph"},
        "wind_beaufort": {"type": "number", "description": "0 calm .. 6 strong breeze"},
        "wind_gustiness": {"type": "number", "description": "0 steady .. 1 very gusty"},
        "wind_why": {"type": "string"},
        "water_flow_toward": {"type": "string", "enum": DIRECTIONS, "description": "the way a river's current runs; 'none' for still water or the sea"},
        "water_speed_m_s": {"type": "number", "description": "surface current, m/s"},
        "swell_period_s": {"type": "number", "description": "seconds between breaking waves at a sea shore; 0 when there is none"},
        "water_why": {"type": "string"},
        "clouds_drift_toward": {"type": "string", "enum": DIRECTIONS},
        "clouds_speed": {"type": "string", "enum": ["still", "slow", "moderate", "fast"]},
        "clouds_why": {"type": "string"},
        "boats": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                  "required": ["id", "role", "travel_toward", "speed_m_s", "why"],
                  "properties": {"id": {"type": "integer"},
                                 "role": {"type": "string", "enum": ["underway", "moored", "drifting"]},
                                 "travel_toward": {"type": "string", "enum": DIRECTIONS, "description": "the way its bow points and it moves"},
                                 "speed_m_s": {"type": "number"},
                                 "why": {"type": "string"}}}},
        "plants": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                   "required": ["id", "stiffness", "amplitude_deg", "frequency_hz", "why"],
                   "properties": {"id": {"type": "string"},
                                  "stiffness": {"type": "string", "enum": ["conifer", "broadleaf", "willow", "palm", "shrub", "herb", "grass"]},
                                  "amplitude_deg": {"type": "number", "description": "how far its top bends in this wind, degrees"},
                                  "frequency_hz": {"type": "number", "description": "its sway rate, Hz"},
                                  "why": {"type": "string"}}}},
    },
}

MOTION_SYSTEM = """You direct the motion of a 3D scene rebuilt from one photograph. Say how each listed thing moves
in the moment the photograph shows, judged from the photograph itself: wake, spray, the lean of trees and grass,
flags, ripples, cloud streaks, where a boat sits (against a quay, mid-channel) and which way its bow points.
Directions are in the photograph's terms: left, right, toward_camera, away_from_camera; 'none' when there is no
such motion, 'unknown' when the photograph does not say. Use physical values: a river runs 0.2-2 m/s, a tour boat
travels 2-5 m/s, a moored boat 0; a conifer sways 1-4 degrees at 0.2-0.5 Hz in a breeze, a broadleaf tree 2-6
degrees at 0.3-0.8 Hz, flowers and grass 5-20 degrees at 0.8-2 Hz. Measurements from a generated video are given
when there are some, with a confidence: prefer a confident measurement to your reading, and say when you disagree.
One entry for every listed boat and plant id."""


class MotionDirector(AssetDirector):
    def __init__(self, model: str = "anthropic:claude-sonnet-5-5", cache_dir: Optional[Path] = None, **kw):
        from pipeline.object_completion.asset_director import _ROOT
        super().__init__(model, cache_dir=cache_dir or (_ROOT / "cache" / "motion_director"), **kw)

    def direct(self, photo: Image.Image, photo_sha: str, boats: list[dict], plants: list[dict], facts: str,
               measurements: Optional[dict] = None) -> Optional[dict]:
        h = hashlib.sha256(f"motion:v{SCHEMA_VERSION}:{self.model_id}:{photo_sha}:".encode())
        h.update(json.dumps([boats, plants, facts, measurements or {}], sort_keys=True, default=str).encode())
        key = "motion-" + h.hexdigest()[:40]
        record = self._cached(key)
        if record is not None:
            self.log(f"Motion Director: cached reading {key[:20]} (no call)")
            return record
        W, H = photo.size
        lines = ["Boats (box = x, y, width, height as fractions of the photograph; 'outside' when not in it):"]
        for b in boats:
            box = b.get("photo_box")
            where = (f"box=[{box[0] / W:.3f}, {box[1] / H:.3f}, {box[2] / W:.3f}, {box[3] / H:.3f}]"
                     if box else "outside the photograph")
            lines.append(f"  id {b['id']}: {b.get('name') or 'boat'}, {b.get('length_m', 0):.0f} m long, {where}, "
                         f"{b.get('bank_m', 0):.0f} m from the nearest bank")
        lines.append("Plants (one id per kind; the scene has many of each):")
        for p in plants:
            lines.append(f"  id {p['id']}: {p.get('name') or p.get('class')}, about {p.get('height_m', 0):.1f} m tall, {p.get('count', 1)} placed")
        lines.append("Scene facts: " + facts)
        if measurements:
            lines.append("Measurements from the generated video (priors, with confidence 0-1):\n"
                         + json.dumps(measurements, indent=1, default=str)[:6000])
        reply = self._call("motion", photo, MOTION_SYSTEM, "\n".join(lines), "record_motion", MOTION_SCHEMA, expected_usd=0.06)
        if reply is None or not isinstance(reply.data, dict):
            return None
        d = dict(reply.data)
        record = {"wind": {"toward": d.get("wind_toward"), "beaufort": d.get("wind_beaufort"),
                           "gustiness": d.get("wind_gustiness"), "why": d.get("wind_why")},
                  "water": {"flow_toward": d.get("water_flow_toward"), "speed_m_s": d.get("water_speed_m_s"),
                            "swell_period_s": d.get("swell_period_s"), "why": d.get("water_why")},
                  "clouds": {"drift_toward": d.get("clouds_drift_toward"), "speed": d.get("clouds_speed"), "why": d.get("clouds_why")},
                  "boats": [b for b in (d.get("boats") or []) if isinstance(b, dict)],
                  "plants": [p for p in (d.get("plants") or []) if isinstance(p, dict)]}
        record["model"] = f"{reply.provider}:{reply.model}"
        record["usd"] = round(reply.usd, 4)
        self.ledger.record("Motion Director", reply, key, "motion")
        self._store(key, record)
        return record
