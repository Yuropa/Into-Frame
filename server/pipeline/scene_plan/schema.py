"""Scene plan schema v2 (docs/hybrid-scene-plan.md section 3).

Camera frame: ahead = 0 deg, azimuth positive clockwise (to the right), metres and
degrees. Every claim block carries `evidence` and `because`; a numeric field is
null when the photograph gives no basis for it.
"""

SCHEMA_VERSION = 2

# v2: every landmark, object and species names the pipeline's own object category and
# a short detector phrase, so the plan can prompt Object Labeling (Grounding DINO) and
# the built-structure veto directly instead of being normalised from free text ("stone
# arch bridge" token-matches "rock"). "environment" marks scenery that is not an object
# (a summit, a glacier, a snow patch, a trail).
# camera.standing_on (the surface under the tripod) was added without a new version: it is
# optional, a plan read before it existed is still a whole plan, and without it the
# near-ground plate takes the opening words of the plan's `behind` (what lies around the
# viewer out of frame; terrain_texture_generation._near_ground_prompt). A photograph taken from a bluff
# shows the beach below and never the bluff: the pipeline raised the viewer's footing and
# paved it with that sentence (Shark Fin stood on a mound of cobbles).
from pipeline.object_typing.categories import OBJECT_CATEGORIES

CATEGORY_ENUM = sorted(OBJECT_CATEGORIES) + ["environment"]

EVIDENCE = ["visible", "inferred", "invented"]

_evidence = {"type": "string", "enum": EVIDENCE,
             "description": "visible: read directly off the photograph; inferred: a confident reading from "
                            "context (place, scale cues, shadows); invented: plausible but unsupported."}
_because = {"type": "string", "description": "one short sentence: what in the photograph supports this"}
_num = {"type": ["number", "null"]}
_category = {"type": "string", "enum": CATEGORY_ENUM,
             "description": "the pipeline's object category this is; 'environment' for scenery that is not an "
                            "object (summit, glacier, snow patch, trail, quay)"}
_phrase = {"type": "string", "description": "2-4 plain English words an open-vocabulary detector would find it "
                                             "by, e.g. 'stone arch bridge', 'moored barge', 'magenta paintbrush flower'"}


def _claim(props: dict, required: list[str] | None = None) -> dict:
    props = dict(props)
    props["evidence"] = _evidence
    props["because"] = _because
    return {"type": "object", "properties": props, "required": (required or []) + ["evidence", "because"],
            "additionalProperties": False}


PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["place", "camera", "sun", "atmosphere", "horizon", "water", "terrain", "landmarks",
                 "species", "objects", "motion", "behind", "acceptance"],
    "properties": {
        "place": _claim({
            "name": {"type": ["string", "null"], "description": "the real place if identifiable, else null"},
            "kind": {"type": "string", "description": "e.g. alpine meadow, river quay in a city, sea cliff"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }, ["kind", "confidence"]),
        "camera": _claim({
            "height_m": _num,
            "pitch_deg": _num,
            "on_structure": {"type": ["string", "null"], "description": "bridge, balcony, slope, boat... or null"},
            "standing_on": {"type": ["string", "null"],
                            "description": "the surface directly under the camera as a material, in a few plain "
                                           "words a texture could be painted from: 'dry grass and bare dirt', "
                                           "'asphalt road deck', 'wet sand', 'meadow turf'. Usually out of frame: "
                                           "infer it from the vantage and the place"},
        }, ["height_m", "pitch_deg"]),
        "sun": _claim({
            "azimuth_deg": _num, "elevation_deg": _num,
            "warmth": {"type": "string", "enum": ["cool", "neutral", "golden"]},
            "sky": {"type": "string", "enum": ["clear", "hazy", "overcast"]},
            "time_of_day": {"type": "string"},
        }, ["azimuth_deg", "elevation_deg", "warmth", "sky"]),
        "atmosphere": _claim({
            "haze_near_m": _num, "haze_far_m": _num,
            "tint_rgb": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        }, ["haze_far_m"]),
        "horizon": _claim({"skyline": {"type": "string"}}, ["skyline"]),
        "water": _claim({
            "kind": {"type": "string", "enum": ["none", "river", "lake", "sea", "pond", "stream"]},
            "level_below_camera_m": _num,
            "width_m": _num,
            "bank_left": {"type": ["string", "null"], "enum": ["quay", "natural", "beach", "cliff", None]},
            "bank_right": {"type": ["string", "null"], "enum": ["quay", "natural", "beach", "cliff", None]},
            "continues_upstream": {"type": ["boolean", "null"]},
            "continues_downstream": {"type": ["boolean", "null"]},
            "flow_dir_deg": _num,
        }, ["kind"]),
        "terrain": _claim({
            "kind": {"type": "string", "enum": ["flat", "valley", "slope", "ridge", "coast", "urban"]},
            "slope_pct": _num,
            "ground": {"type": "string", "description": "what the ground surface is"},
        }, ["kind", "ground"]),
        "landmarks": {"type": "array", "items": _claim({
            "name": {"type": "string"}, "class": {"type": "string"},
            "category": _category, "phrase": _phrase,
            "bearing_deg": _num, "distance_m": _num, "height_m": _num,
            "side": {"type": "string", "enum": ["left", "centre", "right"]},
        }, ["name", "class", "category", "phrase", "bearing_deg", "distance_m", "height_m", "side"])},
        "species": {"type": "array", "items": _claim({
            "name": {"type": "string"},
            "class": {"type": "string", "enum": ["flower", "grass", "shrub", "tree", "reed", "moss"]},
            "phrase": _phrase,
            "size_m": _num,
            "colours": {"type": "array", "items": {"type": "string"}},
            "form": {"type": "string"},
            "density": {"type": "string", "enum": ["sparse", "patchy", "dense"]},
            "where": {"type": "string"},
            "crop_hint": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                          "description": "x, y, w, h as fractions of the photograph where one clear specimen is"},
        }, ["name", "class", "phrase", "size_m", "colours", "form", "density", "where"])},
        "objects": {"type": "array", "items": _claim({
            "class": {"type": "string"}, "category": _category, "phrase": _phrase,
            "count": {"type": "integer"}, "where": {"type": "string"},
            "size_m": _num,
        }, ["class", "category", "phrase", "count", "where"])},
        "motion": {"type": "array", "items": _claim({
            "target": {"type": "string"},
            "kind": {"type": "string", "enum": ["sway", "travel", "flow", "drift", "ripple"]},
            "amount": {"type": "string"},
            "direction_deg": _num,
        }, ["target", "kind", "amount"])},
        "behind": {"type": "object", "additionalProperties": False, "required": ["text"],
                   "properties": {"text": {"type": "string",
                                           "description": "what plausibly lies behind the camera; always invented; "
                                                          "used only as prompt text"}}},
        "acceptance": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "text", "how", "priority"],
            "properties": {
                "id": {"type": "string"}, "text": {"type": "string"},
                "how": {"type": "string", "enum": ["visual", "probe"]},
                "priority": {"type": "string", "enum": ["must", "should"]},
            }}},
    },
}

SYSTEM_PROMPT = """You read one outdoor photograph for a 3D reconstruction pipeline that builds the scene \
from the photograph itself (depth, segmentation, outpainting). You do not place geometry. You settle what the \
photograph cannot: the vantage, the scale, the light, the atmosphere, what lies off frame, which plant species \
and props are present, and what must move.

Frame: the camera looks along azimuth 0; azimuth is positive clockwise (to the right), elevation positive up. \
Metres and degrees. Heights are above the ground or water directly below the camera.

Provenance is the point. Every block carries `evidence`:
- visible: you can point at it in the photograph (both river banks in frame -> width; a quay wall; lupines).
- inferred: a confident reading from cues (the camera is on a bridge, so 10-14 m up; sun from shadow direction).
- invented: plausible for the place but unsupported by pixels (what is behind the camera, a skyline you cannot see).
The pipeline uses visible claims to bound its own measurements, inferred claims only when its measurement fails, \
and invented claims as text only. Over-claiming `visible` corrupts the scene; under-claiming wastes a fact.

Landmarks, objects and species are also how the pipeline learns what to look for: give each its `category` \
from the fixed list (what the pipeline can build; `environment` for scenery it cannot) and a `phrase` of 2-4 \
plain words a detector would find it by. Name every visible structure, not only the famous one: a bridge, \
the buildings behind a tree line, moored boats, a quay wall, each as its own entry with a count.

The camera block is the vantage: its height, what it stands on (`on_structure`: a bluff top, a bridge, or null \
for open ground) and the surface under the tripod (`standing_on`), which the photograph rarely shows and the \
pipeline has to build when the view looks down on the scene.

Be concrete and numeric. Ranges are not accepted; give the best single number and say the basis in `because`. \
Species: name the plant as a botanist would, give its size, colours, form (spike, mat, tussock, conifer...), \
density and where it grows in frame; include a `crop_hint` box around one clear specimen when there is one. \
Acceptance items come only from visible claims; `must` only for what would make the scene a different place."""

USER_PROMPT = """Read this photograph and produce the scene plan with the scene_plan tool. The pipeline's own \
caption of it: "{caption}". The photograph is {width}x{height} px."""
