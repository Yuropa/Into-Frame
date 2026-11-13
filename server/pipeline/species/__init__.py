"""Species layer (docs/hybrid-scene-plan.md section 6, Route A): procedural plant
meshes built in headless Blender from a spec whose colours are measured upstream.

    blender-5.2 --background --factory-startup --python pipeline/species/blender_species.py -- spec.json out_dir

`blender_species.py` runs inside Blender's own Python and imports nothing from this
package; this module only holds the spec helpers the pipeline side can import.
"""
from __future__ import annotations

FORMS = ("spike", "raceme", "seedhead", "conifer", "tuft", "shrub")

# species class / plan form words -> builder form
FORM_WORDS = {
    "spike": "spike", "bract": "spike", "paintbrush": "spike", "lousewort": "spike", "bistort": "spike",
    "raceme": "raceme", "lupine": "raceme", "pea": "raceme",
    "seed": "seedhead", "seedhead": "seedhead", "anemone": "seedhead", "pasqueflower": "seedhead",
    "conifer": "conifer", "fir": "conifer", "spruce": "conifer", "pine": "conifer", "spire": "conifer",
    "tuft": "tuft", "tussock": "tuft", "grass": "tuft", "sedge": "tuft",
    "shrub": "shrub", "bush": "shrub", "mat": "shrub", "heather": "shrub",
}


def form_for(text: str, fallback: str = "tuft") -> str:
    """Pick a builder form from a plan's `form` / name text ("upright clustered bract spikes")."""
    words = [w.strip(".,;()").lower() for w in str(text or "").replace("-", " ").split()]
    for w in words:
        if w in FORM_WORDS:
            return FORM_WORDS[w]
    for w in words:
        for key, form in FORM_WORDS.items():
            if w.startswith(key):
                return form
    return fallback
