"""Scene Plan: one vision-model reading of the photograph, cached by photo hash.

The plan never places geometry. It carries provenance-labelled claims (visible /
inferred / invented) that other stages consult through ScenePlan.prior(), which
applies the trust rules in docs/hybrid-scene-plan.md section 4 in one place.
"""
from pipeline.scene_plan.plan import ScenePlan, PriorResult  # noqa: F401
