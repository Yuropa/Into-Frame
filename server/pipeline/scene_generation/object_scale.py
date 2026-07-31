"""
Scene-learned object scale correction.

Object world size is not measured independently anywhere in the pipeline. Scene
Generation derives it from the panorama depth map, in projection.py:

    world_size = angular_extent_of_bbox x depth_at_bbox_centre

so an object is exactly as big as the depth map says the thing it sits on is far.
That makes size and depth the *same* quantity wearing two hats, and it means every
depth-map compression lands on object size at full strength.

Panorama depth is compressed twice on purpose (see panorama_depth/calibration.py):
_apply_panorama_depth_curve extrapolates past the hero-photo overlap with a capped
log slope, and _compress_far_range then folds everything above 0.6 * max_depth_m
asymptotically into max_depth_m. Both are deliberate and both are load-bearing for
terrain reconstruction, which is why this module does not touch them. The result is
a world that is *angularly* faithful -- a 5 m tree at 95 m subtends the same angle
as the real 15 m tree at 300 m, so the panorama still lines up -- but metrically
shrunk with distance. Mono viewing cannot tell the difference. Stereo can: binocular
disparity reports 95 m while retinal size reports a 5 m object, and the scene reads
as a scale model rather than a landscape.

This module recovers the missing scale from semantics instead of from geometry.
Classes like `person` and `car` have tight real-world height priors, so any detected
instance of one yields an independent measurement of how badly size was compressed
at that instance's depth:

    ratio = prior_height / measured_height

Those ratios are pooled across the whole scene, binned by depth, and fitted into a
monotonic depth -> correction curve. Scene Generation then multiplies placed object
dimensions by it. Position, terrain snapping, and both depth maps are left alone --
only the object's own extent changes.

The deliberate consequence: a corrected object no longer exactly matches the
panorama pixels it was cropped from, because its angular size grows while its
distance stays put. That trade is the entire point. Angular fidelity is what the
uncorrected pipeline already has and what makes the scene feel miniature in a
headset; physical plausibility against the viewer's own body is what it lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pipeline.object_typing.categories import GRASS_TUFT_CATEGORY


# Real-world height priors, {class: (metres, weight)}.
#
# Height, not width or footprint: an equirectangular bounding box's horizontal
# extent depends on the object's yaw relative to the camera (a car seen end-on
# versus broadside differs by ~2.5x) while its vertical extent does not. Height is
# also the dimension a viewer judges scale against, since they have their own to
# compare with.
#
# Weight encodes how tightly the class actually constrains height, and is the only
# thing standing between a plausible correction and a fabricated one. A person is
# 1.7 m give or take 15%; a `building` spans a garden shed to a cathedral and is
# worth nothing as an anchor no matter how confidently it was classified. Classes
# absent from this table contribute no anchors at all -- that is the default, and
# adding a class here is a claim about its variance, not about its detectability.
OBJECT_HEIGHT_PRIORS: dict[str, tuple[float, float]] = {
    # Tight: manufactured or anatomical, narrow distribution, and the classes
    # CLIP is most reliable on.
    "person":       (1.70, 1.00),
    "car":          (1.50, 1.00),
    "bus":          (3.20, 0.90),
    "motorcycle":   (1.10, 0.80),
    "bicycle":      (1.10, 0.80),
    "fire_hydrant": (0.75, 0.80),
    "bench":        (0.90, 0.70),
    "trash_bin":    (1.00, 0.70),
    "chair":        (0.90, 0.60),
    "table":        (0.75, 0.60),

    # Moderate: real spread, still informative in bulk.
    "truck":        (3.20, 0.60),
    "streetlight":  (8.00, 0.50),
    "traffic_light": (3.00, 0.40),
    "bus_stop":     (2.50, 0.40),
    "fence":        (1.50, 0.30),

    # Vegetation (bush/flower/tree) is deliberately ABSENT. It was here on the
    # reasoning that a landscape capture often has nothing else, weighted low enough
    # that a handful of tight anchors would outvote it. Both halves of that failed:
    #
    #   - "Outvoted by tight anchors" assumes tight anchors exist. In a wilderness
    #     capture they never do, so the loose classes are not outvoted, they are the
    #     entire fit. On the Rainier capture all 64 anchors were flower and tree.
    #
    #   - The prior has to describe what the DETECTOR boxes, not what the word means.
    #     `flower: 0.35 m` is a whole flowering plant; GroundingDINO boxes a single
    #     paintbrush bloom, measured at 0.04-0.17 m. `tree: 8.0 m` is a whole tree;
    #     after ObjectInstanceRefinementStage's watershed pass a `tree` is frequently
    #     one fragment of a canopy. Neither ratio is a compression measurement -- it
    #     is a part-vs-whole mismatch, and it is one-sided, so it always reads as the
    #     scene being shrunk.
    #
    # Together those produced a fitted 4.91x applied to the whole scene, from anchors
    # sitting at 1.7-2.5 m -- inside the hero-photo overlap, below _compress_far_range's
    # 60 m knee, in the region where the depth transform is exactly identity and there
    # is by construction no compression to measure. A 10.4 m tree 11.9 m from the
    # viewer was rendered 51 m tall.
    #
    # Absence is the honest state here, and the module already handles it correctly:
    # fewer than min_anchors usable anchors returns None, which means "leave every
    # object exactly the size its own box and depth imply". Re-adding a vegetation
    # class needs a prior calibrated against what the detector actually emits at that
    # point in the pipeline, not against the dictionary definition of the word.
}


# Classes whose world size was authored directly in metres rather than derived from
# the depth map, and which must therefore NOT be corrected.
#
# The correction undoes depth compression, so it only applies to sizes that inherited
# that compression in the first place. Grass tufts do not: GrassCoverStage._tuft_height
# returns a configured constant (tuft_height_m, 0.35 m) and documents at length why it
# refuses to derive the number from imagery -- nothing upstream segments an individual
# blade, so there is no measurement to compress. Scaling those by a factor fitted from
# unrelated objects turns ankle-high ground cover into 2.8 m reeds.
#
# Distribution-synthesized objects are deliberately absent from this set even though
# they are also `synthetic`: DistributionSynthesisStage samples their footprint from
# real exemplar detections (dist.sizes), which were measured through the depth map and
# are compressed exactly like any other detection.
METRIC_AUTHORED_CATEGORIES: frozenset[str] = frozenset({GRASS_TUFT_CATEGORY})


def is_metric_authored(cls: Optional[str], metadata: dict) -> bool:
    """True when this object's size is already real-world metric and must be left alone.

    Checks an explicit `metric_size` flag first so any future stage that authors sizes
    in metres can opt out by setting it, then falls back to the category set. The
    fallback is not redundant: metadata is cached per stage, so a resumed run can carry
    grass entries written before the flag existed, and those must still be excluded.
    """
    if metadata.get("metric_size"):
        return True
    return cls in METRIC_AUTHORED_CATEGORIES


@dataclass
class ScaleAnchor:
    """One measurement of size compression, from a single detected object."""
    index: int
    cls: str
    depth: float
    measured_height: float
    prior_height: float
    weight: float

    @property
    def ratio(self) -> float:
        return self.prior_height / self.measured_height


# Quantile of each depth bin's ratio distribution used as that bin's correction.
#
# Below 0.5 on purpose. The dominant measurement error here is one-sided: a bounding
# box clipped by an occluder, the panorama seam, or the crop's own edge reports the
# object as SHORTER than it is, which inflates prior/measured. There is no
# corresponding mechanism that reports an object as taller than it is, so the error
# distribution has a long right tail and a hard left edge, and its median sits above
# the truth by however contaminated the bin happens to be.
#
# Genuine within-class height variation is roughly symmetric, so sampling below the
# median costs a few percent of under-correction on clean anchors. That is the right
# direction to be wrong in: an under-corrected object is merely still too small,
# while an over-corrected one is a person three times the height of the terrain
# they are standing on.
_RATIO_QUANTILE: float = 0.4


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float = _RATIO_QUANTILE) -> float:
    """Weighted quantile -- robust to the truncated/occluded boxes that a plain
    weighted mean would let drag the whole bin."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0:
        return float(np.quantile(v, q))
    cumulative = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cumulative, total * q)), len(v) - 1)])


class ObjectScaleModel:
    """A fitted depth -> size-correction curve.

    Stored as knots and evaluated by interpolation in log-depth/log-factor space,
    matching how the correction physically behaves: depth compression is
    multiplicative and roughly geometric in distance, so a curve that is linear in
    log-log is the natural shape, and interpolating there keeps the factor strictly
    positive without any special-casing.

    Deliberately flat-clamped outside the fitted range, unlike the depth curve in
    panorama_depth/calibration.py which extrapolates. That curve has to extrapolate
    because real terrain genuinely continues past its fit. Here, extrapolating would
    mean inventing an ever-growing correction for the objects we have the *least*
    evidence about -- the far ones, where anchors are sparsest and truncated boxes
    are most common. Holding the last measured factor is the conservative failure.
    """

    def __init__(self, depths: np.ndarray, factors: np.ndarray, anchor_count: int):
        self.depths = np.asarray(depths, dtype=np.float64)
        self.factors = np.asarray(factors, dtype=np.float64)
        self.anchor_count = anchor_count
        self._log_depths = np.log(np.maximum(self.depths, 1e-6))
        self._log_factors = np.log(np.maximum(self.factors, 1e-6))

    def factor_at(self, depth: float) -> float:
        if not np.isfinite(depth) or depth <= 0:
            return 1.0
        if len(self._log_depths) == 1:
            return float(self.factors[0])
        return float(np.exp(np.interp(np.log(depth), self._log_depths, self._log_factors)))

    def describe(self) -> str:
        pairs = ", ".join(f"{d:.0f}m -> {f:.2f}x" for d, f in zip(self.depths, self.factors))
        return f"[{pairs}] from {self.anchor_count} anchors"

    def to_dict(self) -> dict:
        return {
            "anchor_count": self.anchor_count,
            "knots": [
                {"depth_m": float(d), "factor": float(f)}
                for d, f in zip(self.depths, self.factors)
            ],
        }


def collect_anchor(
    index: int,
    cls: Optional[str],
    metadata: dict,
    depth: float,
    measured_height: float,
) -> Optional[ScaleAnchor]:
    """Build a ScaleAnchor if this object is trustworthy enough to measure with.

    Rejections are all about whether the *class label* can be believed, because the
    prior is keyed entirely on it -- a misread label does not add noise, it adds a
    confident wrong number. Synthetic instances are excluded for a different reason:
    DistributionSynthesisStage sampled their size from real exemplars that were
    themselves measured under compression, so treating them as measurements would
    feed the fit its own output.
    """
    if not cls or cls not in OBJECT_HEIGHT_PRIORS:
        return None
    if metadata.get("synthetic") or metadata.get("position_only") or metadata.get("low_confidence"):
        return None
    if not np.isfinite(depth) or depth <= 0:
        return None
    if not np.isfinite(measured_height) or measured_height <= 1e-4:
        return None

    prior_height, class_weight = OBJECT_HEIGHT_PRIORS[cls]
    # Fold the classifier's own confidence into the anchor weight when it is
    # available, so a marginal `car` counts for less than a certain one.
    confidence = metadata.get("confidence")
    if isinstance(confidence, (int, float)) and 0.0 < float(confidence) <= 1.0:
        class_weight *= float(confidence)

    return ScaleAnchor(
        index=index,
        cls=cls,
        depth=float(depth),
        measured_height=float(measured_height),
        prior_height=float(prior_height),
        weight=float(class_weight),
    )


def fit_object_scale(
    anchors: Sequence[ScaleAnchor],
    num_bins: int = 6,
    min_anchors: int = 6,
    min_per_bin: int = 3,
    max_correction: float = 8.0,
    trim_percentile: float = 10.0,
) -> Optional[ObjectScaleModel]:
    """Fit a monotonic depth -> correction curve from per-object size anchors.

    Returns None when the scene cannot support a fit, which is a real and expected
    outcome: a wilderness capture may contain no person, car, or bench anywhere.
    None means "leave every object exactly as it is" -- never "guess".

    The bin count adapts down to the anchor supply, and collapses to a single global
    factor when there are too few anchors to resolve any depth dependence. That
    degenerate case is still useful, and is exactly the uniform scale factor you
    would derive by hand.
    """
    usable = [a for a in anchors if np.isfinite(a.ratio) and a.ratio > 0]
    if len(usable) < min_anchors:
        return None

    depths = np.array([a.depth for a in usable], dtype=np.float64)
    ratios = np.array([a.ratio for a in usable], dtype=np.float64)
    weights = np.array([a.weight for a in usable], dtype=np.float64)

    # Trim in log-ratio space before fitting. The dominant error mode is a box
    # clipped by the panorama seam or by an occluder, which reports a fraction of the
    # object's true height and therefore an enormously inflated ratio -- a far bigger
    # distortion than any genuine within-class variation.
    log_ratios = np.log(ratios)
    lo, hi = np.percentile(log_ratios, [trim_percentile, 100.0 - trim_percentile])
    keep = (log_ratios >= lo) & (log_ratios <= hi)
    if keep.sum() >= min_anchors:
        depths, ratios, weights = depths[keep], ratios[keep], weights[keep]

    bin_count = int(np.clip(len(depths) // max(min_per_bin, 1), 1, num_bins))

    if bin_count <= 1:
        factor = float(np.clip(_weighted_quantile(ratios, weights), 1.0 / max_correction, max_correction))
        return ObjectScaleModel(
            np.array([float(np.median(depths))]), np.array([factor]), len(usable)
        )

    edges = np.quantile(depths, np.linspace(0.0, 1.0, bin_count + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_index = np.clip(np.digitize(depths, edges) - 1, 0, bin_count - 1)

    centres: list[float] = []
    factors: list[float] = []
    for b in range(bin_count):
        in_bin = bin_index == b
        if not np.any(in_bin):
            continue
        centres.append(float(np.median(depths[in_bin])))
        factors.append(_weighted_quantile(ratios[in_bin], weights[in_bin]))

    if not centres:
        return None
    if len(centres) == 1:
        factor = float(np.clip(factors[0], 1.0 / max_correction, max_correction))
        return ObjectScaleModel(np.array(centres), np.array([factor]), len(usable))

    order = np.argsort(centres)
    centres_sorted = np.array(centres)[order]
    factors_sorted = np.array(factors)[order]

    # Correction must not shrink with distance. Depth compression is monotonic in
    # depth by construction (both the extrapolation cap and _compress_far_range are),
    # so any bin that says otherwise is reporting noise, not a real reversal -- most
    # often a single sparse far bin carried by one truncated box.
    factors_sorted = np.maximum.accumulate(factors_sorted)
    factors_sorted = np.clip(factors_sorted, 1.0 / max_correction, max_correction)

    return ObjectScaleModel(centres_sorted, factors_sorted, len(usable))
