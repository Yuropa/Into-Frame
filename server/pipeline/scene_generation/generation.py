import json
from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import (
    ENVIRONMENT_CATEGORIES as _ENV_CATEGORIES, normalize_category,
)
from pipeline.object_typing.categories import GRASS_TUFT_CATEGORY
from pipeline.scene_generation.projection import (
    height_map_y_at,
    mesh_y_at,
    terrain_local_xz,
    unproject_bbox,
    unproject_bbox_equirect,
)
from pipeline.scene_generation.object_scale import collect_anchor, fit_object_scale, is_metric_authored
from util.lod_metrics import angular_height_px
from scene.scene import Scene
from scene.object import Object3D
import numpy as np


class SceneGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        eye_height_meters: float = 1.8,
        min_mesh_height_fraction: float | None = None,
        min_mesh_angular_px: float = 250.0,
        viewer_px_per_degree: float = 34.0,
        viewer_move_radius_m: float = 2.0,
        object_scale_correction: bool = True,
        object_scale_max_correction: float = 8.0,
        object_scale_min_anchors: int = 6,
        object_scale_num_bins: int = 6,
        metric_height_m: dict[str, float] | None = None,
        metric_height_jitter: float = 0.25,
        overlap_rejection: bool = True,
        overlap_min_separation: float = 0.35,
        overlap_min_separation_overrides: dict[str, float] | None = None,
        overlap_exempt_classes: list[str] | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Sent to the client as the target world-space depth of the terrain
        # center below the viewer; the client pushes the whole scene down to match.
        self.eye_height_meters = eye_height_meters
        # Floor on a category mesh's own vertical extent, as a fraction of its
        # largest, below which the instance is billboarded rather than scaled to
        # match its detection's height. Resolved here rather than as a default
        # argument because _MIN_MESH_HEIGHT_FRACTION is defined below this class;
        # see it for the four-capture sweep behind the value.
        self.min_mesh_height_fraction = (
            _MIN_MESH_HEIGHT_FRACTION if min_mesh_height_fraction is None
            else float(min_mesh_height_fraction)
        )
        # Bake-time mesh-vs-billboard cutoff, as the screen height the instance
        # subtends rather than its distance from the camera. An instance that
        # renders at least this many display pixels tall uses its bucket's 3D
        # mesh; anything smaller takes the card or billboard LOD. See
        # util/lod_metrics.py for why projected size and not distance, and why
        # height and not area.
        #
        # This replaced mesh_lod_distance_m and its per-class override table. The
        # override table only ever had one entry -- grass_tuft, pinned at 3 m to
        # undo a global cutoff tuned for subjects you look AT -- and that entry is
        # exactly the mismatch a size metric removes: grass measures p50 44 px /
        # p90 111 px on the Rainier capture and falls below any useful threshold
        # on its own, with no class named anywhere.
        self.min_mesh_angular_px = float(min_mesh_angular_px)
        # Display angular resolution the threshold above is quoted against. 34
        # px/deg is Vision Pro class. This exists so the threshold can be read as
        # what it actually is (a resolvability limit) and so it tracks the target
        # display instead of silently meaning something else on new hardware --
        # the product min_mesh_angular_px / viewer_px_per_degree is the real
        # quantity, ~7.4 deg at the defaults.
        self.viewer_px_per_degree = float(viewer_px_per_degree)
        # How far the viewer may walk from the capture origin. Angular size is
        # evaluated at the CLOSEST they can get (distance minus this), not at the
        # bake position -- see angular_height_px. The old distance cutoff did not
        # need this because distance changes by at most the distance walked;
        # angular size goes as 1/d and does not share that property.
        self.viewer_move_radius_m = float(viewer_move_radius_m)
        # Learn a depth -> size-correction curve from objects with known real-world
        # heights and apply it to every placed object's extent. See
        # scene_generation/object_scale.py for why object size arrives compressed and
        # why the correction belongs here rather than in the depth map.
        self.object_scale_correction = object_scale_correction
        # Ceiling (and, reciprocally, floor) on that correction. Bounds the damage
        # when a scene's anchors are few and unrepresentative -- a wrong 8x is bad,
        # an unbounded one takes an object across the whole terrain grid.
        self.object_scale_max_correction = object_scale_max_correction
        # Minimum anchors before any correction is attempted. Below this the fit is
        # skipped entirely and objects keep their uncorrected size.
        self.object_scale_min_anchors = object_scale_min_anchors
        # Depth bins used to resolve the correction's depth dependence. Adapts down
        # automatically when there are too few anchors to fill them.
        self.object_scale_num_bins = object_scale_num_bins
        # {class: real-world height in metres}. For classes where the detector boxes a
        # PART of what has to be placed, so the measurement is accurate but describes
        # the wrong object -- see the metric-height block in run(). Empty by default:
        # measuring is the right answer everywhere it works, and each entry here is an
        # explicit claim that it does not for that class.
        self.metric_height_m = dict(metric_height_m or {})
        # Fractional spread around the authored height, so an authored class does not
        # render as a field of identical clones. Mirrors grass's tuft_height_jitter.
        self.metric_height_jitter = float(metric_height_jitter)
        # Drop an object whose footprint lands too far inside one already placed.
        # Objects arrive from several independent producers -- detections, painted
        # distribution points, per-class scatter -- none of which can see what the
        # others put down, so nothing upstream is in a position to prevent two of
        # them occupying the same ground.
        self.overlap_rejection = bool(overlap_rejection)
        # Centre separation as a fraction of the two footprint radii summed: 1.0 is
        # exactly touching, 0 is concentric. An object is rejected below this.
        # Deliberately well under 1 -- real scenes are full of legitimately
        # overlapping footprints (a chair tucked under a table, a boat against a
        # dock), and the target here is the pathological case, not contact.
        self.overlap_min_separation = float(overlap_min_separation)
        # Per-class override of that separation, {class_name: fraction}. The global
        # value is deliberately permissive because the scene is mostly discrete
        # objects whose footprints legitimately touch. It is far too permissive for
        # a class placed as a POPULATION, where the same "pathological case only"
        # reasoning stops applying: those instances come from several producers at
        # once (real detections, painted distribution points) and are dense enough
        # that ordinary overlap IS the artefact.
        #
        # Measured on the Rainier capture. 44 real flower detections land within
        # 1.5 m of the camera (median 0.57 m) because the panorama's near-nadir band
        # over-splits one flower mass into dozens of boxes, and Distribution
        # Synthesis then paints 275 more on top of them. Their median centre
        # separation is 0.118 m against a median footprint width of 0.137 m -- so
        # they overlap by default, while the global 0.35 only rejects centres inside
        # 5.2 cm and passed 63 of the 115 instances within 3 m as interpenetrating.
        #
        # 0.8 rather than 1.0 (exactly tangent): real vegetation does grow into its
        # neighbours and demanding tangency reads as a planted grid, plus it starts
        # culling real detections rather than the painted surplus -- at 1.0, 20 of
        # 44 real flowers are dropped by each other, against 14 at 0.8. Measured
        # effect at 0.8: 63 -> 16 interpenetrating within 3 m, near-field flower
        # density 21.6 -> 12.1 per m2, 82 vegetation instances rejected of which 56
        # are synthetic.
        #
        # NOTE what this does NOT fix: the near field stays far denser than the rest
        # of the disc (12.1 per m2 inside 1 m against 0.1 past 5 m). That gradient is
        # not overlap -- it is 44 real detections genuinely occupying the first
        # metre, and Distribution Synthesis painting at the density they imply out to
        # its full_density_radius_m floor of 3 m. Removing it means either deleting
        # real detections or emptying the meadow (see the group_radius block in
        # distribution_synthesis.py for why that radius only ever extends), so it is
        # deliberately left alone here.
        self.overlap_min_separation_overrides = dict(overlap_min_separation_overrides or {})
        # Classes exempt from the test entirely, and from the occupancy set that
        # feeds it. Ground cover is the case this exists for: it is placed BY a
        # density, thousands of instances deliberately interpenetrating to read as
        # continuous cover, and testing it would both reject most of it and make the
        # check O(n^2) over the whole population instead of over the few dozen
        # discrete objects that actually matter.
        self.overlap_exempt_classes = frozenset(
            overlap_exempt_classes if overlap_exempt_classes is not None else [GRASS_TUFT_CATEGORY]
        )

# Objects whose estimated real-world largest dimension exceeds this threshold (meters)
# are assumed to be large scene elements (mountains, hills, sky) and are skipped.
#
# This is a backstop, not the primary defence: an object this big is nearly always
# a background crop whose depth the panorama map could not measure, and Panorama
# Asset Generation's max_background_fraction gate rejects those on the evidence
# rather than on the symptom. What survives to here is the residue -- a box with a
# real depth sample that still unprojects absurdly large.
#
# Lowered 40 -> 25. At 40 the gate only ever caught the truly degenerate (a 349 m
# crop of sky on a Shark Fin Cove capture) while passing a 38.1 m "lighthouse" of
# blank sky, and on a Paris capture a 38.5 m and a 21.3 m "boat" on the Seine. 25 m
# still clears everything those captures place legitimately -- the largest are a
# 14.1 m building at 31.8 m and a 12.1 m riverboat -- and it is above the height of
# any class in OBJECT_HEIGHT_PRIORS by a wide margin. Raise it for a capture whose
# subject genuinely is a single large structure filling the frame.
_MAX_OBJECT_SIZE_M = 25.0

# A category mesh is scaled to match its instance's detected HEIGHT (see the mesh
# branch in run()), which needs its own vertical extent as the divisor. Below this
# fraction of its largest extent the mesh is effectively a flat sheet, that divisor
# is noise, and matching height would blow the other axes up by 1/extent_y -- so the
# instance is billboarded instead.
#
# Config default only; the live value is SceneGenerationConfiguration's
# min_mesh_height_fraction. See config.yaml for the sweep this was chosen from.
#
# Lowered 0.25 -> 0.10. At 0.25 this is not the sheet test it reads as: dividing by
# the LARGEST extent makes it "is this taller than a quarter of its length?", which a
# low-profile subject fails by being correctly shaped. On the Paris capture it
# rejected all four boat reconstructions -- category_mesh_boat_0 is [1.000, 0.560,
# 2.613], a 2.6-long, 1.0-wide, 0.56-tall riverboat, and it missed the bar by 0.036 --
# and with them all 27 boat instances on the Seine, which is the whole subject of that
# capture. building_0 (0.205) went the same way.
#
# Replayed over every instance of all four landscape captures, the threshold turns out
# to be almost entirely inert. Instances admitted to a mesh, by threshold:
#
#     thresh   Paris     Rainier    Iceland   Shark Fin
#     0.25     49/86     294/473    0/80      0/42
#     0.20     54/86     294/473    0/80      0/42
#     0.15     69/86     294/473    0/80      0/42
#     0.10     70/86     294/473    0/80      0/42
#     0.00     70/86     294/473    0/80      0/42
#
# Rainier does not move at all -- not the count and not the class histogram (211
# flower, 46 plant, 15 person, 11 tree, 6 rock, 3 bush, 2 boat at 0.25 and at 0.00
# alike) -- and neither do Iceland or Shark Fin. Only Paris ever depended on it.
#
# What makes that safe is that the three gates below already catch everything this one
# was raised to catch. Rainier's own Y-gate rejections all fail the aspect test
# independently: aircraft_0 at 12.8:1 against its box, animal_0 at 11.9:1, and ship_0,
# whose extents are [0.926, 1.000, 41.182] -- 41x longer in Z than tall, and note it
# passes the THICKNESS test at min/mid 0.926, so aspect is the only thing standing
# between that mesh and the scene. The original 47.2 m other_0 sheet is likewise an
# aspect and rendered-size failure, not a Y failure.
#
# 0.10 rather than 0.00 because the divisor still wants a floor: extent_y is what
# scale divides by, and while the rendered-size backstop bounds the result, leaving
# the ratio unguarded means a near-degenerate mesh reaches that backstop by way of a
# 1000x scale. 0.10 sits below Paris's lowest legitimate keeper (boat_3 at 0.156, and
# table_0 at 0.134) with room, and 2.5x under the old value.
_MIN_MESH_HEIGHT_FRACTION = 0.10

# "Is this a flat picture, or does it have volume in the round?", asked as the thinnest
# extent over the SECOND largest -- which is what catches a mesh that is a flat CARD
# rather than a flat pancake.
#
# _MIN_MESH_HEIGHT_FRACTION above only ever looks at extent_y, so it sees a horizontal
# sheet and is blind to a vertical one -- an upright card has extent_y == its largest
# extent and scores a perfect h_frac of 1.000. Measured on the Rainier capture,
# category_mesh_tree_0 came back [0.965, 1.000, 0.0829]: a panel twelve times wider
# than it is thick, h_frac 1.000, and 67 of its 108 instances were placed, rendering
# up to 5.3 m tall x 5.1 m wide x 0.44 m thick. That is the "large rectangular slab"
# in the scene, and it is a tree. category_mesh_forest_0 (0.0814) is the same shape.
#
# The cause is upstream and not fixable here: SAM3D reconstructs a shallow single-view
# relief from a foliage crop, and Mesh.repair()'s Poisson pass seals that relief into a
# thin closed shell. A billboard is the honest representation of a subject that was
# never reconstructed in the round, and rejecting here is what falls back to one.
#
# Measured against the LARGEST extent this reads as a sheet test but is really an
# "is it tall and narrow" test, and those are different shapes: a sheet has one small
# axis and two large (a tree card, [0.95, 1.00, 0.10]), while a mast, a trunk or a
# person has TWO small axes ([0.15, 1.00, 0.14]). Only the second largest tells them
# apart, and dividing by the largest cannot: on the 2026-08-08 run the sheets sat at
# 0.044-0.111 of their largest and the keepers at 0.131-0.866, a gap of 1.18x, so the
# threshold sat inside SAM3D's own run-to-run drift -- the same tree bucket measured
# 0.083 one run and 0.097 the next.
#
# Over the second largest the two populations separate by 6x instead: sheets score
# 0.082 (forest), 0.084 (statue), 0.101 (tree), 0.108 (rock), 0.159 (bush), 0.207
# (bench), 0.225 (aircraft); keepers score 0.657 (table), 0.811 (traffic_light), 0.850
# (boat), 0.866 (a grass card), 0.878-0.994 (every flower, person, building, tower,
# plant). 0.4 sits in the middle of that gap with ~4x of margin on either side, which
# is what makes it robust to a reconstruction that comes back slightly different.
#
# It also has to admit a deliberately narrow card: a 3-plane fan is [W, 1.0, 0.866W]
# for any width W, which scores a constant 0.866 here and would have been rejected
# outright by the old test for any conifer narrower than about 1:8.
_MIN_MESH_THICKNESS_FRACTION = 0.4

# A reconstructed category mesh describes the same object its instance's 2D box
# measured, so the two have to agree about that object's shape. This is the ratio
# between the mesh's own width:height and the detection's, above which they don't:
# the mesh is a bad reconstruction of something else, and the instance is billboarded
# rather than scaled to fit.
#
# This is the check that actually catches unbounded mesh scaling, because it looks at
# the mismatch rather than at either side alone -- and both `other` instances that
# blew up on the Rainier capture fail it by two orders of magnitude:
#
#     idx 77   detection 0.70 x 3.00 m  -> 0.23:1      mesh 15.7:1   ratio 67x
#     idx 7    detection 1.81 x 1.01 m  -> 1.79:1      mesh 15.7:1   ratio 8.8x
#
# while the legitimate meshes sit near 1: the same capture's flowers are a 0.42:1
# mesh against a ~0.36:1 detection, a ratio of 1.2. 4.0 leaves a wide margin over
# that and still rejects everything measured to have gone wrong. It would also have
# caught the 38.1 m Shark Fin Cove "lighthouse", which was a sky crop meshed as a
# broad flat shell against a tall narrow box.
_MAX_MESH_ASPECT_RATIO = 4.0

# How horizontally elongated a mesh must be before the aspect test above will judge
# it against its END-ON silhouette instead of its broadside one.
#
# The relaxation only makes sense for a shape that HAS a distinct end-on view. For a
# compact or upright mesh the two horizontal extents are nearly equal and the
# relaxation is free permissiveness. Measured elongation, max(x,z)/min(x,z):
#
#     Paris   boat_3      7.86     boat_2  3.92     boat_0  2.41     building_0 8.79
#     Rainier tree_0      1.76     plant_2 1.09     flower_0 1.11    person_0   1.05
#
# 2.0 sits in the gap. Above it are the long, low subjects the gate was mis-reading;
# below it are the upright ones it was reading correctly all along.
#
# Replayed over every instance of all four landscape captures, with the band form
# below: Rainier and Shark Fin come out BYTE-IDENTICAL (294/473 and 0/42, same class
# histograms), Paris gains 6 boats on the 2026-08-24 run and 7 boats plus 2 landmark
# on the 2026-08-18 one, and no capture loses an instance.
#
# The one cost is Iceland, which gains 3 `animal` meshes (category_mesh_animal_2,
# [1.000, 0.560, 2.833] -- elongated enough to qualify, and it clears the height and
# sheet gates on its own). Those are hallucinated detections and this makes them
# meshes rather than billboards. Judged acceptable because they were junk either way
# and the junk vetoes are the right place to remove them, but it is a real cost and
# worth re-checking if Iceland's typing improves.
_MIN_MESH_ELONGATION_FOR_END_ON = 2.0



def mesh_instance_scale(
    mesh_extents: "np.ndarray",
    width: float,
    height: float,
    min_height_fraction: float = _MIN_MESH_HEIGHT_FRACTION,
) -> tuple[float, str | None, str | None]:
    """Uniform scale that renders `mesh_extents` at the instance's detected size.

    Returns (scale, rejection_reason, gate). A non-None reason means this mesh does
    not describe this detection and the caller should billboard the instance instead
    -- the scale is still returned for the log line, not for use. `gate` names which
    of the four tests fired, as a stable short slug ("height", "sheet", "aspect",
    "rendered_size", "degenerate"), so a run's rejections can be attributed without
    re-parsing the prose reason. It is None exactly when the reason is.

    Scale is set by HEIGHT, for the reason object_scale.py's prior table documents:
    an equirectangular box's horizontal extent depends on the object's yaw relative
    to the camera (a tree seen through a gap vs. broadside differ by a lot) while its
    vertical extent does not. Width is the unreliable axis, so it must not be what
    sets scale.

    The gates below exist because that divisor is unbounded. Dividing by a small
    extent_y is how a mesh of one object becomes a scene-spanning sheet over another,
    and neither the detected size cull (_MAX_OBJECT_SIZE_M, which reads the box, not
    the render) nor the classifier that assigned the class can see it happen. Note
    that the height gate is the WEAKEST of the four and deliberately so -- see
    _MIN_MESH_HEIGHT_FRACTION for the sweep showing the other three do the work.
    """
    extent_y = float(mesh_extents[1])
    extent_max = float(max(mesh_extents))
    extent_x, extent_z = float(mesh_extents[0]), float(mesh_extents[2])
    if extent_max <= 0.0 or extent_y <= 0.0:
        return 0.0, "mesh has no extent", "degenerate"

    if extent_y < min_height_fraction * extent_max:
        return (
            float(height) / extent_y,
            f"flat in Y (extent_y {extent_y:.4f} is {extent_y / extent_max:.3f} of "
            f"its largest {extent_max:.4f}, under {min_height_fraction})",
            "height",
        )

    # Flat in ANY axis -- an upright card clears the vertical test above with a perfect
    # h_frac and is still a sheet. See _MIN_MESH_THICKNESS_FRACTION.
    extent_sorted = sorted(float(v) for v in mesh_extents)
    extent_min, extent_mid = extent_sorted[0], extent_sorted[1]
    if extent_mid > 0.0 and extent_min < _MIN_MESH_THICKNESS_FRACTION * extent_mid:
        return (
            float(height) / extent_y,
            f"a flat sheet (thinnest axis {extent_min:.4f} is "
            f"{extent_min / extent_mid:.3f} of its second largest {extent_mid:.4f}, "
            f"under {_MIN_MESH_THICKNESS_FRACTION})",
            "sheet",
        )

    # Aspect agreement: does the detection's width:height match a silhouette this
    # mesh could actually cast? Compared as a ratio-of-ratios so it stays symmetric.
    #
    # The original test used max(extent_x, extent_z) / extent_y -- the mesh seen
    # BROADSIDE -- and so assumed the detection caught the mesh along its longest
    # horizontal axis. Anything photographed end-on fails that on its merits.
    # Measured on the 2026-08-24 Paris run it rejected 5 of the 7 boat instances that
    # reached it, the hero riverboat among them: idx 66 is a bow-on detection whose
    # box is 334x355 px -- nearly square, correctly, because you are looking at the
    # stern -- against a correct 1.00 x 0.58 x 2.41 reconstruction. The gate took the
    # 2.41 LENGTH, called the mesh 4.13:1, compared it to the box's 0.72:1 and
    # rejected at 5.8x. category_mesh_building_0 failed the same way at 20.3x.
    #
    # So a box NARROWER than broadside is measured against the end-on extent instead.
    # But only for a mesh that is actually elongated horizontally, which is the whole
    # of the fix and the reason for _MIN_MESH_ELONGATION_FOR_END_ON: for a compact or
    # upright mesh the two extents are nearly equal, "end-on" is not a meaningfully
    # different view, and applying the relaxation there just weakens the gate for
    # everything. Rainier is what forces that restraint -- ungated, the same
    # relaxation admits category_mesh_tree_0, normalised extents
    # [0.567, 0.285, 1.000] (a tree reconstructed lying down) and 15 more instances
    # of it, which is exactly the flat-slab artefact this gate family exists to stop.
    #
    # A box WIDER than broadside is left exactly as it was: no yaw can explain it, so
    # it is real evidence, and touching that branch costs well-formed upright meshes
    # the old test passed.
    if height > 1e-6 and width > 1e-6:
        aspect_broadside = max(extent_x, extent_z) / extent_y
        aspect_end_on = min(extent_x, extent_z) / extent_y
        elongation = max(extent_x, extent_z) / max(min(extent_x, extent_z), 1e-6)
        box_aspect = float(width) / float(height)

        # For an elongated mesh, every silhouette between end-on and broadside is
        # reachable by some yaw, so a box anywhere in that band is no evidence at
        # all. For anything else the band collapses to a point and this reduces
        # exactly to the original symmetric test.
        if elongation >= _MIN_MESH_ELONGATION_FOR_END_ON:
            reachable_lo, reachable_hi = aspect_end_on, aspect_broadside
        else:
            reachable_lo = reachable_hi = aspect_broadside

        if box_aspect < reachable_lo:
            disagreement, reference, view = reachable_lo / box_aspect, reachable_lo, "end-on"
        elif box_aspect > reachable_hi:
            disagreement, reference, view = box_aspect / reachable_hi, reachable_hi, "broadside"
        else:
            disagreement, reference, view = 1.0, box_aspect, "within range"
        if disagreement > _MAX_MESH_ASPECT_RATIO:
            return (
                float(height) / extent_y,
                f"aspect {reference:.2f}:1 ({view}) disagrees with the detection's "
                f"{box_aspect:.2f}:1 by {disagreement:.1f}x "
                f"(over {_MAX_MESH_ASPECT_RATIO:.1f}x)",
                "aspect",
            )

    scale = float(height) / extent_y

    # Backstop on the RENDERED size. _MAX_OBJECT_SIZE_M above culls on the detected
    # box, which is the wrong quantity for this failure and always passes it: the
    # 47.2 m Rainier sheet was a 3.0 m detection, well inside a 25 m cull. What
    # reaches the scene is scale * extents, so that is what has to be bounded.
    #
    # This is also what bounds the height gate's relaxation: a low-profile mesh scaled
    # by a small extent_y arrives here at a large scale, and this is where it stops.
    rendered_max = scale * extent_max
    if rendered_max > _MAX_OBJECT_SIZE_M:
        return scale, (
            f"would render {rendered_max:.1f} m across at {scale:.1f}x, "
            f"over the {_MAX_OBJECT_SIZE_M:.0f} m limit"
        ), "rendered_size"

    return scale, None, None


def _mesh_geometry_record(
    mesh_geometry: dict, mesh_key: str, cls: str, extents, idx: int, scale: float
) -> dict:
    """Create-or-update this mesh's entry in the per-run geometry report.

    Shared by the reconstruction path and the card path so a scene where every
    instance ended up on a card still reports what was placed and at what size --
    reading "no record" as "nothing rendered" is exactly the confusion
    _log_mesh_geometry exists to prevent.
    """
    record = mesh_geometry.get(mesh_key)
    if record is None:
        ex = [float(v) for v in extents]
        largest = max(ex) or 1.0
        ordered = sorted(ex)
        record = mesh_geometry[mesh_key] = {
            "class": cls,
            "extents": [round(v, 4) for v in ex],
            "height_fraction": round(ex[1] / largest, 4),
            # Over the SECOND largest, matching what _MIN_MESH_THICKNESS_FRACTION
            # actually tests.
            "thickness_fraction": round(ordered[0] / (ordered[1] or 1.0), 4),
            "aspect": round(max(ex[0], ex[2]) / ex[1], 3) if ex[1] > 0 else None,
            "instances": 0,
            "rejected": 0,
            # Running extremes, not per-instance lists. Ground cover puts tens of
            # thousands of instances through one card mesh (16,031 grass tufts
            # across 3 on the Rainier capture), and keeping every value would
            # write a debug file of raw numbers that buries the handful that
            # matter. The extremes plus a few named examples are what a blow-up
            # actually shows up in.
            "scale_min": None, "scale_max": None,
            "rendered_min_m": None, "rendered_max_m": None,
            "examples": [],
        }
    record["instances"] += 1
    scale_v = round(float(scale), 3)
    rendered_v = round(float(scale) * max(record["extents"]), 3)
    for lo, hi, value in (
        ("scale_min", "scale_max", scale_v),
        ("rendered_min_m", "rendered_max_m", rendered_v),
    ):
        if record[lo] is None or value < record[lo]:
            record[lo] = value
        if record[hi] is None or value > record[hi]:
            record[hi] = value
    # Keep the biggest few by rendered size -- those are the ones to look at, and a
    # reader wants an index to go and find them by.
    record["examples"].append({"idx": idx, "scale": scale_v, "rendered_m": rendered_v})
    record["examples"].sort(key=lambda e: -e["rendered_m"])
    del record["examples"][8:]
    return record


class SceneGenerationStage(PipelineStage):
    """
    Assembles the final 3D scene by unprojecting each detected object's bounding box
    into world space (using depth + camera parameters) and placing its mesh or billboard
    at the computed position. Also adds the terrain mesh (if present) at the origin.

    Input keys:
      SemanticKey.INPUT        → ContextKey.INPUT          (Image, used for bbox scale)
      SemanticKey.DEPTH        → ContextKey.DEPTH          (Depth, for world-space placement)
      SemanticKey.INTRINSICS   → ContextKey.INTRINSICS     (CameraIntrinsics)
      SemanticKey.EXTRINSICS   → ContextKey.EXTRINSICS     (CameraExtrinsics)
      SemanticKey.PANORAMA     → ContextKey.PANORAMA       (Image, used as scene skybox)
      SemanticKey.OBJECT_COUNT → ContextKey.OBJECT_COUNT   (int)

    Dynamic context keys per object (index i):
      crop_{i}                            → Image  (object texture)
      metadata_{i}                        → object ({"box": [...], "score": float,
                                                       "class": str, "bucket": int})
      category_mesh_{class}_{bucket}      → Mesh   (optional, from Panorama Asset
                                                       Generation; used when this
                                                       instance renders at least
                                                       min_mesh_angular_px tall,
                                                       card/billboard otherwise)
      "billboard_pools" (generic object)  → {"{class}::{bucket}": [crop idx, ...]}
                                             (curated top-K pool per bucket, from
                                             Panorama Asset Generation)

    Optional:
      ContextKey.TERRAIN_MESH        → Mesh        (placed at origin if present)
      ContextKey.WATER_MESH          → Mesh        (placed at origin if present)
      ContextKey.HEIGHT_MAP          → Depth       (dense grid the terrain mesh was
                                        built from; sampled as the ground-height
                                        fallback when the mesh raycast misses)
      ContextKey.HEIGHT_MAP_PARAMS   → dict        (that grid's extent, for the above)
      ContextKey.TERRAIN_FORMATIONS  → list[dict]  (each references a dynamic
                                        "terrain_formation_{id}" Mesh key, already
                                        in absolute world space -- placed at origin)

    Output key (SemanticKey.OUTPUT) → ContextKey.SCENE (Scene)
    """

    def __init__(self, config: SceneGenerationConfiguration) -> None:
        super().__init__(config)
        self._gen = None

    @classmethod
    def config_class(cls) -> type[SceneGenerationConfiguration]:
        return SceneGenerationConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.OBJECT_COUNT: ContextKey.OBJECT_COUNT,
            SemanticKey.EXTRINSICS: ContextKey.EXTRINSICS,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.DEPTH: ContextKey.DEPTH,
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.SCENE
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        count_key, extrinsics_key, intrinsics_key, input_key, depth_key, panorama_key, output_key = self._resolved_keys()

        object_count = context.input_object(count_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        extrinsics = context.input_extrinsics(extrinsics_key)
        depth = context.input_depth(depth_key)
        input = context.input_image(input_key)
        # Objects were detected on the ORIGINAL panorama, so their bounding boxes must be
        # unprojected using depth measured on that same original panorama (PANORAMA_OBJECT_DEPTH)
        # — not PANORAMA_DEPTH, which is recomputed after foreground inpainting and no longer
        # has real depth at removed-object locations (it has the reconstructed background
        # instead), which would silently misplace exactly the near-camera objects most likely
        # to have been removed.
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(panorama_key)
        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)

        # Every mesh an object can stand on. Formations are separate landmasses sitting
        # ON TOP of the base terrain, which is deliberately depressed beneath them (see
        # TerrainMeshGenerator's formation_depression_m) -- snapping to terrain_mesh
        # alone therefore buries anything standing on one, by the depression depth plus
        # the formation's own height. Highest surface at the query point wins.
        #
        # The water surface is in here for the same reason. The base terrain is NOT
        # holed at water -- TerrainMeshGenerator carves its water vertices
        # water_depression_m (0.5 m) BELOW the water surface so an animated water plane
        # can never expose the lakebed, and emits WATER_MESH over those same faces at
        # the un-depressed elevation. So the raycast does hit something at every water
        # cell; it just hits the lakebed, and everything standing on a shoreline or in
        # shallow water came out exactly that depression depth underground. Water only
        # ever wins the max() where that carve happened, since WATER_MESH has no faces
        # anywhere else.
        ground_meshes = [terrain_mesh, context.input_mesh(ContextKey.WATER_MESH)]
        for formation in context.input_object(ContextKey.TERRAIN_FORMATIONS) or []:
            ground_meshes.append(context.input_mesh(formation["mesh_key"]))
        ground_meshes = [m for m in ground_meshes if m is not None]

        # Dense grid fallback for the raycast (see height_map_y_at). Same frame as the
        # ground meshes' own vertices, so it is queried with the same terrain-local XZ.
        height_map = context.input_depth(ContextKey.HEIGHT_MAP)
        height_map_grid_m = float(
            (context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get("grid_size_meters", 0.0)
        )
        if height_map is None or height_map_grid_m <= 0.0:
            height_map = None

        # The camera's own world XZ, which terrain_local_xz has to remove before undoing
        # the yaw -- the terrain grids are all built with the camera at their origin while
        # object positions carry the full extrinsics translation.
        camera_xz = (
            (float(extrinsics.translation[0]), float(extrinsics.translation[2]))
            if extrinsics is not None else (0.0, 0.0)
        )

        # How each placed object got its ground height. A missed raycast used to fall
        # silently through to the object's raw unprojected Y, which is the one value
        # guaranteed NOT to agree with the reconstructed terrain -- so a wholesale snap
        # failure looked identical to a scene that simply had no terrain, and read in
        # the client as objects floating above or buried under the ground.
        snap_counts = {"mesh": 0, "height_map": 0, "unsnapped": 0}
        unsnapped_examples: list[str] = []

        # Category meshes refused for an instance by mesh_instance_scale, per mesh key.
        # Summarised after the loop because the per-instance warning fires once per
        # instance and a bad bucket mesh is shared by all of them -- the count is the
        # signal that a whole reconstruction is wrong, not just one placement.
        mesh_rejections: dict[str, int] = {}

        # Per category mesh: its own normalised shape, and what every instance that
        # used it actually renders at. A bucket mesh is shared, so one bad
        # reconstruction is a defect repeated across the scene -- and the quantity
        # that matters (scale * extents) appears nowhere else in a run's output. The
        # 47 m Rainier sheet was a 3.0 m detection scaled 47.2x by a mesh 0.064 tall;
        # every number in that sentence was invisible before this.
        mesh_geometry: dict[str, dict] = {}

        def ground_y_at(world_x: float, world_z: float, yaw_degrees: float) -> float | None:
            local_x, local_z = terrain_local_xz(
                world_x, world_z, yaw_degrees, camera_x=camera_xz[0], camera_z=camera_xz[1]
            )
            hits = [mesh_y_at(local_x, local_z, m) for m in ground_meshes]
            hits = [y for y in hits if y is not None]
            if hits:
                snap_counts["mesh"] += 1
                return max(hits)
            if height_map is not None:
                sampled = height_map_y_at(local_x, local_z, height_map, height_map_grid_m)
                if sampled is not None:
                    snap_counts["height_map"] += 1
                    return sampled
            snap_counts["unsnapped"] += 1
            return None

        scene = Scene()
        scene.extrinsics = extrinsics
        # camera_height = world Y of the camera = terrain_y_at_nadir + camera_height_meters.
        # Using the extrinsics translation directly is equivalent and simpler.
        scene.camera_height = float(extrinsics.translation[1]) if extrinsics is not None else 0.0

        scene.eye_height_meters = self.config.eye_height_meters
        if terrain_mesh is not None:
            terrain_center_y = mesh_y_at(0.0, 0.0, terrain_mesh)
            scene.terrain_center_y = float(terrain_center_y) if terrain_center_y is not None else 0.0

        # Rotate the skybox to align the panorama center with the camera's forward direction.
        # The panorama center (theta=0) = +Z in camera space; the extrinsics rotation tells
        # us where +Z ended up in world space.  The skybox _Rotation (Y-axis, degrees) must
        # match that yaw so terrain and panorama line up.
        if extrinsics is not None:
            cam_forward = extrinsics.rotation @ np.array([0.0, 0.0, 1.0])
            scene.skybox_rotation = float(np.degrees(np.arctan2(cam_forward[0], cam_forward[2])))

        # Both of these are load-bearing for the terrain snap below (see
        # terrain_local_xz) and neither is otherwise visible in a run's logs, so a
        # scene whose objects don't sit on the ground can't currently be told apart
        # from one where the camera pose was never the problem. One line, once.
        self.log_info(
            f"Camera pose: yaw {scene.skybox_rotation:.2f}°, "
            f"translation ({camera_xz[0]:.3f}, {scene.camera_height:.3f}, {camera_xz[1]:.3f}) m"
            + ("" if height_map is not None else " — no height map available as a snap fallback")
        )

        if depth is not None:
            valid = depth.depth[np.isfinite(depth.depth) & (depth.depth > 0)]
            if len(valid) > 0:
                scene.far_clip_plane = float(np.percentile(valid, 99)) * 1.5
                scene.far_clip_plane = max(10.0, min(scene.far_clip_plane, 1000.0))

        if panorama is not None:
            scene.skybox = panorama_key

        if object_count is not None:
            # Curated per-(class, bucket) billboard pools from Panorama Asset
            # Generation -- top-K crops by quality score, usable at any
            # distance, not just "every crop in the class" (which included
            # poorly-scored/occluded crops with no ranking).
            billboard_pools: dict[str, list[int]] = context.input_object("billboard_pools") or {}
            # Indices Panorama Asset Generation rejected as background rather than
            # objects. Absent on a context written before that key existed, in which
            # case nothing is skipped and this behaves exactly as it used to.
            background_rejected: set[int] = set(context.input_object("background_rejected") or [])
            background_skipped = 0

            # Which visual variants each class actually has, taken from the pool
            # keys Panorama Asset Generation just wrote ("{class}::{bucket}").
            # Used only to give an unbucketed instance a plausible variant to
            # render -- see the `bucket is None` branch below.
            class_variants: dict[str, list[int]] = {}
            for pool_key in billboard_pools:
                pool_class, _, pool_bucket = pool_key.rpartition("::")
                if pool_class and pool_bucket.isdigit():
                    class_variants.setdefault(pool_class, []).append(int(pool_bucket))
            for variants in class_variants.values():
                variants.sort()

            camera_position = (
                np.array(extrinsics.translation, dtype=float) if extrinsics is not None else np.zeros(3)
            )

            rng = np.random.default_rng(self.seed)

            def unproject_box(box):
                """Bbox -> (world position, width, height), in whichever space this run has."""
                if panorama_depth is not None and panorama is not None:
                    return unproject_bbox_equirect(
                        box, panorama.width, panorama.height,
                        pano_depth=panorama_depth, extrinsics=extrinsics,
                    )
                return unproject_bbox(
                    box, input.width, input.height,
                    depth_map=depth, intrinsics=intrinsics, extrinsics=extrinsics,
                )

            # Learn the scene's size compression before placing anything, so that
            # every object -- including the very first -- is corrected against the
            # whole scene's evidence rather than whatever happened to precede it.
            #
            # This re-unprojects the anchor-eligible subset rather than reusing the
            # placement loop's own results, because the fit has to be complete before
            # that loop starts. Only classes in OBJECT_HEIGHT_PRIORS qualify, so this
            # is a small fraction of a typical detection set.
            scale_model = None
            if self.config.object_scale_correction:
                anchors = []
                for idx in range(object_count):
                    metadata = context.input_object(f"metadata_{idx}") or {}
                    box = metadata.get("box")
                    if not box:
                        continue
                    result = unproject_box(box)
                    if result is None:
                        continue
                    anchor_position, _, anchor_height = result
                    anchor_depth = float(np.linalg.norm(
                        np.array(anchor_position, dtype=float) - camera_position
                    ))
                    anchor = collect_anchor(
                        idx, metadata.get("class"), metadata, anchor_depth, anchor_height
                    )
                    if anchor is not None:
                        anchors.append(anchor)

                scale_model = fit_object_scale(
                    anchors,
                    num_bins=self.config.object_scale_num_bins,
                    min_anchors=self.config.object_scale_min_anchors,
                    max_correction=self.config.object_scale_max_correction,
                )

                if scale_model is None:
                    self.log_info(
                        f"Object scale correction: {len(anchors)} usable anchors "
                        f"(need {self.config.object_scale_min_anchors}) — leaving object sizes uncorrected"
                    )
                else:
                    by_class: dict[str, int] = {}
                    for a in anchors:
                        by_class[a.cls] = by_class.get(a.cls, 0) + 1
                    self.log_info(
                        f"Object scale correction fitted: {scale_model.describe()}; "
                        f"anchor classes: {dict(sorted(by_class.items()))}"
                    )
                    if self.temp is not None:
                        import json
                        debug = {
                            "model": scale_model.to_dict(),
                            "anchors": [
                                {
                                    "index": a.index, "class": a.cls,
                                    "depth_m": round(a.depth, 2),
                                    "measured_height_m": round(a.measured_height, 3),
                                    "prior_height_m": a.prior_height,
                                    "ratio": round(a.ratio, 3),
                                    "weight": round(a.weight, 3),
                                }
                                for a in sorted(anchors, key=lambda a: a.depth)
                            ],
                        }
                        (self.temp / "object_scale_debug.json").write_text(json.dumps(debug, indent=2))

            # Occupancy of the ground, as (x, z, radius, y_low, y_high) per placed
            # object, and the order they get to claim it in. Whoever is placed first
            # keeps its spot, so the order decides who survives a collision:
            # real detections before painted fill (a measured object is evidence, a
            # synthesized one is decoration), and within each, higher detector score
            # first. That ordering is also what makes this robust against two
            # detections of the SAME subject -- the weaker one now loses its ground
            # instead of being placed inside the stronger one.
            # Parallel arrays rather than a list of tuples so each test is one
            # vectorised pass instead of a Python loop over everything placed so far.
            # The test is inherently O(n^2), and while ground cover is exempt (the
            # only population in the millions), a painted non-grass distribution is
            # still capped at DistributionSynthesisStage's max_instances_per_group
            # (12,000) -- enough that an interpreted inner loop would dominate the
            # whole stage.
            occ_x = np.zeros(object_count, dtype=np.float64)
            occ_z = np.zeros(object_count, dtype=np.float64)
            occ_r = np.zeros(object_count, dtype=np.float64)
            occ_y_low = np.zeros(object_count, dtype=np.float64)
            occ_y_high = np.zeros(object_count, dtype=np.float64)
            occ_n = 0
            rejected_overlap = 0
            rejected_overlap_by_class: dict[str, int] = {}
            rejected_examples: list[str] = []

            def _placement_rank(i: int) -> tuple[int, float, int]:
                m = context.input_object(f"metadata_{i}") or {}
                return (1 if m.get("synthetic") else 0, -float(m.get("score") or 0.0), i)

            placement_order = sorted(range(object_count), key=_placement_rank)

            generation_task = self.create_progress(object_count, "Creating Objects…")
            for idx in placement_order:
                metadata = context.input_object(f"metadata_{idx}") or {}

                # Normalized for the same reason Panorama Asset Generation normalizes
                # it (see normalize_category): post-typing detections carry free-text
                # detector labels the environment test cannot recognize. It has to
                # happen in BOTH stages or not at all -- this is also the name that
                # builds mesh_key/pool_key below, and those have to match the keys
                # asset generation published under.
                cls = normalize_category(metadata.get("class"))
                if cls in _ENV_CATEGORIES or cls == "indeterminate":
                    self.log_info(f"Skipping {cls} object {idx}")
                    self.advance_progress(generation_task)
                    continue

                if metadata.get("position_only"):
                    # ObjectCategoryClusteringStage visually corroborated this
                    # low-confidence crop against class `cls` but it's still not
                    # trustworthy enough to render from its own crop -- only its
                    # position feeds ObjectDistributionStage's spatial pattern for
                    # `cls`. Never placed, meshed, or video-tracked.
                    self.advance_progress(generation_task)
                    continue

                if idx in background_rejected:
                    # Panorama Asset Generation measured this box to be mostly sky
                    # and/or mostly unmeasured depth and declined to build anything
                    # for it (max_background_fraction). That is a claim the detection
                    # is not an object at all, not merely that it isn't worth an
                    # asset, so placing it anyway put a fabricated metre size at a
                    # fabricated distance -- the three worst objects in the scene on
                    # the Rainier capture were all this. Its size comes from
                    # angular_extent x depth and its depth is the far clamp, so both
                    # numbers are invented and no amount of downstream gating can
                    # recover a real one.
                    background_skipped += 1
                    self.advance_progress(generation_task)
                    continue

                if metadata.get("synthetic"):
                    # Points painted across a region by DistributionSynthesisStage carry
                    # their world XZ + footprint directly — there's no detection bbox to
                    # unproject. Y is a placeholder; it gets replaced by the terrain snap
                    # below, same as every other object.
                    syn_x, _, syn_z = metadata["world_position"]
                    position = (syn_x, 0.0, syn_z)
                    width = metadata["world_width"]
                    height = metadata["world_height"]
                else:
                    # A missing box here means metadata_{idx} is an incomplete/stale cache
                    # entry (e.g. a resumed run whose OBJECT_COUNT outran what actually got
                    # written) rather than a real detection or synthesized point -- skip it
                    # the same way a failed unprojection below is skipped, instead of
                    # crashing on a KeyError three stages downstream of the real gap.
                    box = metadata.get("box")
                    if not box:
                        self.log_warning(f"No box for object {idx} (incomplete metadata), skipping")
                        self.advance_progress(generation_task)
                        continue

                    result = unproject_box(box)
                    if result is None:
                        self.log_warning(f"Could not unproject bbox for object {idx}, skipping")
                        self.advance_progress(generation_task)
                        continue
                    position, width, height = result

                max_dim = max(width, height)
                if max_dim > _MAX_OBJECT_SIZE_M:
                    self.log_info(f"Skipping object {idx} ({cls}): estimated size {max_dim:.1f}m exceeds limit")
                    self.advance_progress(generation_task)
                    continue

                # Author this class's height in metres instead of measuring it -- the
                # same call GrassCoverStage._tuft_height already makes for tufts, for
                # the same reason, and applied here for the classes where the detector
                # boxes a PART of the thing that has to be placed.
                #
                # `flower` is the case this exists for. GroundingDINO boxes the bloom,
                # not the plant: on the Rainier meadow that is 0.04-0.17 m (median
                # 0.10), and the measurement is not wrong -- a paintbrush bloom really
                # is that big, and at 1.7-2.5 m it sits in the depth map's identity
                # region where there is no compression to blame. But the thing that
                # belongs in the scene is a plant standing in a meadow, and 33 of 33
                # flowers came out shorter than the SHORTEST grass tuft around them
                # (0.25 m, median 0.35 m), so every one of them was placed correctly
                # and buried completely.
                #
                # Deliberately after the size cull, like the scale correction below:
                # the cull judges whether the raw measurement describes a discrete
                # object at all, and that judgement belongs on what was measured.
                #
                # Width scales with height so the crop keeps its own aspect ratio --
                # the billboard and the category mesh are both pictures of the bloom,
                # and stretching one axis alone would shear them.
                metric_height = self.config.metric_height_m.get(cls) if cls else None
                if metric_height and height > 1e-6:
                    jitter = 1.0 + float(rng.uniform(
                        -self.config.metric_height_jitter, self.config.metric_height_jitter
                    ))
                    target = float(metric_height) * jitter
                    width *= target / height
                    height = target
                    # Authored in metres, so it must never also be multiplied by a
                    # fitted depth->size correction. Same exemption grass already has.
                    metadata = {**(metadata or {}), "metric_size": True}

                # Undo the scene's measured size compression. Deliberately AFTER the
                # cull above, not before it: the cull's job is to reject things that
                # were never discrete objects (a mountainside, a hillside, sky), and
                # that judgement belongs on the raw measurement. Applying it to
                # corrected sizes would start culling exactly the large-but-legitimate
                # objects this correction exists to restore, and would make the set of
                # placed objects depend on the fit -- so the same capture would gain
                # and lose objects as the anchors changed.
                #
                # Size only. Position is untouched, so the object still stands where
                # the depth map put it and still snaps to the same terrain point --
                # terrain and both depth maps are left exactly as they were.
                if scale_model is not None and not is_metric_authored(cls, metadata):
                    object_depth = float(np.linalg.norm(
                        np.array(position, dtype=float) - camera_position
                    ))
                    scale_factor = scale_model.factor_at(object_depth)
                    if abs(scale_factor - 1.0) > 1e-3:
                        width *= scale_factor
                        height *= scale_factor

                # Snap the object's base to the terrain surface. The Y offset that
                # lands the *bottom* of the object on the terrain depends on the
                # object's actual placed vertical extent, which differs between the
                # billboard (scaled exactly to `height`) and category-mesh (scaled
                # uniformly to `mesh_scale`) branches below — so it's resolved per
                # branch rather than once here.
                #
                # position[0]/position[2] are in WORLD space (extrinsics.rotation is
                # already baked in by unproject_bbox/unproject_bbox_equirect above),
                # but the ground meshes' own vertices are stored in their native,
                # unrotated frame (+Z = panorama theta 0) — the same yaw compensation
                # applied to the terrain Object3D's own rotation (see
                # scene.skybox_rotation above) has to be undone before raycasting, or
                # the query lands at the wrong point whenever that yaw isn't ~0: near
                # the finite terrain grid's edges this misses the mesh outright
                # (silently falling back to the object's raw unprojected Y below — the
                # floating/sinking objects this was reported as), and even where it
                # still hits the mesh, it samples the wrong patch of terrain relief.
                # ground_y_at handles that, and takes the highest of the base terrain
                # and every formation standing on it.
                terrain_y = ground_y_at(position[0], position[2], scene.skybox_rotation)

                if terrain_y is None and len(unsnapped_examples) < 5:
                    unsnapped_examples.append(
                        f"{idx} ({cls}) at x={position[0]:.1f} z={position[2]:.1f}, "
                        f"left at its unprojected y={position[1]:.2f}"
                    )

                place_y = terrain_y + height / 2.0 if terrain_y is not None else position[1]

                # Rejection sampling against the ground already claimed. Done here,
                # after the terrain snap has resolved the object's final footprint and
                # height, and before anything is added to the scene.
                #
                # Footprints are circles of radius width/2 in world XZ, tested only
                # between objects whose vertical extents actually overlap -- two things
                # at the same (x, z) but different heights (a lamp head over a bench)
                # are not intersecting, and rejecting one of them would be wrong.
                # Exemption is by class alone, deliberately NOT via is_metric_authored:
                # that answers "was this size authored in metres" (so the scale
                # correction must skip it), which is a different question from "is this
                # ground cover that is supposed to interpenetrate".
                if self.config.overlap_rejection and cls not in self.config.overlap_exempt_classes:
                    radius = max(float(width) / 2.0, 1e-3)
                    y_low, y_high = place_y - height / 2.0, place_y + height / 2.0
                    # Asymmetric by construction: the threshold is the ARRIVING
                    # object's, tested against ground already claimed. That makes
                    # the result depend on placement_order, which is exactly the
                    # precedence wanted here -- _placement_rank puts real detections
                    # ahead of synthetic ones and orders each by descending score, so
                    # the painted surplus yields to real evidence and a weak
                    # detection yields to a strong one, rather than whichever
                    # happened to be enumerated first.
                    min_separation = self.config.overlap_min_separation_overrides.get(
                        cls, self.config.overlap_min_separation
                    )
                    blocker = None
                    if occ_n:
                        reach = np.maximum(occ_r[:occ_n] + radius, 1e-9)
                        separation = np.hypot(
                            occ_x[:occ_n] - position[0], occ_z[:occ_n] - position[2]
                        ) / reach
                        # Only against objects this one actually shares height with.
                        overlaps_vertically = (y_high > occ_y_low[:occ_n]) & (y_low < occ_y_high[:occ_n])
                        conflicting = overlaps_vertically & (separation < min_separation)
                        if conflicting.any():
                            blocker = float(separation[conflicting].min())
                    if blocker is not None:
                        rejected_overlap += 1
                        rejected_overlap_by_class[cls] = rejected_overlap_by_class.get(cls, 0) + 1
                        if len(rejected_examples) < 5:
                            rejected_examples.append(
                                f"{idx} ({cls}) at x={position[0]:.1f} z={position[2]:.1f}, "
                                f"separation {blocker:.2f} < {min_separation:.2f}"
                            )
                        self.advance_progress(generation_task)
                        continue
                    occ_x[occ_n], occ_z[occ_n] = position[0], position[2]
                    occ_r[occ_n] = radius
                    occ_y_low[occ_n], occ_y_high[occ_n] = y_low, y_high
                    occ_n += 1

                context.add_object(f"metadata_{idx}", {
                    **(metadata or {}),
                    "world_position": list(map(float, (position[0], place_y, position[2]))),
                    "world_width": float(width),
                    "world_height": float(height),
                })

                # An unbucketed instance is one ObjectCategoryClusteringStage never
                # reached -- it had no crop to embed, or it was created after that
                # stage ran (Object Detection / Instance Refinement splits). It is
                # NOT a member of variant 0.
                #
                # Reading it as `metadata.get("bucket") or 0` made it one anyway,
                # and that silently collapsed the whole population onto a single
                # asset: on the Rainier capture 11 of 17 placed flowers had no
                # bucket, so every one of them rendered category_mesh_flower_0
                # while the other 22 generated flower meshes went unused. Spread
                # them instead, deterministically per index, across the variants
                # this class actually has -- an arbitrary-but-stable variant is a
                # far better guess than "always the first one".
                bucket = metadata.get("bucket")
                if bucket is None:
                    variants = class_variants.get(cls)
                    bucket = (
                        variants[np.random.default_rng((self.seed, idx)).integers(len(variants))]
                        if variants else 0
                    )
                bucket = int(bucket)
                mesh_key = f"category_mesh_{cls}_{bucket}"
                pool_key = f"{cls}::{bucket}"
                category_mesh = context.input_mesh(mesh_key)

                # Optional far-LOD mesh for classes a camera-facing billboard
                # can't represent. Ground cover is the case this exists for: a
                # billboard is a single quad that turns to face the viewer, which
                # works for an upright subject seen from eye level and collapses
                # to a line -- then swings through the ground plane -- for grass
                # underfoot. GrassCoverStage builds a fixed-orientation crossed-
                # card mesh under this key instead (see grass_cover/cards.py).
                #
                # Kept generic rather than special-cased on the class: any group
                # that publishes a _card mesh gets the same treatment, and any
                # group that doesn't falls through to the billboard pool exactly
                # as before.
                card_mesh_key = f"{mesh_key}_card"
                card_mesh = context.input_mesh(card_mesh_key)

                # Bake-time projected-size LOD: mesh if this instance renders big
                # enough to resolve AND its bucket actually has one (a bucket only
                # gets a mesh if some instance of it qualified during Panorama
                # Asset Generation -- this instance itself may still be too small
                # even when a sibling instance in the same bucket was prominent
                # enough to trigger meshing).
                #
                # `height` here is the placed, scale-corrected extent, i.e. what
                # the viewer actually sees, not the raw detected angle Panorama
                # Asset Generation gates on. That is deliberate: a mesh is worth
                # spending on what looks big, and object_scale.py's correction is
                # part of how big it looks. It also means the two stages' pixel
                # thresholds are not interchangeable -- see min_mesh_angular_px.
                camera_distance = float(np.linalg.norm(
                    np.array((position[0], place_y, position[2]), dtype=float) - camera_position
                ))
                rendered_px = angular_height_px(
                    height, camera_distance, self.config.viewer_px_per_degree,
                    viewer_move_radius_m=self.config.viewer_move_radius_m,
                )
                use_mesh = (
                    category_mesh is not None
                    and rendered_px >= self.config.min_mesh_angular_px
                )
                using_card = False
                if not use_mesh and card_mesh is not None:
                    # Too small to resolve as geometry (or this bucket never got a
                    # reconstructed mesh at all) but a card LOD exists -- take it
                    # rather than the billboard. Everything below is shared: the
                    # card is an ordinary mesh, so it gets the same base snap,
                    # random yaw, and CategoryMeshRiggingStage will rig it for sway
                    # just like the near-LOD asset.
                    category_mesh, mesh_key, use_mesh = card_mesh, card_mesh_key, True
                    using_card = True

                # A card is authored here, not reconstructed: crossed_card_mesh builds
                # it to a known shape from a crop that is already the instance's own
                # cutout, and it is scaled to the detection's own width and height
                # below. The gates exist to catch a RECONSTRUCTION that turned out to
                # describe something other than its detection, which is a question a
                # card cannot fail -- so it is not asked.
                if use_mesh and not using_card:
                    # Does this mesh actually describe this instance's detection, and
                    # what does it render at if so? Both questions are answered before
                    # anything is placed, because a mesh that fails either one has to
                    # fall through to the billboard branch below -- a billboard is
                    # scaled directly to (width, height) and so cannot blow up the way
                    # an unbounded 1/extent_y can.
                    mesh_extents = category_mesh.mesh.bounds[1] - category_mesh.mesh.bounds[0]
                    mesh_scale, mesh_rejection, mesh_gate = mesh_instance_scale(
                        mesh_extents, width, height,
                        min_height_fraction=self.config.min_mesh_height_fraction,
                    )

                    record = _mesh_geometry_record(
                        mesh_geometry, mesh_key, cls, mesh_extents, idx, mesh_scale
                    )

                    if mesh_rejection is not None:
                        # A rejected reconstruction prefers the card LOD over a
                        # billboard when its group has one. Both are pictures of the
                        # subject, but a card is fixed-orientation crossed planes with
                        # parallax between them, while a billboard is one quad that
                        # turns to face the viewer -- and for the subjects that fail
                        # these gates (trees above all) that swing is the artefact
                        # being traded for. This ordering only exists because the card
                        # substitution above happens BEFORE the gates: without it,
                        # publishing a tree card would change nothing, since the
                        # reconstruction is found first and its rejection went straight
                        # to the billboard branch.
                        fallback = "carding" if card_mesh is not None else "billboarding"
                        self.log_warning(
                            f"Category mesh {mesh_key} rejected for object {idx} ({cls}): "
                            f"{mesh_rejection}; {fallback} instead"
                        )
                        mesh_rejections[mesh_key] = mesh_rejections.get(mesh_key, 0) + 1
                        record["rejected"] += 1
                        record.setdefault("reason", mesh_rejection)
                        # Which gate fired, tallied per mesh. The prose reason above
                        # only survives for the FIRST rejected instance of a bucket
                        # (setdefault), but every instance is judged separately and a
                        # bucket can fail different gates for different detections --
                        # aspect is per-instance, height and sheet are not. Counting
                        # them is what makes the threshold sweep reproducible from a
                        # debug bundle instead of re-derived from GLB extents.
                        if mesh_gate is not None:
                            gates = record.setdefault("gates", {})
                            gates[mesh_gate] = gates.get(mesh_gate, 0) + 1
                        if card_mesh is not None:
                            category_mesh, mesh_key = card_mesh, card_mesh_key
                            using_card = True
                        else:
                            use_mesh = False

                if use_mesh:
                    # Use the shared bucket mesh with a random Y rotation.
                    # Mesh.fit_to_box recenters on the mesh's centroid, not its bounding-box
                    # center, so the mesh's lowest vertex is not reliably at -extent/2 --
                    # for a bottom-heavy shape (e.g. a tree trunk) the centroid sits below
                    # the bbox center, and assuming symmetry would still leave a gap above
                    # the terrain. Sample the mesh's actual lowest vertex (bounds[0][1])
                    # and offset by exactly that, so the true bottom -- not an assumed one
                    # -- lands on terrain_y.
                    #
                    # mesh_scale was resolved above by mesh_instance_scale, which also
                    # decides whether this mesh may be used for this instance at all;
                    # reaching here means it passed. See that function for why height,
                    # and not max(width, height), sets the scale.
                    #
                    # A card is the exception, and is scaled per axis to the detection's
                    # own width and height. Uniform scaling exists because a
                    # reconstruction's proportions are its own evidence and must not be
                    # distorted; a card has no proportions of its own to preserve -- it
                    # is a picture on planes, and a conifer boxed at 1:8 rendered
                    # uniformly would stand as wide as it is tall. Horizontal scale is
                    # shared between X and Z so the yaw fan stays isotropic, and the
                    # card's base sits at y=0 by construction (see cards.py), so the
                    # snap below resolves to 0 either way. Grass is unaffected: its
                    # synthetic instances carry world_width == world_height.
                    if using_card:
                        card_extents = category_mesh.mesh.bounds[1] - category_mesh.mesh.bounds[0]
                        horizontal = float(card_extents[0]) or 1.0
                        vertical = float(card_extents[1]) or 1.0
                        scale_x = scale_z = float(width) / horizontal
                        scale_y = float(height) / vertical
                        _mesh_geometry_record(
                            mesh_geometry, mesh_key, cls, card_extents, idx, scale_y
                        )
                    else:
                        scale_x = scale_y = scale_z = mesh_scale
                    mesh_min_y = float(category_mesh.mesh.bounds[0][1]) * scale_y
                    mesh_place_y = terrain_y - mesh_min_y if terrain_y is not None else position[1]
                    self.log_info(
                        f"Creating {'card' if using_card else 'mesh'} for {idx} "
                        f"({cls}, bucket {bucket}, {camera_distance:.1f}m, "
                        f"{rendered_px:.0f}px)"
                    )
                    mesh_obj = Object3D.mesh(mesh_key, x=position[0], y=mesh_place_y, z=position[2])
                    mesh_obj.set_rotation(0.0, float(rng.uniform(0.0, 360.0)), 0.0)
                    mesh_obj.set_scale(scale_x, scale_y, scale_z)
                    mesh_obj.name = mesh_key
                    mesh_obj.source_index = idx
                    scene.add_object(mesh_obj)
                else:
                    # Pick a random billboard crop from this bucket's curated pool.
                    # Synthetic points have no crop of their own, so they must draw
                    # from a real detection's pool; if none exists there's nothing
                    # to render.
                    crop_pool = billboard_pools.get(pool_key) or ([] if metadata.get("synthetic") else [idx])
                    if not crop_pool and metadata.get("synthetic"):
                        # This painted instance inherited its bucket from a real
                        # exemplar (DistributionSynthesisStage samples size and bucket
                        # together), but not every bucket ends up with a curated pool --
                        # PanoramaAssetGenerationStage disqualifies crops on quality and
                        # occlusion, so a thinly-populated variant can lose all of its
                        # candidates. Falling back to any pool for the same class keeps
                        # the instance in the scene as a sibling variant instead of
                        # silently deleting it, which is the better failure for a
                        # population whose whole purpose is filling the environment.
                        fallback = [
                            i
                            for key, pool in billboard_pools.items()
                            if key.split("::", 1)[0] == cls
                            for i in pool
                        ]
                        if fallback:
                            self.log_info(
                                f"No pool for {pool_key}; synthetic object {idx} "
                                f"falling back to another {cls} variant"
                            )
                            crop_pool = fallback
                    if not crop_pool:
                        self.log_warning(f"No billboard crop available for synthetic object {idx} ({cls}), skipping")
                        self.advance_progress(generation_task)
                        continue
                    chosen_idx = int(rng.choice(crop_pool))
                    # Prefer the delit crop PanoramaAssetGenerationStage wrote, so
                    # this billboard carries albedo like the terrain texture beside
                    # it rather than a photograph with the sun already in it (the
                    # object-vs-ground colour mismatch -- see that stage's
                    # use_intrinsic_delighting). The client resolves Object3D.texture
                    # by name, so pointing at the other asset is the whole change.
                    # Falls back per-crop, so one failed delight costs one billboard.
                    texture_key = f"crop_delit_{chosen_idx}"
                    if context.input_image(texture_key) is None:
                        texture_key = f"crop_{chosen_idx}"
                    self.log_info(f"Creating billboard for {idx} ({cls}, bucket {bucket}, {camera_distance:.1f}m) using {texture_key}")
                    billboard = Object3D.billboard(
                        texture_key,
                        width=width,
                        height=height,
                        x=position[0],
                        y=place_y,
                        z=position[2],
                    )
                    billboard.name = f"billboard_{idx}"
                    billboard.source_index = idx
                    # Record whose crop is actually on screen -- see
                    # Object3D.texture_source_index. SceneAnimationStage keys the
                    # object_video_{i} lookup on this, so a pooled or synthetic
                    # billboard animates with the clip belonging to the crop it
                    # displays instead of a different instance's (or, for synthetic
                    # points, none at all).
                    billboard.texture_source_index = chosen_idx
                    scene.add_object(billboard)
                self.advance_progress(generation_task)

            self.finish_progress(generation_task)

            # Loud on purpose, same reasoning as DistributionSynthesisStage's failure
            # report: an object that never found the ground renders as one floating in
            # the air or buried in the hillside, and until now that was completely
            # silent -- indistinguishable from an object the terrain genuinely placed
            # there. A run where these counts are anything other than "almost all mesh"
            # is a run whose placement should not be trusted.
            snapped = snap_counts["mesh"] + snap_counts["height_map"]
            if snap_counts["height_map"] or snap_counts["unsnapped"]:
                level = (
                    self.log_error
                    if snap_counts["unsnapped"] or snap_counts["mesh"] == 0
                    else self.log_warning
                )
                level(
                    f"Terrain snap: {snap_counts['mesh']} object(s) hit a ground mesh, "
                    f"{snap_counts['height_map']} fell back to the height map, "
                    f"{snap_counts['unsnapped']} found no ground at all"
                )
                for example in unsnapped_examples:
                    level(f"    no ground under object {example}")
            elif snapped:
                self.log_info(f"Terrain snap: all {snapped} placed object(s) hit a ground mesh")

            if background_skipped:
                self.log_info(
                    f"Background rejection: skipped {background_skipped} detection(s) "
                    f"Panorama Asset Generation measured as sky/unmeasured-depth rather "
                    f"than objects"
                )

            if rejected_overlap:
                self.log_info(
                    f"Overlap rejection: dropped {rejected_overlap} object(s) landing inside "
                    f"one already placed (min separation "
                    f"{self.config.overlap_min_separation:.2f}, {occ_n} kept)"
                )
                # Broken out by class because the threshold now varies by class, so
                # one total cannot say whether a population is being thinned as
                # intended or a discrete class is being culled by accident.
                for key, count in sorted(rejected_overlap_by_class.items(), key=lambda kv: -kv[1]):
                    used = self.config.overlap_min_separation_overrides.get(
                        key, self.config.overlap_min_separation
                    )
                    self.log_info(f"    {key}: {count} instance(s) at separation {used:.2f}")
                for example in rejected_examples:
                    self.log_info(f"    rejected object {example}")

            if mesh_rejections:
                self.log_warning(
                    f"Category mesh rejection: {sum(mesh_rejections.values())} instance(s) "
                    f"across {len(mesh_rejections)} mesh(es) billboarded instead of meshed"
                )
                for key, count in sorted(mesh_rejections.items(), key=lambda kv: -kv[1]):
                    self.log_warning(f"    {key}: {count} instance(s)")

            # Reporting must never be able to fail the stage. This exact call took
            # a completed 85-minute run down at the last stage with a NameError on
            # a debug-file write -- every object placed, the scene assembled, and
            # the run discarded by a log line. A diagnostic that can destroy the
            # thing it is diagnosing is worse than no diagnostic.
            try:
                self._log_mesh_geometry(mesh_geometry)
            except Exception as e:
                self.log_warning(f"Could not report mesh geometry ({type(e).__name__}: {e})")

        # Terrain/water/formation vertices were built directly in the panorama's
        # own frame (+Z = theta 0), matching the skybox's UNROTATED orientation --
        # not the camera's actual world orientation, which the extrinsics rotation
        # may yaw away from +Z by an arbitrary amount (see scene.skybox_rotation
        # above). Detected objects don't need this: unproject_bbox/
        # unproject_bbox_equirect already bake extrinsics.rotation into their
        # world_position. Terrain/water/formations get no such per-vertex
        # transform, so without applying the same yaw here they'd disagree with
        # the skybox (and every detected object) on which way is "forward"
        # whenever extrinsics.rotation isn't ~identity.
        ground_rotation = (0.0, scene.skybox_rotation, 0.0)

        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)
        if terrain_mesh is not None:
            self.log_info("Adding terrain mesh to scene")
            terrain = Object3D.mesh(ContextKey.TERRAIN_MESH, x=0.0, y=0.0, z=0.0)
            terrain.set_rotation(*ground_rotation)
            terrain.name = "terrain"
            # Coarse geometry-only collision proxy (TerrainMeshGenerator.
            # generate_physics_mesh) -- Unity attaches this, not the dense
            # textured render mesh above, as a MeshCollider.
            if context.input_mesh(ContextKey.TERRAIN_PHYSICS_MESH) is not None:
                terrain.physics_mesh = ContextKey.TERRAIN_PHYSICS_MESH
            scene.add_object(terrain)

        water_mesh = context.input_mesh(ContextKey.WATER_MESH)
        if water_mesh is not None:
            self.log_info("Adding water mesh to scene")
            water = Object3D.mesh(ContextKey.WATER_MESH, x=0.0, y=0.0, z=0.0)
            water.set_rotation(*ground_rotation)
            water.name = "water"
            scene.add_object(water)

        # Non-primary ground components (see TerrainMeshStage / HeightMapGenerator.
        # _label_ground_components) -- a separate landmass, an isolated rock
        # formation, anything real but genuinely disconnected from the base terrain.
        # Each already carries its own absolute-world-space vertices (built by
        # TerrainMeshGenerator.generate_component_mesh) and its own baked texture
        # (TerrainTextureGenerationStage._texture_formations), so it's placed at the
        # origin exactly like terrain/water above.
        #
        # Object3D.name deliberately does NOT contain "terrain" -- the Unity client's
        # TerrainMaterialManager matches meshes to apply the shared, separately-
        # transmitted SplatMaterial by a case-insensitive substring check against
        # "terrain" (confirmed this session), and a formation mesh already carries
        # its own baked-in texture; it must not be swept up by that name match.
        formations = context.input_object(ContextKey.TERRAIN_FORMATIONS) or []
        for formation in formations:
            formation_mesh = context.input_mesh(formation["mesh_key"])
            if formation_mesh is None:
                continue
            self.log_info(f"Adding formation mesh {formation['id']} to scene")
            formation_obj = Object3D.mesh(formation["mesh_key"], x=0.0, y=0.0, z=0.0)
            formation_obj.set_rotation(*ground_rotation)
            formation_obj.name = f"formation_{formation['id']}"
            # Own collision proxy (see TerrainMeshStage.run's physics_result) --
            # without it, the base terrain's own depression under this formation
            # would be an unfilled hole in the collision surface.
            physics_mesh_key = formation.get("physics_mesh_key")
            if physics_mesh_key and context.input_mesh(physics_mesh_key) is not None:
                formation_obj.physics_mesh = physics_mesh_key
            scene.add_object(formation_obj)

        lighting = context.input_lighting(ContextKey.LIGHTING)
        if lighting is not None:
            self.log_info("Adding environment lighting to scene")
            scene.lighting = lighting

        context.add_scene(output_key, scene)
        return context

    def _log_mesh_geometry(self, mesh_geometry: dict[str, dict]) -> None:
        """Report every category mesh's own shape and what it rendered at.

        The scale a shared bucket mesh gets is per-instance (height / extent_y), so
        the same reconstruction can be fine for one detection and catastrophic for
        the next -- and neither the mesh's shape nor the resulting world size was
        recorded anywhere before this. Without them a scene-spanning slab and a
        correctly-placed object produce identical logs.

        height_fraction and aspect are the two inputs mesh_instance_scale actually
        judges on, printed whether or not it rejected, so a mesh sitting just inside
        the thresholds is visible before it becomes a bug report.
        """
        if not mesh_geometry:
            return

        self.log_info("Category mesh geometry (normalised extents → rendered size):")
        for key, rec in sorted(mesh_geometry.items(), key=lambda kv: -(kv[1]["rendered_max_m"] or 0.0)):
            status = (
                f"ALL {rec['rejected']} REJECTED" if rec["rejected"] == rec["instances"]
                else f"{rec['rejected']}/{rec['instances']} rejected" if rec["rejected"]
                else f"{rec['instances']} placed"
            )
            aspect = "n/a" if rec["aspect"] is None else f"{rec['aspect']:.2f}"
            self.log_info(
                f"    {key:<34} extents {rec['extents']} "
                f"h_frac {rec['height_fraction']:.3f} thin {rec['thickness_fraction']:.3f} "
                f"aspect {aspect}  "
                f"scale {rec['scale_min']:.1f}-{rec['scale_max']:.1f}x  "
                f"rendered {rec['rendered_min_m']:.2f}-{rec['rendered_max_m']:.2f} m  [{status}]"
            )
            if rec.get("reason"):
                self.log_info(f"        reason: {rec['reason']}")
            if rec["examples"]:
                biggest = ", ".join(
                    f"obj {e['idx']} @ {e['scale']:.1f}x = {e['rendered_m']:.2f} m"
                    for e in rec["examples"][:3]
                )
                self.log_info(f"        largest: {biggest}")

        if self.temp is not None or self.output is not None:
            out = (self.temp or self.output) / "mesh_geometry_debug.json"
            try:
                out.write_text(json.dumps({
                    "thresholds": {
                        "min_mesh_height_fraction": self.config.min_mesh_height_fraction,
                        "min_mesh_thickness_fraction": _MIN_MESH_THICKNESS_FRACTION,
                        "max_mesh_aspect_ratio": _MAX_MESH_ASPECT_RATIO,
                        "max_object_size_m": _MAX_OBJECT_SIZE_M,
                        "min_mesh_angular_px": self.config.min_mesh_angular_px,
                        "viewer_px_per_degree": self.config.viewer_px_per_degree,
                        "viewer_move_radius_m": self.config.viewer_move_radius_m,
                    },
                    "meshes": mesh_geometry,
                }, indent=2))
            except Exception as e:
                self.log_warning(f"Could not write mesh_geometry_debug.json: {e}")

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, _, _, _, output_key = self._resolved_keys()

        return context.scene(output_key) is not None