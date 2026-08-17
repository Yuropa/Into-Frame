"""Projected-size metric shared by the two stages that decide mesh vs. billboard.

Both decisions used to be taken on DISTANCE from the camera, at two different
granularities: PanoramaAssetGenerationStage asked "is any instance of this group
within billboard_distance_m" before reconstructing a bespoke mesh, and
SceneGenerationStage asked "is this instance within mesh_lod_distance_m" before
placing that mesh rather than a billboard.

Distance is the wrong axis. What a viewer can actually resolve is the angle a
subject subtends, and distance only stands in for that if every subject is the
same size -- which is exactly where it broke down. The single global cutoff was
tuned for subjects you look AT (a tree at 25 m is worth its mesh) and was
catastrophic for ground cover at your feet at the same distance, so it grew a
per-class override table (mesh_lod_distance_overrides) whose only entry existed
to undo the mismatch for grass. Measured on the Rainier capture: 318 instances
held a full 8,000-triangle mesh while subtending under 250 px of screen height
-- 203 flowers, 43 trees, 39 plants -- roughly 2.5 M of the scene's 4.47 M
triangles spent below a thumbnail. Grass measures p50 44 px / p90 111 px on the
same metric and falls out of any sane threshold with no special case at all.

Angular HEIGHT, not solid angle or width. mesh_instance_scale() documents the
reason at length: an equirectangular box's horizontal extent depends on the
subject's yaw relative to the camera (a tree seen through a gap vs. broadside
differ by a lot) while its vertical extent does not. Width is the unreliable
axis, so it must not be what sets the threshold -- the same argument that makes
height, not max(width, height), set mesh scale.

Expressed in DISPLAY PIXELS rather than degrees or steradians so the threshold
is legible against the thing it is really about ("below ~250 px of screen height
a card is indistinguishable from a reconstruction") and so it moves correctly if
the target display changes -- see viewer_px_per_degree.
"""
from __future__ import annotations

import math


def angular_height_px(
    world_height_m: float,
    distance_m: float,
    px_per_degree: float,
    *,
    viewer_move_radius_m: float = 0.0,
    min_distance_m: float = 0.25,
) -> float:
    """Screen height, in display pixels, of `world_height_m` seen from `distance_m`.

    viewer_move_radius_m is what makes this safe to evaluate ONCE at bake time.
    The old distance cutoff could be baked statically because distance changes by
    at most the distance walked; angular size does not share that property, since
    it goes as 1/d. A 3 m subject 3 m away nearly doubles its angular size when
    the viewer takes one and a half steps toward it, so a threshold evaluated at
    the bake position would demote assets the viewer can then walk up to. Sizing
    at the CLOSEST the viewer can get instead -- distance minus the movement
    radius -- builds that hysteresis into the metric rather than leaving it as an
    unstated assumption about how little the player moves.

    min_distance_m floors the divisor so an instance at or inside the movement
    radius is finite rather than infinite; anything that close is far above any
    useful threshold either way, and the floor only decides how far above.
    """
    if world_height_m <= 0.0 or px_per_degree <= 0.0:
        return 0.0
    reachable = max(float(distance_m) - float(viewer_move_radius_m), float(min_distance_m))
    return math.degrees(2.0 * math.atan(float(world_height_m) / (2.0 * reachable))) * float(px_per_degree)


def angular_height_px_from_box(
    box_height_px: float,
    panorama_height_px: float,
    px_per_degree: float,
) -> float:
    """Same quantity, read straight off an equirectangular detection box.

    An equirect panorama IS an angular map -- one row is 180/H degrees of
    elevation regardless of depth -- so a detection's angular height needs no
    depth sample at all. This is the form PanoramaAssetGenerationStage wants:
    it runs before anything has been placed in world space, and its depth
    samples are the least trustworthy input it has.

    NOTE this is the RAW detected angle. SceneGenerationStage's world sizes
    carry object_scale.py's correction on top (measured 4.45x on the Rainier
    capture), so the same subject scores ~4x higher there. The two thresholds
    are therefore NOT interchangeable -- see the config comments on each.
    """
    if box_height_px <= 0.0 or panorama_height_px <= 0.0 or px_per_degree <= 0.0:
        return 0.0
    return (float(box_height_px) / float(panorama_height_px)) * 180.0 * float(px_per_degree)
