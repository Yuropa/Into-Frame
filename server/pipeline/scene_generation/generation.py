from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import ENVIRONMENT_CATEGORIES as _ENV_CATEGORIES
from pipeline.scene_generation.projection import mesh_y_at, terrain_local_xz, unproject_bbox, unproject_bbox_equirect
from pipeline.scene_generation.object_scale import collect_anchor, fit_object_scale, is_metric_authored
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
        mesh_lod_distance_m: float = 30.0,
        mesh_lod_distance_overrides: dict[str, float] | None = None,
        object_scale_correction: bool = True,
        object_scale_max_correction: float = 8.0,
        object_scale_min_anchors: int = 6,
        object_scale_num_bins: int = 6,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Sent to the client as the target world-space depth of the terrain
        # center below the viewer; the client pushes the whole scene down to match.
        self.eye_height_meters = eye_height_meters
        # Bake-time mesh-vs-billboard cutoff: an instance closer than this to
        # the camera uses its bucket's 3D mesh, farther instances use a
        # billboard. Deliberately mirrors Panorama Asset Generation's
        # billboard_distance_m (meshes only exist for groups with an instance
        # within that same distance) -- each stage parses its own isolated
        # config, so keep the two in sync manually if either changes. A static
        # bake-time cutoff (vs. runtime client-side LOD) is fine here since
        # expected player movement is small (~2 m).
        self.mesh_lod_distance_m = mesh_lod_distance_m
        # Per-class override of that cutoff, {class_name: metres}. The single global
        # value is tuned for subjects you look AT -- a tree at 25 m is worth its mesh.
        # It is badly wrong for ground cover at your feet, where the same distance
        # covers thousands of instances.
        #
        # Measured on the Rainier capture: GrassCoverStage places 6,661 tufts, all of
        # them inside its own max_radius_m of 25 m, so with a global 30 m cutoff every
        # single one took the reconstructed near mesh -- ~275,000 triangles each, 1.83
        # BILLION in total -- and the 12-triangle crossed-card LOD that exists for
        # exactly this purpose never rendered once.
        self.mesh_lod_distance_overrides = dict(mesh_lod_distance_overrides or {})
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

# Objects whose estimated real-world largest dimension exceeds this threshold (meters)
# are assumed to be large scene elements (mountains, hills, sky) and are skipped.
_MAX_OBJECT_SIZE_M = 40.0

# A category mesh is scaled to match its instance's detected HEIGHT (see the mesh
# branch in run()), which needs its own vertical extent as the divisor. Below this
# fraction of its largest extent the mesh is effectively a flat sheet, that divisor
# is noise, and matching height would blow the other axes up by 1/extent_y -- so the
# instance falls back to footprint scaling instead.
_MIN_MESH_HEIGHT_FRACTION = 0.05


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
                                                       instance is within
                                                       mesh_lod_distance_m of the
                                                       camera, billboard otherwise)
      "billboard_pools" (generic object)  → {"{class}::{bucket}": [crop idx, ...]}
                                             (curated top-K pool per bucket, from
                                             Panorama Asset Generation)

    Optional:
      ContextKey.TERRAIN_MESH        → Mesh        (placed at origin if present)
      ContextKey.WATER_MESH          → Mesh        (placed at origin if present)
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
        ground_meshes = [terrain_mesh]
        for formation in context.input_object(ContextKey.TERRAIN_FORMATIONS) or []:
            ground_meshes.append(context.input_mesh(formation["mesh_key"]))
        ground_meshes = [m for m in ground_meshes if m is not None]

        def ground_y_at(world_x: float, world_z: float, yaw_degrees: float) -> float | None:
            local_x, local_z = terrain_local_xz(world_x, world_z, yaw_degrees)
            hits = [mesh_y_at(local_x, local_z, m) for m in ground_meshes]
            hits = [y for y in hits if y is not None]
            return max(hits) if hits else None

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

            generation_task = self.create_progress(object_count, "Creating Objects…")
            for idx in range(object_count):
                metadata = context.input_object(f"metadata_{idx}") or {}

                cls = metadata.get("class")
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

                place_y = terrain_y + height / 2.0 if terrain_y is not None else position[1]

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

                # Bake-time distance-based LOD: mesh if this instance is close
                # enough to the camera AND its bucket actually has one (a
                # bucket only gets a mesh if some instance of it qualified
                # during Panorama Asset Generation -- this instance itself may
                # still be farther than the mesh distance even when a
                # sibling instance in the same bucket was close enough to
                # trigger meshing). Static since expected player movement is
                # small (~2 m) -- see mesh_lod_distance_m.
                camera_distance = float(np.linalg.norm(
                    np.array((position[0], place_y, position[2]), dtype=float) - camera_position
                ))
                lod_distance = self.config.mesh_lod_distance_overrides.get(
                    cls, self.config.mesh_lod_distance_m
                )
                use_mesh = category_mesh is not None and camera_distance <= lod_distance
                if not use_mesh and card_mesh is not None:
                    # Beyond the mesh LOD distance (or this bucket never got a
                    # reconstructed mesh at all) but a card LOD exists -- take it
                    # rather than the billboard. Everything below is shared: the
                    # card is an ordinary mesh, so it gets the same base snap,
                    # random yaw and uniform scale, and CategoryMeshRiggingStage
                    # will rig it for sway just like the near-LOD asset.
                    category_mesh, mesh_key, use_mesh = card_mesh, card_mesh_key, True

                if use_mesh:
                    # Use the shared bucket mesh with a random Y rotation.
                    # Mesh.fit_to_box recenters on the mesh's centroid, not its bounding-box
                    # center, so the mesh's lowest vertex is not reliably at -extent/2 --
                    # for a bottom-heavy shape (e.g. a tree trunk) the centroid sits below
                    # the bbox center, and assuming symmetry would still leave a gap above
                    # the terrain. Sample the mesh's actual lowest vertex (bounds[0][1])
                    # and offset by exactly that, so the true bottom -- not an assumed one
                    # -- lands on terrain_y.
                    # Scale this instance so the mesh's own HEIGHT matches the height
                    # its 2D box subtends at its own depth -- per instance, not per
                    # class and not per scene.
                    #
                    # Height specifically, for the reason object_scale.py's prior table
                    # documents: an equirectangular box's horizontal extent depends on
                    # the object's yaw relative to the camera (a tree seen through a gap
                    # vs. broadside differ by a lot) while its vertical extent does not.
                    # Width is the unreliable axis, so it must not be what sets scale.
                    #
                    # The previous max(width, height) got this wrong twice over, because
                    # it treated the scalar as if it were the mesh's height directly.
                    # fit_to_box(1.0, 1.0) normalises the LARGER of X/Y to 1.0 and leaves
                    # the other below it, so a uniform scale s renders a height of
                    # s * extent_y, not s. On this capture 16 of 28 category meshes are
                    # wider than tall (extent_y median 0.665, min 0.004), and 12 of 59
                    # mesh-rendered objects came out more than 10% from their detected
                    # height -- a tree box 2.70 x 1.92 m rendering 2.48 m tall (1.29x),
                    # another 1.59x, and a flower at 0.08x. Dividing by the mesh's own
                    # extent_y is exact in every case instead of right only when the mesh
                    # is tall AND the detection is taller than wide.
                    mesh_extents = category_mesh.mesh.bounds[1] - category_mesh.mesh.bounds[0]
                    mesh_extent_y = float(mesh_extents[1])
                    # A mesh that is essentially flat in Y has no meaningful height to
                    # match, and dividing by it explodes the other two axes (extent_y
                    # 0.004 would scale a 0.07 m flower into a 17 m wide sheet). That is
                    # a broken reconstruction, not a scale question -- fall back to the
                    # old footprint-driven behaviour rather than trusting the ratio.
                    if mesh_extent_y >= _MIN_MESH_HEIGHT_FRACTION * float(max(mesh_extents)):
                        mesh_scale = float(height) / mesh_extent_y
                    else:
                        self.log_warning(
                            f"Category mesh {mesh_key} is degenerate in Y "
                            f"(extent {mesh_extent_y:.4f} of {float(max(mesh_extents)):.4f}); "
                            f"scaling object {idx} by footprint instead of height"
                        )
                        mesh_scale = float(max(width, height))
                    mesh_min_y = float(category_mesh.mesh.bounds[0][1]) * mesh_scale
                    mesh_place_y = terrain_y - mesh_min_y if terrain_y is not None else position[1]
                    self.log_info(f"Creating mesh for {idx} ({cls}, bucket {bucket}, {camera_distance:.1f}m)")
                    mesh_obj = Object3D.mesh(mesh_key, x=position[0], y=mesh_place_y, z=position[2])
                    mesh_obj.set_rotation(0.0, float(rng.uniform(0.0, 360.0)), 0.0)
                    mesh_obj.set_scale(mesh_scale, mesh_scale, mesh_scale)
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
                    self.log_info(f"Creating billboard for {idx} ({cls}, bucket {bucket}, {camera_distance:.1f}m) using crop_{chosen_idx}")
                    billboard = Object3D.billboard(
                        f"crop_{chosen_idx}",
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

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, _, _, _, output_key = self._resolved_keys()

        return context.scene(output_key) is not None