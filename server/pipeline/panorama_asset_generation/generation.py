import json
import numpy as np
import torch
from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator, ModelGeneratorType
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import (
    ENVIRONMENT_CATEGORIES as _ENV_CATEGORIES, CategoryFilter, normalize_category,
    VEGETATION_CATEGORIES, GRASS_TUFT_CATEGORY,
)
from pipeline.grass_cover.cards import crossed_card_mesh
from util.lod_metrics import angular_height_px_from_box
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from PIL import Image as PILImage


# How far a candidate's width:height may disagree with its bucket's median before it
# is refused as that bucket's CARD TEXTURE -- see the card_sources block in _curate.
# Deliberately the same 4x as scene_generation's _MAX_MESH_ASPECT_RATIO, which asks
# the same question of a reconstruction against its detection: these are both "does
# this thing describe the subject it claims to", and there is no reason for the two
# to disagree about how much shape mismatch is too much.
_MAX_CARD_ASPECT_RATIO = 4.0


def _largest_consistent(pool: list[dict], reference: list[dict]) -> "dict | None":
    """Pick the highest-RESOLUTION candidate from `pool` whose shape fits its group.

    Used for both things this stage builds out of one crop: the card texture and
    the mesh representative. Both were previously taken from composite_score's
    ranking, and composite_score has no size term at all -- it ranks on
    confidence, fill ratio, depth proximity and occlusion -- so its winner is
    routinely the smallest usable crop in the group. For an image that is about
    to be stretched over metres of geometry, or fed to a single-view 3D
    reconstructor, pixel count is the property that decides whether the result is
    anything at all.

    `reference` supplies the aspect median the shape guard compares against, and
    is deliberately a different (wider) list than `pool`: the pool a mesh may be
    drawn from is already filtered down by distance, occlusion and size and can
    be a single entry, whose "median" says nothing. The group's whole candidate
    set is the stable shape reference.

    Returns None when nothing is measurable, so the caller can keep its old
    behaviour rather than being handed a bad pick.
    """
    def measurable(s) -> bool:
        b = s.get("box")
        return bool(b) and b[2] > 0 and b[3] > 0

    # Degenerate boxes are dropped, not clamped: a zero height gives an aspect of
    # 1/epsilon, which as a member of the list the median is taken over drags the
    # reference far enough to reject every real crop in the group.
    candidates = [s for s in pool if measurable(s)]
    if not candidates:
        return None
    shapes = sorted(s["box"][2] / s["box"][3] for s in reference if measurable(s)) \
        or sorted(s["box"][2] / s["box"][3] for s in candidates)
    median_aspect = shapes[len(shapes) // 2]

    def agrees(s) -> bool:
        aspect = s["box"][2] / s["box"][3]
        return max(aspect / median_aspect, median_aspect / aspect) <= _MAX_CARD_ASPECT_RATIO

    # If the guard rejects everything, the group has no self-consistent shape to
    # appeal to -- fall back to raw size rather than to nothing.
    shaped = [s for s in candidates if agrees(s)] or candidates
    return max(shaped, key=lambda s: s["box"][2] * s["box"][3])


def _delit_key(idx: int) -> str:
    """Asset name for a crop with its baked-in lighting removed.

    SceneGenerationStage prefers this over crop_{idx} when it exists, and the
    client resolves whichever name lands in Object3D.texture by string, so no
    client change is needed for a billboard to pick up the delit image.
    """
    return f"crop_delit_{idx}"
from util.device_utils import DeviceStrategy, preferred_device
from util.crop_scoring import composite_score, occlusion_score, mask_fill_ratio


class PanoramaAssetGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        billboard_distance_m: float = 10.0,
        min_mesh_angular_px: float = 90.0,
        viewer_px_per_degree: float = 34.0,
        generator_type: str = "TRELLIS",
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        billboard_top_k: int = 4,
        score_weight_confidence: float = 0.35,
        score_weight_fill_ratio: float = 0.25,
        score_weight_depth: float = 0.25,
        score_weight_occlusion: float = 0.6,
        occlusion_covered_fraction_threshold: float = 0.35,
        occlusion_disqualify_fraction: float = 0.6,
        occlusion_depth_margin: float = 0.10,
        max_background_fraction: float = 0.5,
        use_intrinsic_delighting: bool = True,
        intrinsic_delight_strength: float = 0.6,
        intrinsic_resolution: int = 768,
        intrinsic_agg_num: int = 2,
        max_mesh_faces: int = 8000,
        card_categories: list[str] | None = None,
        card_planes: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Classes that get a crossed-card LOD alongside (or instead of) their
        # reconstruction, published under category_mesh_{class}_{bucket}_card.
        #
        # Vegetation, because that is where a camera-facing billboard fails and where
        # single-view reconstruction cannot succeed. SAM3D reconstructs a shallow
        # relief from one crop, which for a conifer is a card with the thickness of a
        # card -- measured at 0.083 and 0.097 of its second-largest extent on two runs
        # of the Rainier capture, rejected as a sheet by SceneGenerationStage, and
        # every tree fell back to a quad that turns to face the viewer. In stereo, at
        # the distance trees actually stand, that quad reads as flat and swings as the
        # head moves. Three fixed planes carry the same photograph with parallax
        # between them and cost 12 triangles.
        #
        # grass_tuft is excluded because GrassCoverStage builds its own cards from
        # exemplar patches and applies a blade silhouette to them; these crops are
        # already alpha cutouts of one instance and need no silhouette.
        self.card_categories = frozenset(
            card_categories if card_categories is not None
            else (VEGETATION_CATEGORIES - {GRASS_TUFT_CATEGORY})
        )
        self.card_planes = int(card_planes)
        # Face budget per category mesh. SAM3D returns its reconstruction undecimated
        # and nothing here used to touch it, which is affordable for a hero object and
        # is not what these are: a category mesh is shared by every instance of its
        # (class, bucket), and those run to four figures. Measured on the Rainier
        # capture -- category_mesh_flower_0 at 348,650 faces x 1,187 placed instances
        # is 414 M triangles on its own, and the scene's category meshes totalled
        # 961 M across 21,185 objects.
        #
        # GrassCoverStage already reached this conclusion for its own assets and set
        # max_near_mesh_faces to 4000, measuring p95 surface error at 0.19% of the
        # bounding-box diagonal there (8000 -> 0.12%). 8000 here because these are
        # subjects the viewer looks AT rather than ground cover underfoot, and because
        # Mesh.decimate returns the original untouched when it is already inside
        # budget -- so a small asset pays nothing. 0 disables.
        self.max_mesh_faces = int(max_mesh_faces)
        # Strip the baked-in sun out of the crops that become billboards and
        # category meshes, the same way TerrainTextureGenerationStage already
        # strips it out of its reference patches -- and, critically, by the same
        # amount.
        #
        # This is the colour mismatch between objects and the ground they stand
        # on. Both come from the same panorama pixels, but they arrive at the
        # renderer through paths that disagree about what a texture IS: terrain
        # reference patches are blended 60% toward IntrinsicDiffusion's predicted
        # albedo, while object crops were passed through raw, at full baked
        # lighting. The client then lights BOTH again with the estimated sun and
        # ambient probe (SceneParamManager.ApplySun), so the same tree photographed
        # once renders at one brightness and saturation baked into the terrain
        # texture and a visibly different one as a billboard beside it -- reported
        # on the Rainier capture as background trees not matching the terrain's.
        #
        # Delighting objects rather than re-lighting terrain is the direction that
        # is also more correct: a renderer applying its own light wants albedo, not
        # a photograph of a lit surface. Matching strengths matters more than the
        # absolute value -- keep this equal to the terrain stage's own
        # intrinsic_delight_strength, and change them together.
        self.use_intrinsic_delighting = bool(use_intrinsic_delighting)
        self.intrinsic_delight_strength = float(intrinsic_delight_strength)
        self.intrinsic_resolution = int(intrinsic_resolution)
        self.intrinsic_agg_num = int(intrinsic_agg_num)
        # A detection whose box is mostly BACKGROUND is not an object, whatever it
        # got typed as. Two independent readings of "background", both already
        # computed by earlier stages, and a box failing either is rejected:
        #
        #   sky      -- fraction of the box PanoramaRegionStage types RegionType.SKY
        #   unmeasured depth -- fraction of the box sitting at the panorama depth
        #                       map's far clamp (max_depth_m, 100 m). Depth there is
        #                       not a measurement; it is the value assigned to
        #                       everything the map could not place, and 43% of a
        #                       typical panorama is at it.
        #
        # This is the gate that matters for the "large floating object" failure,
        # because the two defects compound. Size is angular_extent x depth (see
        # scene_generation/projection.py), so a box the depth map could not measure
        # is not merely mispositioned -- its metre size is angular_extent x 100,
        # a fabricated number that grows with how far away the thing looked.
        # Measured across five captures, every object over 10 m came from such a
        # box: a 38 m "lighthouse" of blank sky standing in the sea (Shark Fin
        # Cove), an 11.7 m moon hanging over a meadow (Mount Rainier), and in one
        # Paris capture a 37 m tower, a 21 m boat, a 16 m boat and a 14 m boat, all
        # at 99.9 m median depth with 59-100% of their pixels at the clamp.
        #
        # 0.5 rather than something stricter because a real structure legitimately
        # silhouettes against sky: over those same captures the genuine buildings
        # and boats measured 0.00-0.47 on both fractions and the rejects 0.54-1.00.
        # 1.0 disables the gate.
        self.max_background_fraction = float(max_background_fraction)
        self.billboard_distance_m = float(billboard_distance_m)
        # A group only earns a bespoke category mesh if some instance of it
        # subtends at least this many display pixels of HEIGHT. Meshing is
        # otherwise gated on distance alone (billboard_distance_m), which lets a
        # tiny-but-close subject through -- e.g. foreground alpine-meadow flowers
        # sit 0.6-1 m from the camera yet each occupy only 20-80 px of the frame
        # and, split into a bucket per colour, spawn a separate 3D mesh apiece
        # (observed: 7 flower meshes, several from singleton buckets, one from a
        # conf-0.03 "sheep in grass" miscrop). Bespoke meshes are meant for
        # prominent subjects; anything below this stays billboard-only (its pool
        # is still curated, so it isn't dropped -- just not meshified). 0 disables
        # the size gate (distance-only, prior behaviour).
        #
        # Replaced min_mesh_area_fraction, which measured the same intent as box
        # AREA over panorama area. Area is the wrong axis for the same reason
        # SceneGenerationStage scales meshes by height: an equirect box's width
        # depends on the subject's yaw relative to the camera and its height does
        # not, so an area gate rejects a broadside subject and admits the same
        # subject seen edge-on. Height is measured directly off the box with no
        # depth sample involved -- an equirect row IS an angle. See
        # util/lod_metrics.py.
        #
        # MUST STAY BELOW SceneGenerationStage's own min_mesh_angular_px, and by a
        # wide margin, or a group gets meshed here and nothing renders the asset.
        # The two are not the same measurement: this one is the RAW detected
        # angle, while Scene Generation measures the placed size, which carries
        # object_scale.py's correction on top (4.45x on the Rainier capture, and
        # it is fitted per run, so the ratio is not a constant to tune against).
        # 60 px here against 250 px there leaves room for a correction as low as
        # ~0.25x before the ordering inverts.
        self.min_mesh_angular_px = float(min_mesh_angular_px)
        # Display angular resolution the threshold above is quoted against; must
        # match Scene Generation's for the two to be comparable at all.
        self.viewer_px_per_degree = float(viewer_px_per_degree)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)
        self.billboard_top_k = billboard_top_k
        self.score_weight_confidence = score_weight_confidence
        self.score_weight_fill_ratio = score_weight_fill_ratio
        self.score_weight_depth = score_weight_depth
        self.score_weight_occlusion = score_weight_occlusion
        self.occlusion_covered_fraction_threshold = occlusion_covered_fraction_threshold
        self.occlusion_disqualify_fraction = occlusion_disqualify_fraction
        self.occlusion_depth_margin = occlusion_depth_margin


class PanoramaAssetGenerationStage(PipelineStage):
    """
    For each (class, bucket) visual-similarity group present in the scene (see
    ObjectCategoryClusteringStage -- bucket sub-divides a class into visually
    distinct variants, e.g. flower colors), curates a top-K billboard crop pool
    (billboard_top_k, ranked by the same composite score regardless of
    distance -- SceneGenerationStage draws from this pool at ANY distance) and,
    if any instance of the group is closer than billboard_distance_m, meshifies
    its best-scoring eligible instance as category_mesh_{class}_{bucket}.
    Groups where every instance is farther than billboard_distance_m stay
    billboard-only.

    metadata_{i}['position_only'] (ObjectCategoryClusteringStage -- a low-confidence
    crop visually corroborated against some class, trusted only for its world
    position) is always excluded here: never a mesh representative, never in a
    billboard pool, regardless of its class/bucket/score. So is
    metadata_{i}['synthetic'] (DistributionSynthesisStage's painted points, which
    run before this stage and are already counted in OBJECT_COUNT) -- a painted
    point has no crop of its own and only ever consumes a pool, never supplies one.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (with 'class', 'bucket', 'box'),
            crop_{i}, ContextKey.PANORAMA_OBJECT_DEPTH (depth on the ORIGINAL panorama,
            matching what objects were detected against), ContextKey.PANORAMA
    Writes: category_mesh_{class}_{bucket} for each qualifying group,
            "billboard_pools" ({"{class}::{bucket}": [idx, ...]}) for every group
    Config: billboard_distance_m (default 10.0 m), billboard_top_k (default 4),
            min_mesh_angular_px (default 90 -- a group whose largest box subtends
            less screen height than this stays billboard-only),
            viewer_px_per_degree (default 34.0 -- display the above is quoted in),
            generator_type (default TRELLIS),
            max_mesh_faces (default 8000 -- per-mesh face budget; 0 disables)
    """

    @classmethod
    def config_class(cls):
        return PanoramaAssetGenerationConfiguration

    def __init__(self, config: PanoramaAssetGenerationConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to process, skipping")
            return context

        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        # The ORIGINAL panorama's region typing (not the _terrain one, which is
        # derived from the object-removed panorama and so has nothing to say about
        # where the objects were). Optional: absent, the background gate falls back
        # to its depth half alone.
        region_type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        region_type_map = region_type_depth.depth if region_type_depth is not None else None

        (group_best, billboard_pools, card_sources, skipped_debug,
         disqualified_debug, synthetic_skipped) = self._curate(
            object_count, context.input_object, context.input_image, panorama_depth, pano_w, pano_h,
            region_type_map=region_type_map,
        )

        context.add_object("billboard_pools", billboard_pools)
        # Detections this stage judged to be BACKGROUND rather than objects, handed
        # forward so SceneGenerationStage can decline to place them.
        #
        # max_background_fraction already identifies them correctly -- the whole
        # "large floating object" failure it was built for -- but it only gated what
        # this stage BUILDS. Scene Generation places every surviving metadata_{i}
        # regardless of whether an asset was curated for it, so a detection rejected
        # here still arrived in the scene, just without a curated pool behind it.
        # Measured on the Rainier capture (2026-08-17 13:33 run), all three of them
        # standing on the mountainside at 60-96 m: a 20.3 x 7.7 m "aircraft" with 74%
        # of its box at the depth clamp, a 16.7 x 8.5 m "animal" at 67%, and a
        # 4.8 x 9.2 m "person" whose box is 100% sky AND 100% unmeasured depth.
        #
        # Only the `background` reason travels. The others must not: `environment`
        # and `indeterminate` are already skipped independently by Scene Generation,
        # `category_filter` is a rendering preference rather than a claim the thing
        # is not real, and `position_only`/`too_small_for_mesh` are explicitly still
        # meant to be placed (as billboards) -- forwarding those would empty the scene.
        context.add_object(
            "background_rejected",
            sorted({d["idx"] for d in skipped_debug if d.get("reason") == "background"}),
        )
        self._write_debug(skipped_debug, group_best, disqualified_debug, billboard_pools, synthetic_skipped)

        # Everything that will actually be rendered: every curated billboard pool
        # plus every mesh representative. Done after curation so the rejected
        # crops -- environment, background, position_only -- cost nothing.
        rendered = {i for pool in billboard_pools.values() for i in pool}
        rendered.update(idx for idx, _depth, _score in group_best.values())
        # Card textures are picked by resolution, not by pool rank, so the crop a
        # card uses is routinely NOT in its group's top-K -- it has to be delit
        # explicitly or the card ends up the one asset in the scene still carrying
        # baked-in sunlight while everything around it is albedo.
        rendered.update(
            idx for key, idx in card_sources.items()
            if key.partition("::")[0] in self.config.card_categories
        )
        if self.config.use_intrinsic_delighting:
            written = self._delight_crops(context, rendered)
            self.log_info(
                f"Delit {written}/{len(rendered)} rendered crop(s) at strength "
                f"{self.config.intrinsic_delight_strength:.2f} to match the terrain texture"
            )

        # Before the group_best early-out below: a card is built from a curated crop,
        # not from a reconstruction, so it exists for every pooled group regardless of
        # whether anything was close enough to mesh. A scene where nothing qualified
        # for 3D still wants its vegetation on cards rather than on quads.
        cards = self._build_card_meshes(context, billboard_pools, card_sources)
        if cards:
            self.log_info(f"Built {cards} crossed-card LOD mesh(es)")

        if not group_best:
            self.log_info("No objects within 3D generation distance")
            return context

        # Second pass: generate one mesh per qualifying (class, bucket) group.
        asset_task = self.create_progress(len(group_best), "Generating 3D assets…")
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=self.config.generator_type)

        try:
            for (obj_class, bucket), (idx, depth, score) in group_best.items():
                mesh_key = f"category_mesh_{obj_class}_{bucket}"

                cached = context.mesh(mesh_key)
                if cached is not None:
                    self.log_info(f"  {mesh_key}: cached ({cached.vertex_count}v {cached.face_count}f)")
                    self.advance_progress(asset_task)
                    continue

                self.log_info(f"  {mesh_key}: {depth:.1f} m, score {score:.2f} → 3D mesh (crop_{idx})")
                # Prefer the delit crop: meshify BAKES this image into the GLB's
                # texture, so a lit crop here bakes the sun into geometry that the
                # client then lights again -- the same double-lighting the
                # billboard path has, but permanent. Falls back to the raw crop
                # when delighting is off or failed for this one image.
                crop = context.image(_delit_key(idx)) or context.input_image(f"crop_{idx}")

                fill_ratio = mask_fill_ratio(crop)
                if fill_ratio is not None and fill_ratio <= 0.0:
                    self.log_info(f"  {mesh_key}: crop_{idx} has an empty mask, skipping mesh (billboard-only)")
                    self.advance_progress(asset_task)
                    continue

                temp_path = self.temp / mesh_key if self.temp is not None else None
                if temp_path is not None:
                    temp_path.mkdir(parents=True, exist_ok=True)
                super().clean_up()
                try:
                    mesh = gen.meshify(crop, temp_path, seed=self.seed)
                    mesh = mesh.repair()
                    raw_faces = mesh.face_count
                    mesh = mesh.decimate(self.config.max_mesh_faces)
                    mesh.fit_to_box(1.0, 1.0)
                except Exception as e:
                    # A single degenerate crop (e.g. a near-empty mask the
                    # generator's own preprocessing collapses to nothing)
                    # shouldn't take down every other group's mesh --
                    # scene_generation.py already falls back to this group's
                    # billboard pool when category_mesh_{class}_{bucket} is
                    # absent.
                    self.log_info(f"  {mesh_key}: meshify failed ({e}), falling back to billboard-only")
                    self.advance_progress(asset_task)
                    continue
                context.add_mesh(mesh_key, mesh)
                decimated = (
                    f" (decimated from {raw_faces}f)" if mesh.face_count < raw_faces else ""
                )
                self.log_info(
                    f"  {mesh_key}: {mesh.vertex_count}v {mesh.face_count}f{decimated}"
                )
                self.advance_progress(asset_task)
        finally:
            gen.close()

        self.finish_progress(asset_task)
        return context

    def _build_card_meshes(
        self, context: PipelineContext, billboard_pools: dict, card_sources: dict,
    ) -> int:
        """Publish a crossed-card LOD for every configured group with a curated pool.

        Textured with the group's HIGHEST-RESOLUTION eligible crop (card_sources, see
        _curate) rather than its top-ranked one, preferring the delit variant for the
        same reason the mesh path does: the client lights the card again, so it wants
        albedo rather than a photograph with the sun in it. The crop's own alpha is the
        silhouette (crossed_card_mesh masks at 0.5), which is why no shaping pass is
        needed here.

        Keyed category_mesh_{class}_{bucket}_card, which SceneGenerationStage already
        resolves generically -- it takes the card beyond the mesh LOD distance, when a
        bucket never got a reconstruction, and now when a reconstruction was rejected.
        """
        if not self.config.card_categories or self.config.card_planes < 1:
            return 0

        built = 0
        for pool_key, pool in billboard_pools.items():
            obj_class, _, bucket = pool_key.partition("::")
            if obj_class not in self.config.card_categories or not pool:
                continue

            key = f"category_mesh_{obj_class}_{bucket}_card"
            if context.mesh(key) is not None:
                continue

            # Falls back to the pool's top crop only if this group produced no
            # measurable box at all -- prior behaviour, and never worse than it.
            source = card_sources.get(pool_key, pool[0])
            crop = context.image(_delit_key(source)) or context.input_image(f"crop_{source}")
            if crop is None:
                continue
            texture = crop.rgba() if hasattr(crop, "rgba") else crop

            try:
                mesh = crossed_card_mesh(texture, plane_count=self.config.card_planes)
            except Exception as e:
                # One unusable crop costs one group its card, not the whole stage its
                # cards -- that group simply falls back to a billboard as before.
                self.log_info(f"  {key}: card build failed ({e}), skipping")
                continue

            context.add_mesh(key, mesh)
            self.log_info(
                f"  {key}: {self.config.card_planes} planes from crop_{source} "
                f"({texture.width}x{texture.height} px)"
            )
            built += 1
        return built

    def _delight_crops(self, context: PipelineContext, indices: "set[int]") -> int:
        """Write crop_delit_{i} for every crop that will be rendered.

        Only the crops that actually reach the client are delit -- the mesh
        representatives and the curated billboard pools -- rather than all of
        OBJECT_COUNT. IntrinsicDiffusion is a diffusion model and this is a
        per-image cost, so on a capture with hundreds of detections the
        difference is the whole run.

        Deliberately a NEW key rather than an overwrite of crop_{i}.
        VideoObjectExtractionStage runs after this one and matches crop_{i}
        against the generated video with SAM2; that video is rendered from the
        original, fully-lit panorama, so handing the tracker a delit crop would
        make it match its own subject worse. Rendering wants albedo, tracking
        wants the photograph, and they can have one each.
        """
        if not indices or not self.config.use_intrinsic_delighting:
            return 0
        from pipeline.intrinsic_images.image_intrinsics import ImageIntrinsics
        from util.image_utils import Image as UtilImage

        task = self.create_progress(len(indices), "Delighting object crops…")
        intrinsics = ImageIntrinsics(self.preferred_device)
        written = 0
        try:
            for idx in sorted(indices):
                crop = context.input_image(f"crop_{idx}")
                if crop is None:
                    self.advance_progress(task)
                    continue
                pil = crop.rgba() if hasattr(crop, "rgba") else crop
                try:
                    result = intrinsics.intrinsic_images(
                        UtilImage(pil.convert("RGB")),
                        temp_path=self.temp,
                        resolution=self.config.intrinsic_resolution,
                        agg_num=self.config.intrinsic_agg_num,
                    )
                    albedo = result.albedo_image()
                except Exception as exc:
                    # One crop the model chokes on must not cost the whole scene
                    # its delighting -- and a crop with no delit variant falls
                    # back to crop_{i} downstream, which is exactly the old
                    # behaviour for that one object.
                    self.log_info(f"  crop_{idx}: delighting failed ({exc}), leaving it lit")
                    self.advance_progress(task)
                    continue

                strength = self.config.intrinsic_delight_strength
                if strength < 1.0:
                    blended = (
                        np.asarray(albedo.convert("RGB"), dtype=np.float32) * strength
                        + np.asarray(pil.convert("RGB"), dtype=np.float32) * (1.0 - strength)
                    )
                    albedo = PILImage.fromarray(blended.clip(0, 255).astype(np.uint8), "RGB")
                # The alpha channel is the segmentation cutout; without it a
                # billboard renders as an opaque rectangle of sky around its
                # subject.
                if pil.mode == "RGBA":
                    albedo = albedo.convert("RGBA")
                    albedo.putalpha(pil.split()[-1])

                context.add_image(_delit_key(idx), albedo)
                written += 1
                self.advance_progress(task)
        finally:
            intrinsics.close()
        self.finish_progress(task)
        return written

    @staticmethod
    def _box_patch(array, box, pano_w, pano_h):
        """The `box` region of a full-sphere map, rescaled to that map's own resolution.

        Boxes are in panorama pixel space, and neither map is guaranteed to share
        it -- the depth map in particular is routinely a different resolution (see
        _sample_object_depth, which does the same rescale for the same reason).
        Indexing a half-size map with panorama coordinates reads the wrong region
        entirely, and for a box near the bottom simply reads nothing.
        """
        if array is None or box is None or not pano_w or not pano_h:
            return None
        rows, cols = array.shape[:2]
        sx, sy = cols / float(pano_w), rows / float(pano_h)
        x1 = max(0, int(round(float(box[0]) * sx)))
        y1 = max(0, int(round(float(box[1]) * sy)))
        x2 = min(cols, max(x1 + 1, int(round((float(box[0]) + float(box[2])) * sx))))
        y2 = min(rows, max(y1 + 1, int(round((float(box[1]) + float(box[3])) * sy))))
        patch = array[y1:y2, x1:x2]
        return patch if patch.size else None

    def _background_fractions(
        self, box, region_type_map, panorama_depth, far_depth, pano_w, pano_h,
    ) -> tuple[float, float]:
        """(sky fraction, unmeasured-depth fraction) of this box. See max_background_fraction."""
        sky = clamped = 0.0
        patch = self._box_patch(region_type_map, box, pano_w, pano_h)
        if patch is not None:
            sky = float(np.mean(patch == int(RegionType.SKY)))
        if panorama_depth is not None and far_depth:
            patch = self._box_patch(panorama_depth.depth, box, pano_w, pano_h)
            if patch is not None:
                clamped = float(np.mean(patch >= far_depth * 0.99))
        return sky, clamped

    def _curate(
        self, object_count: int, get_metadata, get_image, panorama_depth, pano_w, pano_h,
        region_type_map=None,
    ) -> tuple[dict[tuple[str, int], tuple[int, float, float]], dict[str, list[int]], list, list, int]:
        """Shared by run() and has_expected_output() (callers pass either the
        input_* accessors to see state as of the previous stage, or the plain
        accessors to see this stage's own already-cached output).

        Returns (group_best, billboard_pools, skipped_debug, disqualified_debug,
        synthetic_skipped).
        group_best: (class, bucket) -> (idx, depth, score) of the winning mesh
        representative, only for groups with at least one instance closer than
        billboard_distance_m. billboard_pools: "{class}::{bucket}" -> top-K
        crop indices by score, for EVERY (class, bucket) group regardless of
        distance -- a group's billboard pool must stay usable even when no
        instance qualified for meshing.
        """
        threshold = self.config.billboard_distance_m
        skipped_debug = []
        synthetic_skipped = 0
        depth_by_idx: dict[int, tuple[list, float]] = {}
        candidates_by_group: dict[tuple[str, int], list[dict]] = {}
        far_depth = (
            float(np.nanmax(panorama_depth.depth)) if panorama_depth is not None else None
        )

        for idx in range(object_count):
            metadata = get_metadata(f"metadata_{idx}")
            if metadata is None:
                continue

            box = metadata.get("box")
            depth = self._sample_object_depth(box, panorama_depth, pano_w, pano_h)
            if box is not None and depth is not None:
                depth_by_idx[idx] = (box, depth)

            # Detections created after ObjectTypingStage (ObjectDetectionStage runs
            # 17 stages later) carry free-text GroundingDINO labels rather than one
            # of this module's category names, so the membership test below cannot
            # see them for what they are -- "moon moonlight", "cloud", "sea water"
            # and "stone" all read as placeable objects. See normalize_category.
            obj_class = normalize_category(metadata.get("class"))
            if obj_class in _ENV_CATEGORIES or obj_class == "indeterminate":
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "environment" if obj_class in _ENV_CATEGORIES else "indeterminate",
                })
                continue

            if not metadata.get("synthetic") and self.config.max_background_fraction < 1.0:
                sky_fraction, clamped_fraction = self._background_fractions(
                    box, region_type_map, panorama_depth, far_depth, pano_w, pano_h,
                )
                limit = self.config.max_background_fraction
                if sky_fraction > limit or clamped_fraction > limit:
                    skipped_debug.append({
                        "idx": idx,
                        "class": obj_class,
                        "reason": "background",
                        "sky_fraction": round(sky_fraction, 3),
                        "unmeasured_depth_fraction": round(clamped_fraction, 3),
                    })
                    continue
            if not self.config.category_filter.allows(obj_class or ""):
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "category_filter",
                })
                continue
            if metadata.get("position_only"):
                # ObjectCategoryClusteringStage visually corroborated this
                # low-confidence crop enough to trust its position for
                # ObjectDistributionStage, but not enough to anchor a bucket,
                # get meshed, or appear as a billboard from its own crop.
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "position_only",
                })
                continue
            if metadata.get("synthetic"):
                # DistributionSynthesisStage runs BEFORE this stage and bumps
                # OBJECT_COUNT, so the loop above walks its painted points too --
                # but a painted point has no detection box and no crop_{idx} of
                # its own; it exists to CONSUME a pool, never to populate one.
                # Left in, each one scored on pure defaults (confidence and
                # fill_ratio both fall back to 0.5, depth is unsamplable so it's
                # forced to the far value, giving a flat ~0.30) and so outranked
                # every real detection in a bucket whose instances all sit past
                # billboard_distance_m with a typical sub-0.5 CLIP confidence.
                # Those buckets' billboard_pools then filled with indices that
                # have no image behind them, and SceneGenerationStage rendered
                # crop_{idx} for a crop that was never written -- observed as
                # distant classes (trees) disappearing from the scene while close
                # ones (flowers) survived on their nonzero depth score.
                #
                # Counted rather than listed per-index: a painted population is
                # routinely thousands of points, which would swamp asset_debug.json
                # with entries carrying no information beyond their own count.
                synthetic_skipped += 1
                continue

            bucket = metadata.get("bucket") or 0
            key = (obj_class, int(bucket))
            candidates_by_group.setdefault(key, []).append({
                "idx": idx, "box": box, "depth": depth, "metadata": metadata,
            })

        group_best: dict[tuple[str, int], tuple[int, float, float]] = {}
        billboard_pools: dict[str, list[int]] = {}
        card_sources: dict[str, int] = {}
        disqualified_debug = []
        for (obj_class, bucket), candidates in candidates_by_group.items():
            scored = []
            for candidate in candidates:
                # A candidate whose depth couldn't be sampled still belongs in
                # the billboard pool (it just can't be scored on depth or
                # compared for occlusion) -- treat it as "far" rather than
                # dropping it, so a bad depth sample doesn't silently shrink
                # the billboard pool.
                depth = candidate["depth"] if candidate["depth"] is not None else threshold * 10.0
                occlusion = occlusion_score(
                    candidate["idx"], candidate["box"], depth, depth_by_idx, self.config.occlusion_depth_margin,
                )
                crop = get_image(f"crop_{candidate['idx']}")
                score = composite_score(
                    candidate["metadata"], crop, depth, occlusion, threshold,
                    self.config.score_weight_confidence, self.config.score_weight_fill_ratio,
                    self.config.score_weight_depth, self.config.score_weight_occlusion,
                    self.config.occlusion_covered_fraction_threshold,
                )
                disqualified = occlusion >= self.config.occlusion_disqualify_fraction
                scored.append({**candidate, "depth": depth, "occlusion": occlusion, "score": score, "disqualified": disqualified})

            scored_by_rank = sorted(scored, key=lambda s: s["score"], reverse=True)
            pool_key = f"{obj_class}::{bucket}"
            billboard_pools[pool_key] = [s["idx"] for s in scored_by_rank[: self.config.billboard_top_k]]

            # Card texture, chosen by RESOLUTION rather than by composite_score.
            #
            # A card is one image stretched over 3-5 m of geometry and viewed from
            # a few metres away, so the only property that matters is how many
            # pixels the crop actually has. composite_score ranks on confidence,
            # fill ratio, depth proximity and occlusion and -- as the size gate
            # below says of the same defect -- size is not one of its terms, so its
            # winner is routinely a small nearby instance.
            #
            # Measured on the Rainier capture, where this produced the scene's most
            # conspicuous artefact: every one of tree::0's 106 instances rendered as
            # a card up to 5.3 m tall wearing crop_392, a 67x63 px `split_watershed`
            # fragment of the treeline at the horizon captioned "a close up of a
            # tree with a brown background" -- a featureless pale-green smudge with
            # no tree in it. The same group holds crop_170 at 119x325 px: nine times
            # the pixels, and conifer-shaped. plant::0 is 39x77 against 39x366
            # available, flower::0 78x79 against 73x153.
            #
            # Ranked over the group's whole eligible candidate set, not over the
            # top-K pool -- the pool is four entries deep and its best (51x165) is
            # still a third of what the group has. Occlusion-disqualified crops are
            # excluded where possible: a crop that is half some other object is a
            # bad texture no matter how many pixels it has.
            #
            # Guarded on ASPECT, because "largest" on its own will happily pick a
            # large MISdetection. A bucket is a visual-similarity cluster, so its
            # members agree about the subject's shape, and a candidate that does not
            # is the one that does not belong. Same test the placement-side mesh
            # gates use (_MAX_MESH_ASPECT_RATIO, also 4x) and the same reasoning:
            # compared as a ratio-of-ratios against the group's MEDIAN aspect, so it
            # is symmetric and a single outlier cannot move the reference.
            #
            # Rainier again: flower::4's largest candidate is crop_69, 133x69 px --
            # a landscape crop of a SNOW PATCH, captioned "there is a white vase
            # with a bird on it", carrying a perfectly ordinary 0.94 detection score
            # and 0.71 confidence, so nothing else in the pipeline separates it. Its
            # bucket's other seven members are all portrait (median aspect 0.35);
            # crop_69 is 1.93, a 5.5x disagreement. With the guard the bucket picks
            # crop_268 (38x128, "a yellow flower that is on a stick"). Across every
            # carded group on that capture this changes only flower::4.
            undisqualified = [s for s in scored if not s["disqualified"]] or scored
            card_source = _largest_consistent(undisqualified, scored)
            if card_source is not None:
                card_sources[pool_key] = card_source["idx"]

            within_threshold = [s for s in scored if s["depth"] < threshold]
            if not within_threshold:
                continue  # nothing close enough to mesh -- billboard-only group

            eligible = [s for s in within_threshold if not s["disqualified"]] or within_threshold

            # Size gate: a bespoke mesh is only worth generating for a prominent
            # subject. A group whose instances are ALL close-but-tiny (e.g. a bucket
            # of single meadow flowers) stays billboard-only -- its pool was already
            # curated above, so it isn't lost, just not meshified. Skipped when
            # panorama dims or a candidate's box are unavailable (can't measure),
            # preserving distance-only behaviour; disabled entirely at
            # min_mesh_angular_px == 0.
            #
            # Applied to the CANDIDATE SET, with the winner then chosen among whatever
            # survives -- deliberately not to the score-winner alone, which is what it
            # used to do. composite_score ranks on confidence, fill ratio, depth
            # proximity and occlusion; SIZE IS NOT ONE OF ITS TERMS. So the winner is
            # routinely a small nearby instance, and testing the gate against it threw
            # away entire groups that contained a genuinely prominent subject.
            #
            # Measured on a Paris capture (4096x2048 panorama), where the previous
            # area form of this gate rejected 14 of 17 billboard-only groups:
            #
            #     group      score-winner        largest instance in the same group
            #     tower::1   44x44 px            126x529 px  <- the Eiffel Tower
            #     boat::0    46x46 px            297x189 px
            #
            # Both are well ABOVE the gate on their largest instance and both were
            # denied a mesh, so every one of the tower's 14 instances rendered as a
            # billboard. The other 12 rejected groups have largest == winner, i.e.
            # they really are all-tiny, and they stay billboard-only under this too.
            #
            # Note what changed with the switch from area to height: the boat is
            # 297 WIDE by 189 tall, so it scored far higher on area than on height,
            # while the tower is 126x529 and scores far higher on height. Height is
            # the honest one -- 529 rows of panorama is 46 degrees of elevation no
            # matter which way the tower faces, whereas the boat's 297 columns are
            # 297 only because it happens to sit broadside.
            if self.config.min_mesh_angular_px > 0 and pano_h:
                def _angular_px(s) -> "float | None":
                    b = s.get("box")
                    return None if b is None else angular_height_px_from_box(
                        b[3], pano_h, self.config.viewer_px_per_degree
                    )

                measured = [(s, _angular_px(s)) for s in eligible]
                # An unmeasurable box passes, same as before -- "can't measure" has
                # never meant "reject".
                prominent = [
                    s for s, px in measured
                    if px is None or px >= self.config.min_mesh_angular_px
                ]
                if not prominent:
                    # Every px is a real number here (a None would have passed), so
                    # report the closest this group came to clearing the gate.
                    best, best_px = max(measured, key=lambda pair: pair[1])
                    skipped_debug.append({
                        "idx": best["idx"],
                        "class": obj_class,
                        "reason": "too_small_for_mesh",
                        "bucket": bucket,
                        "angular_px": round(best_px, 1),
                    })
                    continue
                eligible = prominent

            # Mesh representative, by RESOLUTION among what survived the gates above
            # -- not by composite_score, for the same reason the card texture is not.
            #
            # This is the single most consequential crop choice the stage makes:
            # SAM3D reconstructs the whole bucket's geometry from it, every instance
            # of the bucket renders that geometry, and a single-view reconstructor
            # given too few pixels does not return a worse object, it returns a
            # DIFFERENT one -- a flat pancake that scene_generation's shape gates
            # then correctly refuse, so the whole bucket falls back to billboards.
            #
            # Measured on the Rainier capture (2026-08-17 13:33 run), where 205 of
            # 471 placed instances were refused this way and every single refused
            # bucket had been reconstructed from a crop far smaller than its own
            # group held:
            #
            #     mesh          score-winner   largest available   gate verdict
            #     person_0      27x65          50x197  (5.6x)      h_frac 0.059, aspect 16.9 -> all 15 refused
            #     tree_0        67x63          119x325 (9.2x)      thickness 0.099 -> all 76 refused
            #     flower_1      32x78          117x166 (7.8x)      thickness 0.311 -> all 59 refused
            #     plant_2       52x40          69x180  (6.0x)      h_frac 0.012, aspect 80.6 -> all 26 refused
            #     rock_0        23x73          44x187  (4.9x)      aspect 1.00 vs 0.23 -> 3 refused
            #
            # A 27x65 crop of a person is not enough evidence for a person, and the
            # pancake SAM3D returned from it is the honest answer to the question it
            # was asked. Asking a better question is this fix.
            #
            # Drawn from `eligible`, so it still respects every quality filter above
            # (within billboard_distance_m, not occlusion-disqualified, over
            # min_mesh_angular_px) -- resolution replaces score only as the final
            # choice among crops already judged usable. The aspect guard's reference
            # is the group's whole candidate set rather than `eligible`, which can be
            # a single entry whose own median means nothing.
            winner = _largest_consistent(eligible, scored) or max(eligible, key=lambda s: s["score"])

            group_best[(obj_class, bucket)] = (winner["idx"], winner["depth"], winner["score"])
            for s in within_threshold:
                if s["disqualified"]:
                    disqualified_debug.append({
                        "idx": s["idx"],
                        "class": obj_class,
                        "bucket": bucket,
                        "occlusion": round(s["occlusion"], 3),
                        "chosen_anyway": s["idx"] == winner["idx"],
                    })

        return (group_best, billboard_pools, card_sources, skipped_debug,
                disqualified_debug, synthetic_skipped)

    def _write_debug(
        self, skipped: list, group_best: dict, disqualified: list, billboard_pools: dict,
        synthetic_skipped: int = 0,
    ):
        if self.output is None:
            return
        payload = {
            "billboard_distance_m": self.config.billboard_distance_m,
            "billboard_top_k": self.config.billboard_top_k,
            "min_mesh_angular_px": self.config.min_mesh_angular_px,
            "viewer_px_per_degree": self.config.viewer_px_per_degree,
            "summary": {
                "skipped_env_or_filtered": len(skipped),
                "skipped_synthetic": synthetic_skipped,
                "groups_billboard_only": len(billboard_pools) - len(group_best),
                "groups_meshified": len(group_best),
            },
            "skipped": skipped,
            "occlusion_disqualified": disqualified,
            "groups": [
                {"class": cls, "bucket": bucket, "representative_idx": idx, "depth_m": round(depth, 2), "score": round(score, 3)}
                for (cls, bucket), (idx, depth, score) in group_best.items()
            ],
            "billboard_pools": billboard_pools,
        }
        with open(self.output / "asset_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _sample_object_depth(self, box, panorama_depth, pano_w, pano_h) -> float | None:
        """Sample median depth in a patch around the bbox centre in the panorama depth map."""
        if box is None or panorama_depth is None or pano_w is None or pano_h is None:
            return None

        bx, by, bw, bh = box
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        sx = panorama_depth.width / pano_w
        sy = panorama_depth.height / pano_h
        dx = int(round(cx * sx))
        dy = int(round(cy * sy))

        r = 5
        x1 = max(0, dx - r)
        x2 = min(panorama_depth.width, dx + r)
        y1 = max(0, dy - r)
        y2 = min(panorama_depth.height, dy + r)

        patch = panorama_depth.depth[y1:y2, x1:x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            # No OBJECT_COUNT anywhere upstream means Object Segmentation is
            # disabled (permanent, not "pending") -- nothing to generate
            # assets for and never will be. Matches what count == 0 would
            # already return below (the loop is simply skipped); treating
            # None differently forced this stage, and everything after it via
            # the dirty cascade, to rerun on every single invocation.
            return True
        if context.object("billboard_pools") is None:
            return False
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        region_type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        group_best, _pools, _cards, _skipped, _disq, _syn = self._curate(
            count, context.object, context.image, panorama_depth, pano_w, pano_h,
            region_type_map=region_type_depth.depth if region_type_depth is not None else None,
        )
        for obj_class, bucket in group_best:
            if context.mesh(f"category_mesh_{obj_class}_{bucket}") is None:
                return False
        # Deliberately NOT checking for the _card meshes, even though run() publishes
        # them. _build_card_meshes skips a group whose crop is missing or whose card
        # build raised, by design -- so requiring the full set here would leave any
        # such group reporting this stage incomplete on every single invocation, and
        # with it every stage downstream via the dirty cascade. A context carrying
        # cards built under an older texture-selection rule is a --rerun, not a
        # cache-validity question.
        # Deliberately NOT checking for crop_delit_{i}. run() tolerates
        # IntrinsicDiffusion failing on an individual crop and leaves that one lit,
        # so requiring the full set here would make a permanently-failing crop
        # report this stage incomplete on every invocation -- and with it every
        # stage downstream, forever. This mirrors run()'s own tolerance, which is
        # the rule that keeps the dirty cascade from latching on.
        #
        # The cost is that turning use_intrinsic_delighting on for an
        # already-cached scene doesn't retroactively delight it; billboards fall
        # back to crop_{i} until the stage is rerun with --rerun.
        #
        # Card LODs are left out for the same reason and with the same cost.
        # _build_card_meshes skips a group whose top crop is missing or whose card
        # fails to build, so requiring the full set here could latch the cascade on a
        # group that can never produce one -- and SceneGenerationStage already treats
        # a missing card as "billboard instead", which is exactly the old behaviour.
        # Enabling cards on a cached scene therefore needs --rerun too.
        return True

    def model_names(self) -> list[str]:
        names = ModelGenerator.model_names(type=self.config.generator_type)
        if self.config.use_intrinsic_delighting:
            from pipeline.intrinsic_images.image_intrinsics import ImageIntrinsics
            names = names + ImageIntrinsics.model_names()
        return names

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.OBJECT_COUNT)
        if count is None or count == 0:
            return None

        seen_groups: set[tuple[str, int]] = set()
        images = []
        total_verts = 0
        total_faces = 0

        for i in range(count):
            meta = context.object(f"metadata_{i}") or {}
            obj_class = meta.get("class", f"Object {i + 1}")
            bucket = int(meta.get("bucket") or 0)
            group = (obj_class, bucket)
            mesh_key = f"category_mesh_{obj_class}_{bucket}"
            mesh = context.mesh(mesh_key)
            crop = context.image(f"crop_{i}")

            if group not in seen_groups:
                seen_groups.add(group)
                if mesh is not None:
                    total_verts += mesh.vertex_count
                    total_faces += mesh.face_count
                if crop is not None and len(images) < 6:
                    label = f"{obj_class} #{bucket}" if bucket else obj_class
                    if mesh is not None:
                        label += f" ({mesh.vertex_count:,}v)"
                    images.append((crop.image, label))

        reconstructed = len(seen_groups)
        stats = {"Categories reconstructed": str(reconstructed)}
        if reconstructed > 0:
            stats["Total vertices"] = f"{total_verts:,}"
            stats["Total triangles"] = f"{total_faces:,}"
            stats["Generator"] = self.config.generator_type.name
        return ReportSection(
            stage_name=self.name,
            title="3D Object Reconstruction",
            body=(
                "One 3D mesh is generated per visual-similarity bucket within each object "
                "category, using the closest instance in that bucket as the representative "
                f"crop. The {self.config.generator_type.name} model reconstructs a textured "
                "mesh, which is normalised to a 1 m canonical box. At scene placement, "
                "instances closer than the mesh distance threshold use the bucket's mesh "
                "(with a random rotation); farther instances use a billboard drawn from the "
                "bucket's curated crop pool."
            ),
            images=images,
            stats=stats,
        )

    def clean_up(self):
        super().clean_up()
