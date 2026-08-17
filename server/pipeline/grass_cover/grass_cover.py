import json
from logging import Logger
from typing import Any, Optional

import numpy as np
import PIL.Image
import torch
from scipy.ndimage import distance_transform_edt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from pipeline.grass_cover.cards import apply_tuft_silhouette, crossed_card_mesh
from pipeline.grass_cover.grass_area import grass_area_mask, area_square_meters
from pipeline.model_generation.model_generation import ModelGenerator, ModelGeneratorType
from pipeline.object_clustering.dinov2_embedder import DinoV2Embedder
from pipeline.object_typing.categories import GRASS_TUFT_CATEGORY
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image

# Written on a completed run so has_expected_output has a marker of its own.
# OBJECT_COUNT and metadata_{idx} are both inherited from earlier stages and so
# can't distinguish "this stage ran" from "some earlier stage wrote objects" --
# the same reasoning DistributionSynthesisStage's _RAN_MARKER documents.
_RAN_MARKER = "grass_cover_complete"
# What contribute_report renders, captured at run time rather than recomputed:
# the mask itself is a 4096^2 array with no other consumer, so keeping the few
# numbers derived from it is cheaper than keeping (or rebuilding) the array.
_SUMMARY_KEY = "grass_cover_summary"


def _mesh_key(bucket: int) -> str:
    return f"category_mesh_{GRASS_TUFT_CATEGORY}_{bucket}"


def _card_mesh_key(bucket: int) -> str:
    return f"category_mesh_{GRASS_TUFT_CATEGORY}_{bucket}_card"


def _observed_grass_fraction(area_stats: dict) -> "float | None":
    """Share of the ground the ORIGINAL PHOTOGRAPH shows that types as grass.

    Front hemisphere only -- +Z is panorama theta 0, the direction the capture was
    taken in and the only part of the equirect that is a photograph rather than a
    generation. Denominator is the depth-sampled area rather than the whole disc,
    so a scene whose near field simply wasn't measured isn't penalised for it;
    this asks what the ground we could see is made of, not how much we could see.

    None when the front hemisphere has no sampled ground at all (nothing to judge
    on -- leave the decision to the other gates rather than inventing a verdict).
    See GrassCoverConfiguration.min_observed_grass_fraction for the measurements.
    """
    front = (area_stats or {}).get("front")
    if not front:
        return None
    sampled = float(front.get("sampled_m2") or 0.0)
    if sampled <= 0.0:
        return None
    return float(front.get("grass_typed_m2") or 0.0) / sampled


def _exemplar_greenness(patches: list) -> "float | None":
    """Mean excess-green of the exemplar patches, over their opaque pixels only.

    (2G - R - B) / 255, the standard vegetation index for ordinary RGB imagery: it
    is positive for anything whose green channel dominates and strongly negative
    for sand, bare rock and dry earth, without needing a model or a colour-space
    conversion. Averaged per patch first so one large patch can't outvote the rest.

    The alpha channel is the region cutout perspective_crop applied (see
    _exemplar_patches), so transparent pixels are outside the vegetation region
    entirely and would otherwise drag every patch toward the padding colour.
    """
    scores: list[float] = []
    for patch in patches:
        arr = np.asarray(patch.convert("RGBA"), dtype=np.float32)
        rgb, alpha = arr[..., :3], arr[..., 3]
        opaque = alpha > 8.0
        if not opaque.any():
            continue
        r, g, b = rgb[..., 0][opaque], rgb[..., 1][opaque], rgb[..., 2][opaque]
        scores.append(float(np.mean((2.0 * g - r - b) / 255.0)))
    return float(np.mean(scores)) if scores else None


class GrassCoverConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        max_radius_m: float = 25.0,
        min_observed_grass_fraction: float = 0.15,
        min_cell_greenness: float = 0.02,
        full_density_greenness: float = 0.15,
        density_floor_fraction: float = 0.15,
        min_exemplar_greenness: float = 0.05,
        instance_spacing_m: float = 0.4,
        max_instances: int = 8000,
        bucket_count: int = 3,
        exemplar_candidates: int = 12,
        exemplar_patch_size: int = 384,
        card_planes: int = 3,
        tuft_height_m: float = 0.35,
        tuft_height_jitter: float = 0.3,
        generator_type: str = "SAM3D",
        build_near_meshes: bool = True,
        max_near_mesh_faces: int = 4000,
        repair_near_meshes: bool = False,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.max_radius_m = float(max_radius_m)
        # Two gates on whether this scene should have grass AT ALL. Both exist
        # because everything upstream of them answers a narrower question -- "is
        # this cell's region type in _GRASS_SOURCE_TYPES" -- and that question has
        # an affirmative answer in scenes with no grass anywhere in them. Measured
        # across five captures, every one grew grass, including a photograph of the
        # Seine from a bridge.
        #
        # WHERE the evidence came from. Only ~1/6 of the panorama is the original
        # photograph; the rest is generated, and the generator invents plausible
        # ground. Region typing then reads that invention as confidently as it reads
        # the real pixels, so a scene can be carpeted from evidence that was never
        # observed. This requires grass to be a real share of the ground actually
        # visible in the photograph, measured over the front hemisphere alone (see
        # grass_area._fill_hemisphere_stats, which already computed exactly this and
        # had no consumer). Front typed-grass as a fraction of front depth-sampled
        # ground, over those five captures:
        #
        #     Mount Rainier (alpine meadow)   94%
        #     Irises (a field of irises)      40%
        #     Iceland (grassy hillside)        5%
        #     Shark Fin Cove (a beach)         2%
        #     Paris (a river)                  1%
        #
        # 0.15 sits in the gap. Iceland is the interesting one: it DOES have a
        # grassy hillside, filling half the frame -- but the segmenter types it
        # `mountain`, which _GRASS_SOURCE_TYPES excludes on purpose (that exclusion
        # is what keeps grass off snowfields), so its real grass was never eligible
        # and the 1749 tufts it placed were 88% behind the camera on invented
        # ground. Rejecting it is correct as things stand; making it PASS is a
        # region-typing problem, not a threshold problem.
        self.min_observed_grass_fraction = float(min_observed_grass_fraction)
        # WHAT the evidence looks like. The exemplar patches are cut from the real
        # panorama pixels inside the vegetation region, and they are what the cards
        # get textured with -- so if they don't look like plants, nothing downstream
        # will either. Excess green ((2G - R - B) / 255) over the opaque pixels of
        # the candidate patches, which is a cheap deterministic test needing no
        # model. Same five captures:
        #
        #     Paris          +0.316      Iceland         +0.127
        #     Irises         +0.165      Shark Fin Cove  -0.123
        #     Mount Rainier  +0.148
        #
        # Shark Fin Cove is the case this catches: a "vegetation" region the
        # segmenter painted onto a generated cliff face, whose real pixels are
        # orange sand, textured 1434 grass cards with beach.
        #
        # Deliberately a veto on "definitely not a plant" rather than a test for
        # "is a plant" -- dry or golden grass is legitimately low on this scale, so
        # the threshold sits far below every green case rather than between them.
        self.min_exemplar_greenness = float(min_exemplar_greenness)
        # Per-CELL colour veto, the finer-grained companion to the whole-scene gate
        # above. RegionType.GROUND collapses `grass`/`earth`/`field` together with
        # `sand`, `snow` and `ice` (see panorama_region_result._LABEL_RULES) and the
        # fine ADE20K label is not persisted anywhere this stage can read, so type
        # alone cannot keep grass off the snowpack ringing an alpine meadow or off a
        # beach inside an otherwise-grassy scene. This asks the panorama what colour
        # each cell actually is. Set just above neutral rather than at anything lush:
        # dry/golden grass sits near zero, sand runs about -0.12 and snow lower, so
        # it removes only what is definitively not vegetation. -inf disables it.
        self.min_cell_greenness = float(min_cell_greenness)
        # Greenness at which a cell earns the FULL instance_spacing_m density, with a
        # linear ramp down to density_floor_fraction at min_cell_greenness above.
        # 0.15 is where the Rainier meadow's own thick vegetation sits -- only 6% of
        # its gate-passing cells reach it, and those are the parts of the photograph
        # that genuinely read as dense grass. Set to 0 (or below min_cell_greenness)
        # to restore uniform density everywhere the veto passes.
        self.full_density_greenness = float(full_density_greenness)
        # Density floor for a cell that only just clears the veto, as a fraction of
        # full. Not zero: those cells are still vegetation by every test applied, and
        # emptying them entirely would carve visible bald patches at the exact
        # boundaries the colour measurement is least certain about.
        self.density_floor_fraction = float(density_floor_fraction)
        # Nominal centre-to-centre spacing of the scatter lattice, before jitter.
        # 0.4 m over the ~886 m2 of grass the Rainier capture yields inside 25 m
        # is ~5500 instances, which is what max_instances is scaled against.
        self.instance_spacing_m = float(instance_spacing_m)
        # Hard cap on placed instances. Each one is its own Object3D in scene.json
        # and its own GameObject on the client -- there is no instanced-renderer
        # path yet -- so this is the budget that actually matters. Exceeding it
        # widens the spacing rather than truncating the list, so the thinning is
        # uniform instead of lopping off whichever corner got generated last.
        self.max_instances = int(max_instances)
        # Distinct visual variants of grass to build assets for. Deliberately
        # small: unlike ObjectCategoryClusteringStage's max_buckets_per_class=8,
        # grass reads as one material and the variation that matters comes from
        # per-instance yaw and scale, not from having many different tufts.
        self.bucket_count = int(bucket_count)
        self.exemplar_candidates = int(exemplar_candidates)
        self.exemplar_patch_size = int(exemplar_patch_size)
        self.card_planes = int(card_planes)
        self.tuft_height_m = float(tuft_height_m)
        self.tuft_height_jitter = float(tuft_height_jitter)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        # Near-LOD reconstructed meshes are the expensive half of this stage
        # (one 3D generation per bucket). Turning them off leaves the crossed
        # cards covering every distance, which still animates and still reads as
        # grass -- useful for iterating on coverage without paying for SAM3D.
        self.build_near_meshes = bool(build_near_meshes)
        # Face budget for the reconstructed near-LOD tuft. SAM3D's raw output is not
        # decimated at all: measured on the Rainier capture the three tufts came back at
        # 316k, 301k and 208k faces, ~8 MB of GLB each. That is hero-object geometry for
        # an asset that is one of thousands of instances of ground cover -- it made the
        # client download 25 GB and the scene 1.8 billion triangles.
        #
        # A tuft is a handful of blades; its silhouette is what reads, and that survives
        # aggressive decimation. 4000 keeps the blade shapes at arm's length (the near
        # LOD only renders within Scene Generation's grass mesh_lod_distance override,
        # ~3 m) while cutting the asset ~70x. 0 disables decimation entirely.
        self.max_near_mesh_faces = int(max_near_mesh_faces)
        # Whether to run Mesh.repair() on the reconstructed tuft. Off, unlike every
        # other consumer of that call, and this is the single biggest thing standing
        # between the near LOD and something that reads as grass.
        #
        # repair() exists to make a hero object watertight: it samples 50k points off
        # the surface, runs Poisson reconstruction at depth 8, then closes whatever
        # that leaves with MeshFix. Every one of those steps is a machine for
        # destroying exactly what a grass tuft IS. Poisson fits a single closed
        # isosurface, so thin separated blades -- a couple of voxels wide in its 256^3
        # grid -- get swallowed into one shell; MeshFix then guarantees that shell is
        # a single watertight solid. The result is a green pillow, which is what makes
        # near tufts read as slabs while the crossed cards behind them read as grass.
        # It also discards the texture entirely (repair() returns baked per-vertex
        # colour only).
        #
        # Nothing downstream needs the mesh to be manifold: it is decimated (open3d's
        # quadric decimation takes triangle soup, and decimate() already carries a
        # guard for "highly disconnected input like separate blades of grass"), rigged
        # by CategoryMeshRiggingStage (which only needs a Y extent), and rendered.
        # SAM3D's raw output does have broken faces, but a hole in a blade at 3 m is
        # invisible in a way that a sealed blob is not.
        self.repair_near_meshes = bool(repair_near_meshes)


class GrassCoverStage(PipelineStage):
    """
    Builds a grassland: derives where grass belongs, generates a small set of
    grass assets from the panorama's own pixels, and scatters instances of them
    across that area.

    Three things make this a separate stage rather than more configuration on
    ObjectDistributionStage/DistributionSynthesisStage, which already scatter
    populations:

      1. Those learn a spatial pattern from detected exemplars. Grass has no
         detected exemplars -- it is segmented as a *region*, not found as
         countable objects -- so there is no point set to fit a PCF to. Coverage
         here is a property of the region, not of a learned arrangement.
      2. They read the region map, which is derived from the object-removed
         panorama and therefore has the meadow classified as sand. See
         grass_area.py for the measurement; this stage sources its area from the
         ORIGINAL panorama's region typing instead.
      3. Grass needs a fixed-orientation crossed-card far LOD (see cards.py)
         rather than SceneGenerationStage's camera-facing billboard fallback.

    Everything downstream is reused as-is. Instances are written as ordinary
    `synthetic` metadata entries of class GRASS_TUFT_CATEGORY, so:
      - SceneGenerationStage places them, snaps them to the terrain, and gives
        each a random yaw, through the same branch it already uses for painted
        distribution points;
      - ObjectMotionClassificationStage marks them stationary (its `synthetic`
        branch) without trying to track any of them in the generated video;
      - CategoryMeshRiggingStage bakes its 3-bone sway skeleton into both LOD
        meshes, since GRASS_TUFT_CATEGORY is in VEGETATION_CATEGORIES;
      - SceneAnimationStage attaches per-instance sway timing.

    Reads:  ContextKey.PANORAMA (the ORIGINAL panorama -- the meadow is still in
            it), ContextKey.PANORAMA_REGION_TYPE_MAP (its region typing),
            ContextKey.HEIGHT_MAP_PANO_U / _V, HEIGHT_MAP_REAL_SAMPLE_MASK,
            HEIGHT_MAP_PARAMS, ContextKey.OBJECT_COUNT
    Writes: category_mesh_{grass_tuft}_{bucket}       (near LOD, reconstructed)
            category_mesh_{grass_tuft}_{bucket}_card  (far LOD, crossed cards)
            metadata_{idx} per scattered instance, ContextKey.OBJECT_COUNT (bumped)
    Debug:  grass_cover.json, grass_area.png, exemplar_{bucket}.png
    """

    @classmethod
    def config_class(cls) -> type[GrassCoverConfiguration]:
        return GrassCoverConfiguration

    def __init__(self, config: GrassCoverConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)
        self._embedder: Optional[DinoV2Embedder] = None

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: GrassCoverConfiguration = self.config

        inputs = self._gather(context)
        if inputs is None:
            context.add_object(_RAN_MARKER, True)
            return context
        panorama, type_map, pano_u, pano_v, sampled, grid_size_meters = inputs

        area_stats: dict = {}
        mask = grass_area_mask(
            pano_u, pano_v, type_map, sampled,
            max_radius_m=cfg.max_radius_m,
            grid_size_meters=grid_size_meters,
            panorama_rgb=np.asarray(panorama.rgb()) if panorama is not None else None,
            min_cell_greenness=cfg.min_cell_greenness,
            stats=area_stats,
        )
        area_m2 = area_square_meters(mask, grid_size_meters)
        self.log_info(
            f"Grass area: {area_m2:.0f} m2 within {cfg.max_radius_m:.0f} m "
            f"({area_m2 / (np.pi * cfg.max_radius_m ** 2):.0%} of the disc)"
        )
        # Front/behind funnel. Only ~1/6 of the panorama is the original photograph
        # and the rest is generated, so grass concentrating in front of the camera is
        # expected to some degree -- but nothing in grass_area_mask is azimuth-
        # dependent, so if it happens the cause is one of these two inputs being
        # weaker behind, and this says which. See _fill_hemisphere_stats.
        for where in ("front", "behind"):
            s = area_stats.get(where)
            if s:
                self.log_info(
                    f"  {where:>6}: {s['in_range_m2']:.0f} m2 in range -> "
                    f"{s['sampled_m2']:.0f} m2 depth-sampled -> "
                    f"{s['grass_typed_m2']:.0f} m2 typed grass -> "
                    f"{s['final_m2']:.0f} m2 final"
                )
        if not mask.any():
            self.log_info("No grass area found, skipping")
            self._write_debug(context, mask, grid_size_meters, [], 0, area_m2)
            context.add_object(_RAN_MARKER, True)
            return context

        # Was any of this actually SEEN? See min_observed_grass_fraction. Checked
        # before the exemplar/mesh work below, which is the expensive half.
        observed = _observed_grass_fraction(area_stats)
        if observed is not None and observed < cfg.min_observed_grass_fraction:
            self.log_info(
                f"Only {observed:.1%} of the ground visible in the original photograph "
                f"types as grass (need {cfg.min_observed_grass_fraction:.0%}) — "
                f"the {area_m2:.0f} m2 found is evidence from the generated panorama, "
                f"not from the capture; skipping grass"
            )
            self._write_debug(context, mask, grid_size_meters, [], 0, area_m2)
            context.add_object(_RAN_MARKER, True)
            return context

        task = self.create_progress(4, "Building grass cover…")

        exemplars = self._exemplar_patches(panorama, type_map, cfg)
        self.advance_progress(task)
        if not exemplars:
            self.log_info("No usable grass exemplar patches in the panorama, skipping")
            self.finish_progress(task)
            self._write_debug(context, mask, grid_size_meters, [], 0, area_m2)
            context.add_object(_RAN_MARKER, True)
            return context

        # Do those patches look like plants? See min_exemplar_greenness.
        greenness = _exemplar_greenness(exemplars)
        if greenness is not None and greenness < cfg.min_exemplar_greenness:
            self.log_info(
                f"Grass exemplars score {greenness:+.3f} excess-green "
                f"(need {cfg.min_exemplar_greenness:+.3f}) — the region typed as "
                f"vegetation isn't green, so its pixels would texture the cards with "
                f"whatever it actually is; skipping grass"
            )
            self.finish_progress(task)
            self._write_debug(context, mask, grid_size_meters, [], 0, area_m2)
            context.add_object(_RAN_MARKER, True)
            return context

        representatives = self._bucket_exemplars(exemplars, cfg)
        self.log_info(f"Grass variants: {len(representatives)} bucket(s) from {len(exemplars)} candidate patch(es)")
        self.advance_progress(task)

        self._build_card_meshes(context, representatives, cfg)
        if cfg.build_near_meshes:
            self._build_near_meshes(context, representatives, cfg)
        self.advance_progress(task)

        tuft_height = self._tuft_height(cfg)
        placed = self._scatter(
            context, mask, grid_size_meters, len(representatives), tuft_height, cfg,
            greenness=area_stats.get("greenness"),
        )
        self.advance_progress(task)
        self.finish_progress(task)

        self._write_debug(context, mask, grid_size_meters, representatives, placed, area_m2)
        context.add_object(_RAN_MARKER, True)
        return context

    # ── Inputs ────────────────────────────────────────────────────────────────

    def _gather(self, context: PipelineContext):
        panorama = context.input_panorama(ContextKey.PANORAMA)
        type_map_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        pano_u_depth = context.input_depth(ContextKey.HEIGHT_MAP_PANO_U)
        pano_v_depth = context.input_depth(ContextKey.HEIGHT_MAP_PANO_V)
        sampled_depth = context.input_depth(ContextKey.HEIGHT_MAP_REAL_SAMPLE_MASK)

        missing = [
            name for name, value in [
                ("PANORAMA", panorama),
                ("PANORAMA_REGION_TYPE_MAP", type_map_depth),
                ("HEIGHT_MAP_PANO_U", pano_u_depth),
                ("HEIGHT_MAP_PANO_V", pano_v_depth),
                ("HEIGHT_MAP_REAL_SAMPLE_MASK", sampled_depth),
            ] if value is None
        ]
        if missing:
            self.log_info(f"Missing {', '.join(missing)} — skipping grass cover")
            return None

        grid_size_meters = (context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get(
            "grid_size_meters", 100.0
        )
        return (
            panorama, type_map_depth.depth, pano_u_depth.depth, pano_v_depth.depth,
            sampled_depth.depth, grid_size_meters,
        )

    # ── Assets ────────────────────────────────────────────────────────────────

    def _exemplar_patches(
        self, panorama, type_map: np.ndarray, cfg: GrassCoverConfiguration,
    ) -> list[PIL.Image.Image]:
        """RGBA cutouts of real grass, cropped upright from the original panorama.

        Deliberately Panorama.perspective_crop rather than the plane-fitting
        Panorama.unwarp_box that TerrainTextureGenerationStage's own reference
        patches use: that one flattens a patch onto the ground plane, which is
        exactly right for a tileable ground material and exactly wrong here. A
        grass card needs the side-on view the camera actually photographed, with
        blades standing up in frame; flattened to top-down it becomes a green
        smear with no silhouette to alpha-cut against.

        The region mask travels through as the alpha channel (perspective_crop's
        own `mask` argument), which is what gives the cards their cutout.

        Only rows below the horizon are considered. Vegetation above it is the
        conifer band at the treeline, which ADE20K labels "plant" identically to
        the meadow -- sampling there would build grass cards out of tree crowns.
        """
        # Boxes are chosen in type-map pixel space and then handed to
        # perspective_crop, which works in the panorama's own pixel space, so the
        # two have to agree. They normally do -- PanoramaRegionStage types the
        # same image -- but SupersamplingStage rewrites PANORAMA in place at 2x,
        # so a run where region typing happened against the pre-supersampled
        # panorama silently halves every coordinate. That failure is invisible
        # rather than loud: the mask still reprojects, it just lands somewhere
        # else, and every patch comes back fully transparent (observed exactly
        # this way against a 2048x1024 panorama and a 4096x2048 type map).
        if type_map.shape != (panorama.height, panorama.width):
            self.log_info(
                f"Region type map {type_map.shape[1]}x{type_map.shape[0]} != panorama "
                f"{panorama.width}x{panorama.height}, resampling to match"
            )
            type_map = np.array(PIL.Image.fromarray(type_map.astype(np.uint8)).resize(
                (panorama.width, panorama.height), PIL.Image.NEAREST,
            ))

        pano_h, pano_w = type_map.shape
        region_mask = (type_map == int(RegionType.VEGETATION)).astype(np.float32)

        remaining = region_mask > 0.5
        # Horizon is row pano_h/2 in an equirectangular image; everything the
        # camera looks *down* at is below it.
        remaining[: pano_h // 2, :] = False
        # The bottom rows converge on the nadir point, where equirectangular
        # stretching is extreme and a square box spans almost no real ground.
        remaining[int(pano_h * 0.95):, :] = False

        patches: list[PIL.Image.Image] = []
        for _ in range(max(1, cfg.exemplar_candidates)):
            if not remaining.any():
                break
            dist = distance_transform_edt(remaining)
            peak = float(dist.max())
            if peak < 8.0:
                break
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            side = min(max(32, int(2 * peak)), pano_h, pano_w)
            half = side // 2
            row0 = int(np.clip(cy - half, 0, pano_h - side))
            col0 = int(np.clip(cx - half, 0, pano_w - side))

            crop = panorama.perspective_crop([col0, row0, side, side], mask=region_mask)
            patches.append(
                crop.resize((cfg.exemplar_patch_size, cfg.exemplar_patch_size), PIL.Image.LANCZOS)
            )

            yy, xx = np.ogrid[:pano_h, :pano_w]
            remaining &= (yy - cy) ** 2 + (xx - cx) ** 2 > peak ** 2

        return patches

    def _bucket_exemplars(
        self, patches: list[PIL.Image.Image], cfg: GrassCoverConfiguration,
    ) -> list[PIL.Image.Image]:
        """Reduce the candidate patches to at most bucket_count visual variants.

        Same DINOv2-embedding + average-linkage agglomerative clustering
        ObjectCategoryClusteringStage uses to split a class into visual variants,
        cut by `maxclust` at bucket_count directly (rather than by a cosine
        distance threshold): the target here is a fixed, small number of variants,
        not "however many genuinely distinct ones exist".

        Each bucket's representative is the patch closest to its own centroid --
        the most typical member, not an outlier that happened to anchor it.
        """
        wanted = max(1, cfg.bucket_count)
        if len(patches) <= wanted:
            return patches

        if self._embedder is None:
            self._embedder = DinoV2Embedder(self.device)
        embeddings = np.stack([self._embedder.embed(Image(p)) for p in patches])

        labels = fcluster(
            linkage(pdist(embeddings, metric="cosine"), method="average"),
            t=wanted, criterion="maxclust",
        )

        representatives: list[PIL.Image.Image] = []
        for label in sorted(set(labels.tolist())):
            members = np.flatnonzero(labels == label)
            centroid = embeddings[members].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            representatives.append(patches[int(members[np.argmax(embeddings[members] @ centroid)])])
        return representatives

    def _build_card_meshes(
        self, context: PipelineContext, representatives: list[PIL.Image.Image], cfg: GrassCoverConfiguration,
    ) -> None:
        for bucket, patch in enumerate(representatives):
            key = _card_mesh_key(bucket)
            # The patch's own alpha is a semantic region mask, near-solid inside a
            # meadow -- see apply_tuft_silhouette for why a card needs a real
            # silhouette carved into it before it reads as grass.
            shaped = apply_tuft_silhouette(patch, np.random.default_rng((self.seed, bucket)))
            mesh = crossed_card_mesh(shaped, plane_count=cfg.card_planes)
            context.add_mesh(key, mesh)
            self.log_info(f"  {key}: {mesh.vertex_count}v {mesh.face_count}f crossed cards")

    def _build_near_meshes(
        self, context: PipelineContext, representatives: list[PIL.Image.Image], cfg: GrassCoverConfiguration,
    ) -> None:
        """One reconstructed mesh per bucket, for instances inside the mesh LOD distance.

        Failures are per-bucket and non-fatal: SceneGenerationStage falls back to
        this bucket's crossed cards when category_mesh_{grass_tuft}_{bucket} is
        absent, exactly as it falls back to a billboard pool for any other class
        whose mesh generation didn't produce one.
        """
        pending = [b for b in range(len(representatives)) if context.mesh(_mesh_key(b)) is None]
        if not pending:
            self.log_info("  near-LOD grass meshes: all cached")
            return

        super().clean_up()
        generator = ModelGenerator(self.preferred_device, type=cfg.generator_type)
        try:
            for bucket in pending:
                key = _mesh_key(bucket)
                temp_path = self.temp / key if self.temp is not None else None
                if temp_path is not None:
                    temp_path.mkdir(parents=True, exist_ok=True)
                super().clean_up()
                try:
                    mesh = generator.meshify(Image(representatives[bucket]), temp_path, seed=self.seed + bucket)
                    if cfg.repair_near_meshes:
                        mesh = mesh.repair()
                    raw_faces = mesh.face_count
                    # Decimate after any repair (which needs the dense surface to fit a
                    # clean watertight mesh) and before fit_to_box (pure scale,
                    # order-independent).
                    mesh = mesh.decimate(cfg.max_near_mesh_faces)
                    mesh.fit_to_box(1.0, 1.0)
                except Exception as e:
                    self.log_info(f"  {key}: meshify failed ({e}), falling back to crossed cards at all distances")
                    continue

                # decimate() returns the input UNCHANGED when quadric decimation
                # collapses the mesh entirely, which its own comment notes is a real
                # possibility "on a highly disconnected input like separate blades of
                # grass" -- i.e. exactly what this path now feeds it, since the
                # Poisson repair that used to hand it one welded shell is off by
                # default (see repair_near_meshes). Shipping that fallback here would
                # mean instancing SAM3D's raw ~275k-face output thousands of times,
                # which is the billion-triangle scene max_near_mesh_faces exists to
                # prevent. The crossed cards are a far better asset than a budget
                # blown by that factor.
                if 0 < cfg.max_near_mesh_faces < mesh.face_count:
                    self.log_warning(
                        f"  {key}: decimation could not reach {cfg.max_near_mesh_faces}f "
                        f"(still {mesh.face_count}f from {raw_faces}f) — dropping the near "
                        f"LOD and using crossed cards at all distances for this bucket"
                    )
                    continue

                context.add_mesh(key, mesh)
                self.log_info(
                    f"  {key}: {mesh.vertex_count}v {mesh.face_count}f reconstructed "
                    f"(decimated from {raw_faces}f"
                    + (", Poisson-repaired" if cfg.repair_near_meshes else "") + ")"
                )
        finally:
            generator.close()

    def _tuft_height(self, cfg: GrassCoverConfiguration) -> float:
        """Real-world height of one tuft, in metres: the configured nominal.

        Deliberately not derived from the imagery. The obvious-looking derivation
        -- take an exemplar patch's angular height and multiply by its depth --
        measures the wrong thing: patch extent is chosen by the distance
        transform in _exemplar_patches for how *solid* the vegetation region is
        there, so it reports the size of a contiguous blob of meadow, not the
        height of a blade within it. Nothing upstream segments individual tufts,
        so there is no measurement of this quantity anywhere in the pipeline.

        An earlier version did fit a number out of the near-ground depth band
        against a hand-picked "a tuft is ~4% of the band" constant. It clamped to
        its own floor on the capture it was written against, which is the tell: it
        was contributing a guess dressed as a measurement. A named constant the
        caller can tune is the same guess, honestly labelled.
        """
        return cfg.tuft_height_m

    # ── Scatter ───────────────────────────────────────────────────────────────

    def _scatter(
        self,
        context: PipelineContext,
        mask: np.ndarray,
        grid_size_meters: float,
        bucket_count: int,
        tuft_height: float,
        cfg: GrassCoverConfiguration,
        greenness: "np.ndarray | None" = None,
    ) -> int:
        """Place instances on a jittered lattice over the grass mask.

        A jittered lattice rather than a learned PCF (DistributionSynthesisStage)
        or plain uniform sampling: grass has no interesting spatial statistics to
        reproduce -- it just needs to be everywhere without visible rows or the
        clumping-and-holes that independent uniform draws produce at this density.
        Jitter of one half-cell breaks the lattice up without letting neighbours
        collide.

        Spacing widens to respect max_instances rather than the list being
        truncated, so thinning is uniform across the whole area.
        """
        rng = np.random.default_rng(self.seed)
        resolution = mask.shape[0]
        half = grid_size_meters / 2.0

        spacing = cfg.instance_spacing_m
        area_m2 = area_square_meters(mask, grid_size_meters)
        if cfg.max_instances > 0:
            # Instances scale as area / spacing^2, so the spacing that just fits
            # the budget is spacing * sqrt(estimated / budget).
            estimated = area_m2 / (spacing * spacing)
            if estimated > cfg.max_instances:
                spacing *= float(np.sqrt(estimated / cfg.max_instances))
                self.log_info(
                    f"  spacing widened {cfg.instance_spacing_m:.2f} m -> {spacing:.2f} m "
                    f"to fit max_instances={cfg.max_instances}"
                )

        steps = max(1, int(np.floor(grid_size_meters / spacing)))
        lattice = (np.arange(steps, dtype=np.float64) + 0.5) * spacing - half
        xs, zs = np.meshgrid(lattice, lattice, indexing="ij")
        xs = xs.ravel()
        zs = zs.ravel()

        jitter = spacing * 0.5
        xs = xs + rng.uniform(-jitter, jitter, xs.shape)
        zs = zs + rng.uniform(-jitter, jitter, zs.shape)

        cols = ((xs + half) / grid_size_meters * resolution).astype(np.intp)
        rows = ((zs + half) / grid_size_meters * resolution).astype(np.intp)
        inside = (rows >= 0) & (rows < resolution) & (cols >= 0) & (cols < resolution)
        xs, zs, rows, cols = xs[inside], zs[inside], rows[inside], cols[inside]

        keep = mask[rows, cols]
        xs, zs, rows, cols = xs[keep], zs[keep], rows[keep], cols[keep]

        # Density follows how much vegetation the panorama actually shows at each
        # cell, instead of being uniform everywhere the threshold passed.
        #
        # min_cell_greenness is a veto -- "is this cell vegetation at all" -- and the
        # scatter then treated everything above it identically. Measured on the
        # Rainier capture, of the cells that pass, 27% sit in 0.02-0.05 (barely
        # distinguishable from bare ground) and only 6% clear 0.15; median 0.075. A
        # melting snow margin, a dry patch and the thick of the meadow all received
        # the same 15 tufts per square metre, which is what makes ground cover read as
        # a carpet laid over the terrain rather than as the meadow in the photograph.
        #
        # The same measurement answers the density question, so this is a weight, not
        # a second threshold: keep-probability ramps linearly from
        # density_floor_fraction at the veto to 1.0 at full_density_greenness. Applied
        # by rejection against the existing lattice, so the spacing/budget logic above
        # is untouched and the result is still a jittered lattice, just thinned where
        # the ground is not green. 0 for full_density_greenness restores the old
        # uniform behaviour.
        weighted = 0
        if greenness is not None and cfg.full_density_greenness > cfg.min_cell_greenness:
            g = np.asarray(greenness)[rows, cols]
            span = cfg.full_density_greenness - cfg.min_cell_greenness
            weight = (g - cfg.min_cell_greenness) / span
            weight = np.clip(weight, cfg.density_floor_fraction, 1.0)
            survives = rng.random(xs.size) < weight
            weighted = int(xs.size - survives.sum())
            xs, zs = xs[survives], zs[survives]
        if xs.size == 0:
            self.log_info("  no lattice points landed on grass, nothing placed")
            return 0

        buckets = rng.integers(0, max(1, bucket_count), size=xs.size)
        heights = tuft_height * (
            1.0 + rng.uniform(-cfg.tuft_height_jitter, cfg.tuft_height_jitter, xs.size)
        )

        next_idx = context.input_object(ContextKey.OBJECT_COUNT) or 0
        for x, z, bucket, height in zip(xs, zs, buckets, heights):
            context.add_object(f"metadata_{next_idx}", {
                "class": GRASS_TUFT_CATEGORY,
                "bucket": int(bucket),
                "synthetic": True,
                # Y is a placeholder -- SceneGenerationStage replaces it with the
                # terrain raycast, same as for any other synthetic point.
                "world_position": [float(x), 0.0, float(z)],
                "world_width": float(height),
                "world_height": float(height),
                # Already real-world metric (see _tuft_height -- a configured
                # constant, deliberately not derived from imagery), so it never
                # inherited the depth map's compression and must be exempt from
                # SceneGenerationStage's object scale correction. Without this the
                # correction treats 0.35 m ground cover like a compressed detection
                # and inflates it by the scene's far-field factor.
                "metric_size": True,
            })
            next_idx += 1

        context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        self.log_info(
            f"  placed {xs.size} grass instance(s) at {spacing:.2f} m spacing "
            f"({xs.size / area_m2:.1f}/m2 average), tuft height {tuft_height:.2f} m"
            + (f" — {weighted} thinned out where the ground is less green" if weighted else "")
        )
        return int(xs.size)

    # ── Debug / plumbing ──────────────────────────────────────────────────────

    def _write_debug(
        self,
        context: PipelineContext,
        mask: np.ndarray,
        grid_size_meters: float,
        representatives: list[PIL.Image.Image],
        placed: int,
        area_m2: float,
    ) -> None:
        summary = {
            "grass_area_m2": round(area_m2, 1),
            "max_radius_m": self.config.max_radius_m,
            "buckets": len(representatives),
            "instances_placed": placed,
        }
        context.add_object(_SUMMARY_KEY, summary)

        if self.output is None:
            return
        with open(self.output / "grass_cover.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self.temp is not None:
            PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(self.temp / "grass_area.png")
            for bucket, patch in enumerate(representatives):
                patch.save(self.temp / f"exemplar_{bucket}.png")

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(_RAN_MARKER) is True

    def model_names(self) -> list[str]:
        names = list(DinoV2Embedder.model_names())
        if self.config.build_near_meshes:
            names += ModelGenerator.model_names(self.config.generator_type)
        return names

    def clean_up(self):
        self._embedder = None
        super().clean_up()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        summary = context.object(_SUMMARY_KEY)
        if summary is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Grass Cover",
            body=(
                "Ground cover was scattered across the area the ORIGINAL panorama's "
                "region typing marks as vegetation -- not the object-removed one, "
                "whose foreground inpainting replaces the near-field meadow with bare "
                "ground. Each instance renders a reconstructed clump close to the "
                "camera and a fixed-orientation crossed-card variant beyond it, both "
                "rigged for wind sway."
            ),
            stats={
                "Grass area": f"{summary['grass_area_m2']:.0f} m²",
                "Variants": str(summary["buckets"]),
                "Instances placed": str(summary["instances_placed"]),
            },
        )
