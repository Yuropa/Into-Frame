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
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.max_radius_m = float(max_radius_m)
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

        mask = grass_area_mask(
            pano_u, pano_v, type_map, sampled,
            max_radius_m=cfg.max_radius_m,
            grid_size_meters=grid_size_meters,
        )
        area_m2 = area_square_meters(mask, grid_size_meters)
        self.log_info(
            f"Grass area: {area_m2:.0f} m2 within {cfg.max_radius_m:.0f} m "
            f"({area_m2 / (np.pi * cfg.max_radius_m ** 2):.0%} of the disc)"
        )
        if not mask.any():
            self.log_info("No grass area found, skipping")
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

        representatives = self._bucket_exemplars(exemplars, cfg)
        self.log_info(f"Grass variants: {len(representatives)} bucket(s) from {len(exemplars)} candidate patch(es)")
        self.advance_progress(task)

        self._build_card_meshes(context, representatives, cfg)
        if cfg.build_near_meshes:
            self._build_near_meshes(context, representatives, cfg)
        self.advance_progress(task)

        tuft_height = self._tuft_height(cfg)
        placed = self._scatter(context, mask, grid_size_meters, len(representatives), tuft_height, cfg)
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
                    mesh = mesh.repair()
                    raw_faces = mesh.face_count
                    # Decimate after repair (which needs the dense surface to fit a clean
                    # watertight mesh) and before fit_to_box (pure scale, order-independent).
                    mesh = mesh.decimate(cfg.max_near_mesh_faces)
                    mesh.fit_to_box(1.0, 1.0)
                except Exception as e:
                    self.log_info(f"  {key}: meshify failed ({e}), falling back to crossed cards at all distances")
                    continue
                context.add_mesh(key, mesh)
                self.log_info(
                    f"  {key}: {mesh.vertex_count}v {mesh.face_count}f reconstructed "
                    f"(decimated from {raw_faces}f)"
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
        xs, zs = xs[keep], zs[keep]
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
            })
            next_idx += 1

        context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        self.log_info(
            f"  placed {xs.size} grass instance(s) at {spacing:.2f} m spacing "
            f"({xs.size / area_m2:.1f}/m2), tuft height {tuft_height:.2f} m"
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
