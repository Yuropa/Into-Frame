from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import torch
from scipy.ndimage import binary_dilation, gaussian_filter, label as ndi_label, zoom
from scipy.ndimage import map_coordinates

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama.panorama_lora import PanoramaLoraType, lora_prompt_prefix, lora_prompt_suffix
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.terrain.terrain_generator import TerrainMeshGenerator
from scene.splat_material import SplatLayer, SplatMaterial
from util.image_utils import Image, lab_color_transfer
from util.device_utils import DeviceStrategy, preferred_device


_GROUND_TYPES: frozenset[RegionType] = frozenset({
    RegionType.GROUND,
    RegionType.TERRAIN,
    RegionType.VEGETATION,
    RegionType.ROAD,
})

# Short on purpose: at FLUX Fill's guidance_scale (see TerrainTextureGenerationConfiguration)
# the text prompt strongly dominates the img2img seed, so an elaborate, hyper-specific
# description would fight the actual photo reference instead of following it. These name
# the material and its 1-2 defining traits only — the real reference crop
# (_extract_largest_region_crop) and per-region LAB colour transfer carry the rest.
_BASE_PROMPTS: dict[RegionType, str] = {
    RegionType.GROUND: (
        "macro close-up photograph of loose soil grit, tiny sharp gravel stones, "
        "cracked dry mud crumbs, fine earthen detail grains"
    ),
    RegionType.TERRAIN: (
        "macro close-up photograph of sharp jagged broken rocks, angular stone fragments, "
        "slate striations and granite micro-crevices"
    ),
    RegionType.VEGETATION: (
        "overhead macro photograph of dense wild grass blades, green clover leaves, "
        "tangled lawn moss filaments, individual pine needles"
    ),
    RegionType.ROAD: (
        "macro close-up photograph of weathered coarse aggregate asphalt, embedded "
        "granite grit pebbles, sharp porous tarmac texture"
    ),
}

# PBR smoothness per region: 0 = perfectly rough/matte, 1 = mirror-smooth.
# Soil/grass/gravel are all near-zero (diffuse).
_LAYER_SMOOTHNESS: dict[RegionType, float] = {
    RegionType.GROUND:      0.05,
    RegionType.TERRAIN:     0.08,
    RegionType.VEGETATION:  0.04,
    RegionType.ROAD:        0.18,
}

_TILE_SUFFIX = (
    ", viewed from directly above at 30 centimetres, flat lay macro photography, "
    "filling the entire frame edge to edge, seamless tileable surface material, "
    "photorealistic PBR diffuse albedo texture, no directional shadows, "
    "soft overcast flat lighting, no depth of field, no horizon, no sky, no background, "
    "ultra-sharp high-frequency surface detail, extreme micro-texture visible, 8K quality"
)


class TerrainTextureGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 42,
        tile_size: int = 1024,
        blend_map_size: int = 512,
        blend_sigma: float = 0.05,
        min_region_fraction: float = 0.02,
        inpainting_type: str = "FLUX",
        num_inference_steps: int = 35,  # Higher headroom to resolve fine grains
        guidance_scale: float = 3.5,
        seam_width_fraction: float = 0.08,
        seam_dilation_px: int = 4,
        use_panorama_layer: bool = True,
        # Exponent applied to the latitude-ramp visibility weight below. 1.0 = the
        # plain ramp; higher sharpens the transition into the nadir/horizon fades.
        panorama_blend_power: float = 1.0,
        # Viewing-latitude cutoffs for the panorama layer's coverage (see
        # _panorama_visibility_weight). Wide by design: the real photo should
        # dominate almost the whole terrain, with synthetic tiles only filling the
        # narrow nadir hole under the camera and a thin band right at the horizon.
        nadir_cutoff_deg: float = -85.0,
        nadir_fade_deg: float = 4.0,
        horizon_fade_deg: float = 1.5,
        # 200 m terrain / 4 m per tile = 50 repeats → ~4 cm/texel at 1024 px
        synthetic_tile_factor: float = 50.0,
        use_photo_reference: bool = True,
        reference_tex_size: int = 2048,
        # The reference crop (_extract_largest_region_crop, a single contiguous, coherently-
        # lit patch of this region from the baked photo) is pasted at native guidance size
        # onto the tile canvas, centred, covering this fraction of the tile's linear size.
        # Everything outside it is masked as "generate" -- FLUX extends the real material
        # outward from genuine context via the pipeline's actual mask/masked-image
        # conditioning, rather than the whole canvas being a soft img2img seed. Kept well
        # under 1.0 so the patch has a comfortable margin before the tile edge, which the
        # seam-fix pass later regenerates anyway.
        reference_patch_fraction: float = 0.5,
        # Gaussian feather radius (px) on the mask boundary around the real patch, so the
        # generate/keep transition blends rather than leaving a hard rectangular seam.
        mask_feather_px: float = 12.0,
        # Diffusers `strength` for the masked ("generate") region. With a proper partial
        # mask the pipeline's own mask/masked-image conditioning is what preserves the real
        # patch -- not this value -- so 1.0 (full generation in the masked region) is the
        # semantically correct default; lower values would also pull noise into the real
        # patch along the feathered mask boundary.
        reference_strength: float = 1.0,
        # Post-process LAB colour nudge toward the reference crop's stats.
        color_transfer_strength: float = 0.60,
        # FLUX_SEAMLESS_TEXTURE LoRA strength. Lower than full (1.0) so its seamless-tiling
        # bias steers generation without overpowering the per-region material prompt or the
        # reference crop's genuine detail.
        lora_scale: float = 0.8,
        # Debug: regenerate one representative region's tile through the real reference-
        # seeded high-fidelity path (actual captured photo/region data, not a synthetic
        # test prompt) at several LoRA scales, so a reasonable lora_scale can be picked
        # against real data from a normal pipeline run. Written to self.temp/lora_sweep/;
        # never affects TERRAIN_MATERIAL or the tiles actually used. No-op unless
        # self.temp is set.
        debug_lora_scale_sweep: bool = False,
        debug_lora_scales: Optional[list[float]] = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.tile_size = tile_size
        self.blend_map_size = blend_map_size
        self.blend_sigma = blend_sigma
        self.min_region_fraction = min_region_fraction
        self.inpainting_type = InPaintingType[inpainting_type]
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seam_width_fraction = seam_width_fraction
        self.seam_dilation_px = seam_dilation_px
        self.use_panorama_layer = use_panorama_layer
        self.panorama_blend_power = panorama_blend_power
        self.nadir_cutoff_deg = nadir_cutoff_deg
        self.nadir_fade_deg = nadir_fade_deg
        self.horizon_fade_deg = horizon_fade_deg
        # UV tiling factor for synthetic region tiles (panorama layer always uses 1.0).
        # At 50× over a 200 m grid one tile covers 4 m → ~0.4 cm/texel at 1024 px.
        self.synthetic_tile_factor = synthetic_tile_factor
        self.use_photo_reference = use_photo_reference
        self.reference_tex_size = reference_tex_size
        self.reference_patch_fraction = reference_patch_fraction
        self.mask_feather_px = mask_feather_px
        self.reference_strength = reference_strength
        self.color_transfer_strength = color_transfer_strength
        self.lora_scale = lora_scale
        self.debug_lora_scale_sweep = debug_lora_scale_sweep
        self.debug_lora_scales = debug_lora_scales if debug_lora_scales is not None else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class TerrainTextureGenerationStage(PipelineStage):
    """
    Generates high-quality tileable region textures and packages them as a SplatMaterial.

    For each ground region type present in the REGION_MAP (above min_region_fraction):
      1. If a real photo is available (PANORAMA_TERRAIN + HEIGHT_MAP), bake it
         top-down and crop a single square from the largest connected component
         of that region in the baked photo (_extract_largest_region_crop). A
         single real crop is inherently coherent — one location, one lighting/
         exposure condition, genuine continuous texture — unlike stitching many
         small patches from potentially very different lighting conditions
         within the same region label, which produces a hard-seamed, visibly
         patchy mosaic.
      2. Paste that crop centred on the tile canvas at reference_patch_fraction
         of the tile's size and build a real partial mask: black (keep) over the
         patch, white (generate) everywhere else, feathered at the boundary
         (_build_reference_canvas). This is genuine inpainting — FLUX's own
         mask/masked-image conditioning extends the real material outward from
         true context, rather than the whole canvas being a soft img2img seed
         under a meaningless full-white mask. A region with no pixels at all
         falls back to plain text generation.
      3. Generate a photorealistic seamlessly tileable tile with FLUX (with the
         FLUX_SEAMLESS_TEXTURE LoRA, two-pass: reference-guided partial-mask
         generation → circular-shift seam inpainting — the patch sits with
         enough margin that the seam-fix band never touches it), then
         sharpen/contrast-boost to recover detail softened by inpainting and
         nudge colour statistics back toward the reference crop. A local
         micro-height map derived from the tile's own high-frequency detail
         is packed into its alpha channel (_pack_local_height_channel) so the terrain
         shader can do height-biased blending between layers instead of a flat
         linear cross-fade.

    SplatMaterial.from_region_map handles weight computation, normalisation, and
    RGBA blend map packing.  The first layer tile is also written to TERRAIN_TEXTURE
    so the terrain mesh can embed a preview in the GLB (with alpha stripped —
    see TerrainMeshGenerator.generate — since that channel holds height, not opacity).

    Reads:
      ContextKey.REGION_MAP           — top-down region type grid (optional)
      ContextKey.PANORAMA_TERRAIN     — real photo, source for the reference crop (optional)
      ContextKey.HEIGHT_MAP           — for top-down baking alignment (optional)
      ContextKey.HEIGHT_MAP_CERTAINTY — certainty mask for the bake (optional)
      ContextKey.INPUT_CAPTION        — scene caption for prompt context (optional)

    Writes:
      ContextKey.TERRAIN_MATERIAL — SplatMaterial (tiles + blend maps)
      ContextKey.TERRAIN_TEXTURE  — first layer tile (GLB preview only)
    """

    @classmethod
    def config_class(cls) -> type[TerrainTextureGenerationConfiguration]:
        return TerrainTextureGenerationConfiguration

    def __init__(self, config: TerrainTextureGenerationConfiguration) -> None:
        super().__init__(config)
        self._inpainter: Optional[InPainting] = None

    def _init_inpainter(self) -> None:
        if self._inpainter is None:
            cfg: TerrainTextureGenerationConfiguration = self.config
            inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
            self._inpainter = InPainting(
                inpaint_device, inpaint_dtype, cfg.inpainting_type,
                lora_type=PanoramaLoraType.FLUX_SEAMLESS_TEXTURE,
                lora_scale=cfg.lora_scale,
            )

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainTextureGenerationConfiguration = self.config
        caption = context.input_object(ContextKey.INPUT_CAPTION)

        # Determine which region types to generate tiles for
        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        present_types = self._present_region_types(
            region_map_depth, cfg.min_region_fraction,
        )
        self.log_info(f"Terrain regions: {[rt.label for rt in present_types]}")

        task = self.create_progress(len(present_types) * 2 + 1, "Terrain Texture Generation…")

        self._init_inpainter()

        reference = self._bake_reference(context, cfg) if cfg.use_photo_reference else None

        tiles: dict[str, PIL.Image.Image] = {}
        try:
            for idx, rt in enumerate(present_types):
                prompt = self._build_prompt(rt, caption)
                if reference is not None:
                    baked_rgb, region_ids, _certainty = reference
                    tiles[rt.label] = self._generate_tileable_tile_high_fidelity(
                        prompt, cfg, seed_offset=idx,
                        region_map=region_ids,
                        baked_color=PIL.Image.fromarray(baked_rgb),
                        region_val=int(rt),
                    )
                else:
                    tiles[rt.label] = self._generate_tileable_tile(prompt, cfg, seed_offset=idx)
                self.log_info(
                    f"Generated {rt.label} tile ({cfg.tile_size}px, seamless"
                    f"{', photo-referenced (largest-region crop)' if reference is not None else ''})"
                )
                self.advance_progress(task)
                self.advance_progress(task)

            if cfg.debug_lora_scale_sweep and self.temp is not None:
                self._debug_lora_scale_sweep(cfg, reference, present_types, caption)
        finally:
            self._inpainter.close()
            self._inpainter = None

        material = self._build_material(context, cfg, present_types, tiles, region_map_depth)

        context.add_splat_material(ContextKey.TERRAIN_MATERIAL, material)

        # Embed first tile in TERRAIN_TEXTURE so the mesh GLB has a preview texture
        if material.layers:
            context.add_image(ContextKey.TERRAIN_TEXTURE, Image(material.layers[0].tile))

        if self.temp is not None:
            material.save(self.temp / "splat_material")
            self._save_debug(material, self.temp / "splat_material")

        self.log_info(
            f"SplatMaterial: {material.layer_count} layer(s), "
            f"{len(material.blend_maps)} blend map(s) at {cfg.blend_map_size}px"
        )
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    # ── Debug: LoRA scale sweep ─────────────────────────────────────────────────

    def _debug_lora_scale_sweep(
        self,
        cfg: "TerrainTextureGenerationConfiguration",
        reference: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
        present_types: list[RegionType],
        caption: Any,
    ) -> None:
        """
        Regenerate one representative region's tile at several LoRA scales,
        through the *exact* same reference-seeded high-fidelity path used for
        the real output — same photo reference, same region data, same seed,
        same seam-fix, same FluxInpaintPipeline production actually uses. Only
        the LoRA scale varies, so results are directly comparable and reflect
        real captured data rather than a synthetic prompt-only test.

        No Fill-pipeline/no-LoRA baseline here: FluxFillPipeline is built around
        a *partial* mask that preserves real content outside it via masked-image
        conditioning. Feeding it the full-white mask this stage uses for whole-
        tile generation zeroes that conditioning entirely, which produced
        degenerate output unrelated to any real LoRA-vs-content question — an
        invalid comparison, not a meaningful baseline. scale 0.0 in the sweep
        below (LoRA loaded but weighted to zero, still on FluxInpaintPipeline)
        is the correct "no LoRA effect" reference point.

        Debug-only: writes PNGs to self.temp/lora_sweep/ and never touches
        TERRAIN_MATERIAL or the tiles used for the actual output. Requires a
        photo reference (falls back to a log warning otherwise).
        """
        if reference is None or not present_types:
            self.log_warning("LoRA sweep skipped: no photo reference / region available")
            return

        rt = present_types[0]
        prompt = self._build_prompt(rt, caption)
        baked_rgb, region_ids, _certainty = reference
        baked_color = PIL.Image.fromarray(baked_rgb)

        sweep_dir = self.temp / "lora_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        self.log_info(f"LoRA sweep: region '{rt.label}', writing to {sweep_dir}")

        for scale in cfg.debug_lora_scales:
            self.log_info(f"LoRA sweep: scale {scale:.2f} (reusing loaded pipeline)")
            self._inpainter.generator.pipeline.set_adapters(["pano"], adapter_weights=[scale])
            img = self._generate_tileable_tile_high_fidelity(
                prompt, cfg, seed_offset=0,
                region_map=region_ids, baked_color=baked_color, region_val=int(rt),
            )
            img.convert("RGB").save(sweep_dir / f"inpaint_lora_{scale:.2f}.png")

        # Restore the configured scale in case anything runs after this in the try block.
        self._inpainter.generator.pipeline.set_adapters(["pano"], adapter_weights=[cfg.lora_scale])
        self.log_info(f"LoRA sweep: done, {len(cfg.debug_lora_scales)} scale(s) + baseline in {sweep_dir}")

    # ── Material construction ─────────────────────────────────────────────────

    def _build_material(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
        present_types: list[RegionType],
        tiles: dict[str, PIL.Image.Image],
        region_map_depth,
    ) -> SplatMaterial:
        """
        Assemble the SplatMaterial with an optional panorama layer prepended.

        When PANORAMA_TERRAIN and HEIGHT_MAP are both available and
        use_panorama_layer is True, the real (unbaked, equirect) panorama is
        inserted as layer 0 with a viewing-latitude visibility blend weight
        (see _panorama_visibility_weight) that keeps it dominant across
        almost the whole terrain. The synthetic region layer weights are
        scaled by (1 − panorama_weight) so all weights sum to 1 everywhere.
        """
        # ── Compute raw synthetic region weights ──────────────────────────────
        sigma_px = max(4.0, cfg.blend_sigma * cfg.blend_map_size)
        synth_weight_maps: dict[str, np.ndarray] = {}

        if region_map_depth is not None:
            rm = region_map_depth.depth
            for rt in present_types:
                if rt.label not in tiles:
                    continue
                mask = zoom(
                    (rm == int(rt)).astype(np.float32),
                    (cfg.blend_map_size / rm.shape[0], cfg.blend_map_size / rm.shape[1]),
                    order=1,
                )
                synth_weight_maps[rt.label] = gaussian_filter(mask, sigma=sigma_px)

        if not synth_weight_maps and tiles:
            # Fallback: uniform cover with the first available tile
            label = next(iter(tiles))
            synth_weight_maps[label] = np.ones((cfg.blend_map_size, cfg.blend_map_size), dtype=np.float32)

        # Normalise synthetic weights so they sum to 1
        synth_total = sum(synth_weight_maps.values()) + 1e-6
        for label in synth_weight_maps:
            synth_weight_maps[label] = synth_weight_maps[label] / synth_total

        # ── Panorama layer ────────────────────────────────────────────────────
        panorama_terrain = context.input_panorama(ContextKey.PANORAMA_TERRAIN) if cfg.use_panorama_layer else None
        height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        height_map_params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)

        if panorama_terrain is not None and height_map_depth is not None:
            grid_size = (height_map_params.get("grid_size_meters") if height_map_params else None) or 100.0
            half = grid_size / 2.0

            pano_tile = self._panorama_tile(panorama_terrain, cfg.tile_size)
            pano_weight = self._panorama_visibility_weight(
                height_map_depth.depth, half, cfg.blend_map_size, cfg.panorama_blend_power,
                nadir_cutoff_deg=cfg.nadir_cutoff_deg,
                nadir_fade_deg=cfg.nadir_fade_deg,
                horizon_fade_deg=cfg.horizon_fade_deg,
            )
            self.log_info(
                f"Panorama layer: mean weight {pano_weight.mean():.2f}, "
                f"coverage {(pano_weight > 0.1).mean() * 100:.0f}%"
            )

            # Scale synthetic weights by (1 − panorama_weight)
            synth_scale = (1.0 - pano_weight).clip(0.0, 1.0)
            weight_maps: dict[str, np.ndarray] = {"panorama": pano_weight}
            for label, sw in synth_weight_maps.items():
                weight_maps[label] = sw * synth_scale

            layers = [SplatLayer(name="panorama", tile=pano_tile, tile_factor=1.0, equirect=True, smoothness=0.1)]
            layers += [
                SplatLayer(
                    name=rt.label,
                    tile=tiles[rt.label],
                    tile_factor=cfg.synthetic_tile_factor,
                    smoothness=_LAYER_SMOOTHNESS.get(rt, 0.1),
                )
                for rt in present_types if rt.label in tiles
            ]
        else:
            weight_maps = synth_weight_maps
            layers = [
                SplatLayer(
                    name=rt.label,
                    tile=tiles[rt.label],
                    tile_factor=cfg.synthetic_tile_factor,
                    smoothness=_LAYER_SMOOTHNESS.get(rt, 0.1),
                )
                for rt in present_types if rt.label in tiles
            ]

        if not layers:
            label, tile = next(iter(tiles.items()))
            return SplatMaterial.from_single_layer(label, tile, cfg.blend_map_size)

        return SplatMaterial.from_weight_maps(layers=layers, weight_maps=weight_maps)

    @staticmethod
    def _panorama_tile(panorama, tile_size: int) -> PIL.Image.Image:
        """
        Resize the full equirectangular panorama to a square tile for storage.

        The full image is kept (not cropped to the lower half) so mountain detail
        near and above the horizon is preserved.  The shader uses standard equirect
        UV projection:
            U = atan2(X, Z) / (2π) + 0.5
            V = 0.5 − φ / π   where φ = atan2(Y, √(X²+Z²))
        Mountains at φ > 0 (above camera level) sit at V < 0.5 in the tile.
        Terrain below the horizon sits at V > 0.5.  Nadir maps to V = 1.0.
        """
        return panorama.image.convert("RGB").resize(
            (tile_size, tile_size), PIL.Image.LANCZOS
        )

    @staticmethod
    def _panorama_visibility_weight(
        height_map: np.ndarray,
        half: float,
        blend_map_size: int,
        blend_power: float,
        nadir_cutoff_deg: float = -85.0,
        nadir_fade_deg: float = 4.0,
        horizon_fade_deg: float = 1.5,
    ) -> np.ndarray:
        """
        Per-texel visibility weight for the panorama layer, based on the
        viewing latitude from the camera (at the world origin).

        An earlier version weighted by surface-normal facing instead, but for
        typical rolling/flat terrain viewed from a ~1-2 m camera height, the
        facing dot product collapses to near-zero within a few metres of the
        camera (it decays as camera_height / distance) — so nearly the whole
        terrain fell back to synthetic FLUX tiles instead of the real photo.

        Latitude-based weighting (matching bake_topdown_texture_with_certainty's
        certainty ramp) keeps the real panorama dominant across the terrain.
        It gates on the *magnitude* of the elevation angle from the camera to
        each ground point, not its sign: a point sitting above the camera's
        own height is exactly as visible in the source panorama as one below
        it, since both are genuine, photographed ground. The only two
        equirectangular failure modes are viewing angles near-vertical (the
        nadir/zenith pole singularity directly under — or, for a tall rise,
        directly toward — the camera) and near-horizontal (a thin band right
        at the horizon, where distant ground is compressed into just a few
        panorama pixel rows and starts to look blocky when magnified).
        Gating on sign instead of magnitude would zero out real, visible
        terrain whenever the height map's values don't happen to fall below
        the camera's zero reference.

        Returns a float32 array of shape (blend_map_size, blend_map_size) in [0, 1].
        """
        us = np.linspace(0.0, 1.0, blend_map_size, dtype=np.float32)
        ug, vg = np.meshgrid(us, us)
        X = (ug - 0.5) * (2.0 * half)
        Z = (vg - 0.5) * (2.0 * half)

        hm_h, hm_w = height_map.shape
        row_c = ((Z + half) / (2.0 * half) * (hm_h - 1)).clip(0, hm_h - 1)
        col_c = ((X + half) / (2.0 * half) * (hm_w - 1)).clip(0, hm_w - 1)
        Y = map_coordinates(
            height_map, [row_c.ravel(), col_c.ravel()], order=1, mode="nearest"
        ).reshape(blend_map_size, blend_map_size).astype(np.float32)
        Y = np.nan_to_num(Y, nan=0.0)

        r_xz = np.sqrt(X.astype(np.float64) ** 2 + Z.astype(np.float64) ** 2).clip(1e-6)
        lat = np.arctan2(Y.astype(np.float64), r_xz)  # elevation angle, camera at origin

        abs_lat_deg     = np.degrees(np.abs(lat))
        pole_cutoff_deg = abs(nadir_cutoff_deg)

        fade_in  = ((pole_cutoff_deg - abs_lat_deg) / nadir_fade_deg).clip(0.0, 1.0)
        fade_out = (abs_lat_deg / horizon_fade_deg).clip(0.0, 1.0)
        weight   = np.minimum(fade_in, fade_out).astype(np.float32)

        return (weight ** blend_power).astype(np.float32)

    # ── Photo reference (weak img2img seed) ─────────────────────────────────────

    def _bake_reference(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Bake the real photo top-down, aligned to REGION_MAP, for use as a weak
        img2img seed per region tile. Returns (baked_rgb, region_ids, certainty),
        all at cfg.reference_tex_size, or None if the source imagery is missing.
        """
        panorama_terrain = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        if panorama_terrain is None or height_map_depth is None or region_map_depth is None:
            return None

        height_map_params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        grid_size = (height_map_params.get("grid_size_meters") if height_map_params else None) or 100.0
        half = grid_size / 2.0

        height_certainty_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        baked_img, bake_certainty = TerrainMeshGenerator.bake_topdown_texture_with_certainty(
            panorama_terrain,
            height_map_depth.depth,
            half, half,
            tex_size=cfg.reference_tex_size,
            height_certainty=height_certainty_depth.depth if height_certainty_depth is not None else None,
        )
        baked_rgb = np.array(baked_img, dtype=np.uint8)

        rm = region_map_depth.depth
        region_ids = zoom(
            rm.astype(np.float32),
            (cfg.reference_tex_size / rm.shape[0], cfg.reference_tex_size / rm.shape[1]),
            order=0,
        ).astype(np.int32)

        return baked_rgb, region_ids, bake_certainty

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _present_region_types(
        self,
        region_map_depth,
        min_fraction: float,
    ) -> list[RegionType]:
        if region_map_depth is None:
            return [RegionType.GROUND]
        rm = region_map_depth.depth
        total = rm.size
        present = [
            rt for rt in _GROUND_TYPES
            if (rm == int(rt)).sum() / total >= min_fraction
        ]
        if not present:
            return [RegionType.GROUND]
        present.sort(key=lambda rt: -(rm == int(rt)).sum())
        return present[:8]

    def _build_prompt(self, rt: RegionType, caption: Any) -> str:
        base = _BASE_PROMPTS.get(rt, "close-up macro photo of natural outdoor ground surface material")
        lora = PanoramaLoraType.FLUX_SEAMLESS_TEXTURE
        return f"{lora_prompt_prefix(lora)}{base}{_TILE_SUFFIX}{lora_prompt_suffix(lora)}"

    def _generate_tileable_tile(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
        inpainter: Optional[InPainting] = None,
    ) -> PIL.Image.Image:
        """
        Pure text-to-image tile generation for regions with no usable photo
        reference at all (either the whole scene has none, or this specific
        region has no trustworthy pixels in the baked photo).

        inpainter: override the stage's own self._inpainter (used by the debug
        LoRA sweep to run an alternate model/LoRA-scale through the identical
        code path). Defaults to self._inpainter.
        """
        inpainter = inpainter if inpainter is not None else self._inpainter
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        base_image = PIL.Image.new("RGB", (T, T), (128, 128, 128))
        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = inpainter.inpaint(
            input_image=base_image,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=1.0,
            seed=seed,
        )

        result_img = self._seam_fix(generated, cfg, prompt, seed, inpainter=inpainter)
        return self._pack_local_height_channel(result_img)

    def _generate_tileable_tile_high_fidelity(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
        region_map: np.ndarray,
        baked_color: PIL.Image.Image,
        region_val: int,
        inpainter: Optional[InPainting] = None,
    ) -> PIL.Image.Image:
        """
        Tile generation guided by a single crop of the largest connected
        component of this region in the baked photo (_extract_largest_region_crop)
        — one real location, so it's inherently coherent (consistent lighting/
        exposure) rather than a mosaic of patches from potentially very
        different conditions. The crop is pasted centred on the canvas and used
        as genuine partial-inpaint context (_build_reference_canvas): FLUX sees
        it as real, unmasked pixels via the pipeline's own mask/masked-image
        conditioning and extends the material outward to fill the tile, rather
        than the whole canvas being a soft img2img seed under a meaningless
        full-white mask. Falls back to plain text generation if the region has
        no pixels at all.

        inpainter: override the stage's own self._inpainter (used by the debug
        LoRA sweep to run an alternate model/LoRA-scale through the identical
        code path, against the same real reference crop). Defaults to
        self._inpainter.
        """
        inpainter = inpainter if inpainter is not None else self._inpainter
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        patch_size = max(16, int(T * cfg.reference_patch_fraction))
        patch = self._extract_largest_region_crop(
            baked_color=baked_color,
            region_map=region_map,
            region_val=region_val,
            patch_size=patch_size,
        )
        if patch is None:
            return self._generate_tileable_tile(prompt, cfg, seed_offset, inpainter=inpainter)

        canvas, mask = self._build_reference_canvas(patch, T, cfg.mask_feather_px)

        generated = inpainter.inpaint(
            input_image=canvas,
            mask_image=mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.reference_strength,
            seed=seed,
        )

        result_img = self._seam_fix(generated, cfg, prompt, seed, inpainter=inpainter)

        # Pass 3: detail recovery — the two inpainting passes can soften the
        # reference crop's crisp stones/grass structure, so restore local contrast.
        result_img = result_img.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
        result_img = PIL.ImageEnhance.Contrast(result_img).enhance(1.15)
        result_img = PIL.ImageEnhance.Sharpness(result_img).enhance(1.20)

        # Pass 4: nudge final colour statistics back toward the real patch (not the
        # filled canvas background, which is only a synthetic mean-colour filler).
        if cfg.color_transfer_strength > 0.0:
            result_img = lab_color_transfer(
                source=patch, target=result_img, strength=cfg.color_transfer_strength,
            )

        return self._pack_local_height_channel(result_img)

    @staticmethod
    def _build_reference_canvas(
        patch: PIL.Image.Image,
        tile_size: int,
        feather_px: float,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        """
        Paste `patch` centred on a tile_size canvas and build the matching
        partial-inpaint mask: black (0, keep) over the patch, white (255,
        generate) everywhere else, boundary Gaussian-feathered so the
        transition blends rather than leaving a hard rectangular seam.

        The canvas background outside the patch is filled with the patch's own
        mean colour. With strength=1.0 the pipeline fully replaces the
        "generate" region regardless of what's behind it, so this filler only
        actually shows through at the feathered boundary — reading as a
        continuation of the patch there instead of a contrasting flat colour.

        Returns (canvas RGB, mask L).
        """
        T = tile_size
        p = patch.size[0]
        mean_color = tuple(
            np.array(patch.convert("RGB")).reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist()
        )

        canvas = PIL.Image.new("RGB", (T, T), mean_color)
        offset = (T - p) // 2
        canvas.paste(patch, (offset, offset))

        mask_arr = np.full((T, T), 255, dtype=np.uint8)
        mask_arr[offset:offset + p, offset:offset + p] = 0
        mask = PIL.Image.fromarray(mask_arr, "L")
        if feather_px > 0:
            mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px))

        return canvas, mask

    def _seam_fix(
        self,
        generated: PIL.Image.Image,
        cfg: TerrainTextureGenerationConfiguration,
        prompt: str,
        seed: int,
        inpainter: Optional[InPainting] = None,
    ) -> PIL.Image.Image:
        """Circular-shift the tile by half its size and inpaint over the new seam, then shift back."""
        inpainter = inpainter if inpainter is not None else self._inpainter
        T = cfg.tile_size
        arr = np.array(generated.convert("RGB"), dtype=np.uint8)
        half = T // 2
        shifted = np.roll(np.roll(arr, half, axis=0), half, axis=1)

        seam_mask = np.zeros((T, T), dtype=np.uint8)
        sw = max(2, int(T * cfg.seam_width_fraction / 2))
        seam_mask[max(0, half - sw): min(T, half + sw), :] = 255
        seam_mask[:, max(0, half - sw): min(T, half + sw)] = 255
        if cfg.seam_dilation_px > 0:
            seam_mask = (
                binary_dilation(seam_mask, iterations=cfg.seam_dilation_px).astype(np.uint8) * 255
            )

        fixed = inpainter.inpaint(
            input_image=PIL.Image.fromarray(shifted),
            mask_image=PIL.Image.fromarray(seam_mask, "L"),
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=seed + 1000,
        )

        result = np.array(fixed.convert("RGB"), dtype=np.uint8)
        result = np.roll(np.roll(result, -half, axis=0), -half, axis=1)
        return PIL.Image.fromarray(result)

    def _extract_largest_region_crop(
        self,
        baked_color: PIL.Image.Image,
        region_map: np.ndarray,
        region_val: int,
        patch_size: int,
    ) -> Optional[PIL.Image.Image]:
        """
        Crop a single square from the largest connected component of this
        region in the baked top-down photo, then resize to patch_size. This is
        a guidance patch pasted into a larger tile canvas (see
        _build_reference_canvas), not necessarily the full tile.

        A previous version scattered many small patches from anywhere in the
        region into a collage. That produced a hard-seamed, visibly patchy
        mosaic whenever the region spanned meaningfully different lighting or
        exposure (sunlit rock vs. shadow vs. a warm sunset glow, all the same
        region label) — each patch only feathered against its own mask
        boundary, never against its neighbours on the canvas, so adjacent
        patches could clash violently. A single crop from one real, contiguous
        patch of material is inherently coherent — one location, one
        lighting/exposure condition, genuine continuous texture — at the cost
        of only capturing that one location's appearance rather than
        sampling the region's full variety.

        The crop is centred on the component's bounding box, sized to the
        larger of its two bbox dimensions (clamped to the image) so resizing
        to patch_size doesn't distort aspect ratio; pixels within that square
        outside the component itself are real neighbouring photo content, not
        masked out, since re-introducing per-pixel masking here would bring
        back the same patchiness this replaces.

        Returns None if the region has no pixels at all in region_map — the
        caller then falls back to plain text-to-image generation.
        """
        W, H = baked_color.size
        rm_h, rm_w = region_map.shape
        region_map_res = (
            zoom(region_map, (H / rm_h, W / rm_w), order=0, prefilter=False)
            if (rm_h, rm_w) != (H, W) else region_map
        )

        mask = region_map_res == region_val
        if not mask.any():
            return None

        labeled, _ = ndi_label(mask)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        largest_label = int(sizes.argmax())
        ys, xs = np.nonzero(labeled == largest_label)

        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
        side = min(max(r1 - r0, c1 - c0), H, W)
        half = side // 2
        sr0 = int(np.clip(cy - half, 0, H - side))
        sc0 = int(np.clip(cx - half, 0, W - side))

        color_arr = np.array(baked_color.convert("RGB"))
        crop = color_arr[sr0:sr0 + side, sc0:sc0 + side]
        return PIL.Image.fromarray(crop).resize((patch_size, patch_size), PIL.Image.LANCZOS)

    @staticmethod
    def _pack_local_height_channel(rgb_img: PIL.Image.Image) -> PIL.Image.Image:
        """
        Derive a seamless local micro-height/displacement map from the tile's
        own illumination detail and pack it into the alpha channel, so the
        terrain shader can do height-biased blending between layers (e.g.
        grass poking through the cracks between dirt clumps) instead of a
        flat linear cross-fade. A high-pass filter isolates local structural
        bumps (rocks, blades, grain) while removing the broad lighting
        gradient, since the tile is meant to be lit flat/overcast already.
        """
        gray = np.array(rgb_img.convert("L"), dtype=np.float32)
        high_freq = gray - gaussian_filter(gray, sigma=8.0)

        h_min, h_max = high_freq.min(), high_freq.max()
        if h_max - h_min > 1e-5:
            height = (high_freq - h_min) / (h_max - h_min) * 255.0
        else:
            height = np.full_like(gray, 128.0)
        # Smooth slightly to avoid single-pixel flicker in the shader's blend weights.
        height = gaussian_filter(height, sigma=1.0).clip(0, 255).astype(np.uint8)

        rgba = np.dstack([np.array(rgb_img.convert("RGB"), dtype=np.uint8), height])
        return PIL.Image.fromarray(rgba, "RGBA")

    # ── Debug output ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_debug(material: SplatMaterial, path: "Path") -> None:
        """Write human-readable debug images alongside the raw splat files."""
        import numpy as np
        from PIL import Image as PILImage

        if not material.blend_maps:
            return

        # Distinct colours per layer for the dominant-region overlay
        palette = [
            (106, 190, 106),   # green
            (139, 90,  43),    # brown
            (80,  140, 200),   # blue
            (80,  80,  80),    # grey
            (200, 180, 100),   # tan
            (169, 80,  80),    # red-brown
            (160, 110, 180),   # purple
            (60,  180, 180),   # teal
        ]

        # Unpack all blend map channels into a list of weight arrays
        weight_arrays: list[np.ndarray] = []
        for bm in material.blend_maps:
            arr = np.array(bm).astype(np.float32) / 255.0   # (H, W, 4)
            for ch in range(4):
                if len(weight_arrays) < material.layer_count:
                    weight_arrays.append(arr[:, :, ch])

        if not weight_arrays:
            return

        h, w = weight_arrays[0].shape
        stack = np.stack(weight_arrays, axis=-1)   # (H, W, N)

        # Dominant-region colour map
        dominant_idx = stack.argmax(axis=-1)       # (H, W) — index of winning layer
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(material.layer_count):
            color_map[dominant_idx == idx] = palette[idx % len(palette)]
        PILImage.fromarray(color_map).save(path / "blend_dominant.png")

        # Per-layer greyscale weight maps
        for idx, layer in enumerate(material.layers):
            grey = (weight_arrays[idx] * 255).clip(0, 255).astype(np.uint8)
            PILImage.fromarray(grey, "L").save(path / f"blend_weight_{layer.name}.png")

    # ── Stage bookkeeping ─────────────────────────────────────────────────────

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.TERRAIN_MATERIAL)

    def model_names(self) -> list[str]:
        return InPainting.model_names(
            self.config.inpainting_type, lora_type=PanoramaLoraType.FLUX_SEAMLESS_TEXTURE,
        )

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        super().clean_up()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        material: Optional[SplatMaterial] = context.splat_material(ContextKey.TERRAIN_MATERIAL)
        if material is None:
            return None
        cfg: TerrainTextureGenerationConfiguration = self.config
        return ReportSection(
            stage_name=self.name,
            title="Terrain Texture Generation",
            body=(
                f"High-quality seamlessly tileable textures were generated for "
                f"{material.layer_count} ground region(s) using FLUX inpainting "
                "(two-pass: full-mask generation then circular-shift seam repair). "
                f"Region weights were packed into {len(material.blend_maps)} RGBA "
                "blend map(s) for runtime per-region blending."
            ),
            images=[(layer.tile, layer.name) for layer in material.layers],
            stats={
                "Regions": ", ".join(l.name for l in material.layers),
                "Tile size": f"{cfg.tile_size} × {cfg.tile_size} px",
                "Blend maps": str(len(material.blend_maps)),
                "Blend map size": f"{cfg.blend_map_size} × {cfg.blend_map_size} px",
            },
        )
