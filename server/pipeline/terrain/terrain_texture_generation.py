import math
from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, zoom
from scipy.ndimage import map_coordinates

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.captioning.image_captioning import ImageCaptioning, CaptioningModel
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.intrinsic_images.image_intrinsics import ImageIntrinsics
from pipeline.supersampling.image_supersampling import ImageSupersampling
from pipeline.panorama.panorama_lora import PanoramaLoraType, lora_prompt_prefix, lora_prompt_suffix
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.terrain.pattern_texture import (
    bake_real_layer, bake_real_layer_from_mesh, synthesize_region_layer, vegetation_bleed_mask,
)
from pipeline.terrain.terrain_generator import TerrainMeshGenerator
from scene.splat_material import SplatLayer, SplatMaterial
from util.image_utils import Image, lab_color_transfer
from util.device_utils import DeviceStrategy, preferred_device
from util.panorama_utils import Panorama


_GROUND_TYPES: frozenset[RegionType] = frozenset({
    RegionType.GROUND,
    RegionType.TERRAIN,
    RegionType.VEGETATION,
    RegionType.ROAD,
})

# Short on purpose: at FLUX Fill's guidance_scale (see TerrainTextureGenerationConfiguration)
# the text prompt strongly dominates the img2img seed, so an elaborate, hyper-specific
# description would fight the actual photo reference instead of following it. These name
# the material and its 1-2 defining traits only — the real reference crops
# (_extract_reference_patches) and per-region LAB colour transfer carry the rest.
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
        # Real pixels of context kept on each side of the seam band when _seam_fix_band
        # crops it out -- gives FLUX genuine surrounding texture to condition on rather
        # than a crop whose edge is exactly the mask boundary (which reads as a hard
        # image border, the same reasoning as heal_wrap_seam's crop_context_px for the
        # skybox panorama's own wrap-seam repair).
        seam_fix_context_px: int = 64,
        # Longest edge (px) the cropped seam band is downscaled to before FLUX, then
        # LANCZOS'd back up to the crop's real size. Deliberately well under tile_size:
        # running FLUX directly at full tile resolution on a strip this narrow tends to
        # stamp a crisp, mismatched patch of new high-frequency detail right on the seam
        # (the same blocky "patch/grid" artifact noted in skybox_inpainting for FLUX run
        # above its comfortable resolution) instead of a smoother fill that blends into
        # its real neighbours on both sides.
        seam_fix_max_size: int = 512,
        # Gaussian feather radius (px, at the crop's own resolution) on the seam band mask
        # before compositing the healed crop back over the original -- wide enough that the
        # fix fades in as a soft transition rather than a hard-edged patch, so real detail
        # everywhere outside the band is preserved untouched.
        seam_fix_feather_px: float = 24.0,
        # Width (as a fraction of tile_size) of the ring around all four tile edges that
        # Pass 1 actually has to generate -- everything inside it is real reference-patch
        # content (see _build_reference_canvas), tiled edge to edge to fill the interior.
        # This is a genuine outpaint (extend a large real photo's edge a little) rather
        # than extrapolating most of the tile from a small inset island of real content,
        # which is what a large border_fill_fraction relative to the patches' own coverage
        # used to mean here (an earlier version of this stage kept a small patch at native
        # size and stretched a blurry copy of it across the *whole* tile as a "background"
        # while marking most of the canvas "generate" anyway -- the model never actually
        # saw that background, since both LaMa and diffusion inpainting zero out or heavily
        # denoise whatever sits under a "generate" pixel before it's ever conditioned on).
        outpaint_border_fraction: float = 0.08,
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
        # Withhold the panorama layer from ground texels whose own panorama pixel is
        # upright vegetation/structure rather than ground -- see
        # _panorama_visibility_weight for the radial-pinwheel failure this fixes.
        panorama_upright_gate: bool = True,
        panorama_upright_feather_frac: float = 0.01,
        panorama_upright_full_radius_m: float = 6.0,
        panorama_upright_falloff_radius_m: float = 25.0,
        # Longest-edge cap (px) for the panorama layer's own tile -- deliberately
        # decoupled from tile_size, which sizes the *synthetic* FLUX tiles (a real
        # generation constraint that doesn't apply here). This layer is just a
        # resize of the real photo, so it should carry as much of that photo's
        # actual resolution as reasonable, not the same 1024px budget the FLUX
        # model needs. Also stops forcing the panorama's native ~2:1 equirect
        # aspect into a square, which discarded vertical resolution for no
        # reason -- the shader samples it with independent U/V in [0,1], so a
        # non-square texture is not a problem. Native resolution is used as-is
        # whenever it's already under this cap.
        panorama_layer_max_resolution: int = 4096,
        # 200 m terrain / 4 m per tile = 50 repeats → ~4 cm/texel at 1024 px
        synthetic_tile_factor: float = 50.0,
        use_photo_reference: bool = True,
        # Flatten each reference patch (_extract_reference_patches) from its own real
        # depth points (Panorama.unwarp_box) instead of a plain perspective reprojection
        # -- removes true oblique-view foreshortening (the ground was genuinely
        # photographed at an angle), not just equirect distortion. Falls back to
        # perspective_crop automatically per-patch if PANORAMA_DEPTH is unavailable or a
        # given patch doesn't have enough valid depth points to fit a plane. Set False to
        # force the old perspective_crop-only behaviour.
        use_point_cloud_unwarp: bool = True,
        # Caption the real reference crop with Florence-2 (_describe_material) and use that
        # description as the prompt's subject instead of the generic per-region-type phrase
        # in _BASE_PROMPTS -- e.g. "a close-up of cracked, sun-bleached mud" instead of the
        # generic "macro close-up photograph of loose soil grit...". Falls back to
        # _BASE_PROMPTS if there's no reference crop, or captioning fails/yields nothing.
        describe_material: bool = True,
        # How many reference crops (_extract_reference_patches) to tile across the tile's
        # interior grid (_build_reference_canvas). Multiple patches give Pass 1 more than
        # one real colour/lighting anchor to match against instead of extrapolating a
        # single patch's colour across the whole interior -- helps with colour drift far
        # from it, at the cost of a visible cell boundary between differently-lit patches.
        num_reference_patches: int = 3,
        # Each extracted patch's own resolution (independent of its on-tile size, which the
        # interior grid resizes to fill its own cell regardless) -- this fraction of the
        # tile's linear size when there's only one; with num_reference_patches > 1, each
        # patch is scaled down by 1/sqrt(n) so total sampled area stays comparable.
        reference_patch_fraction: float = 0.5,
        # Run each real-photo reference patch through IntrinsicDiffusion before it's used
        # as a Pass 1 guidance seed, and swap its RGB for the model's predicted albedo
        # (delit, lighting-free reflectance) while keeping the patch's own region-mask
        # alpha channel untouched. The raw photo patch carries whatever sun/shadow/exposure
        # the source panorama happened to have, which then gets stamped across the whole
        # tile at synthetic_tile_factor repeats -- a lighting pattern that doesn't tile and
        # fights the target tile's own "flat overcast, no directional shadows" prompt.
        # Falls back to the raw patch if the model errors on a given crop.
        use_intrinsic_delighting: bool = True,
        # Blend factor toward the predicted albedo: 1.0 = pure albedo (fully delit), 0.0 =
        # the raw patch untouched. IntrinsicDiffusion's albedo can look unnaturally flat/
        # washed out at full strength -- a partial blend keeps some of the patch's own real
        # shading/micro-contrast while still knocking down the baked-in directional sun/
        # shadow that fights the tile's own flat-overcast prompt.
        intrinsic_delight_strength: float = 0.6,
        # Working resolution for the delighting pass -- independent of tile_size since
        # patches are much smaller than a full tile; upsampled internally then resized
        # back down to the patch's own size.
        intrinsic_resolution: int = 768,
        # Samples averaged per delit patch (IntrinsicDiffusion's agg_num). Lower than the
        # model's own default (4) since this is a guidance seed, not final output -- trades
        # a bit of denoising quality for roughly proportionally faster inference.
        intrinsic_agg_num: int = 2,
        # Run each (optionally delit) reference patch through Swin2SR 2x super-resolution,
        # then LANCZOS back down to patch_size, before it's composited onto the Pass 1
        # canvas. patch_size is well under tile_size (see reference_patch_fraction), so the
        # canvas background fill (_build_reference_canvas pastes the largest patch, then
        # LANCZOS-stretches a copy of it up to the full tile) is normally the single
        # biggest upscale in this whole stage -- stretching directly from a native
        # diffusion-resolution patch leaves it visibly soft. Supersampling first means that
        # stretch (and the patches' own pasted-in detail) comes from a sharper, denoised
        # source instead, the same "render bigger, downsample for quality" trick already
        # used for Flux fills elsewhere (see PanoramaInpaintingConfiguration.supersample_inpaint).
        use_patch_supersampling: bool = True,
        # Pass 1 (_pass1_extend_patch): extend the real patch to fill the whole tile, using
        # it purely as guidance, not something to hard-preserve.
        #   LAMA (default) -- classical texture-completion inpainting (no prompt, no LoRA,
        #     no hallucination): propagates the patch's own real texture into the
        #     surrounding hole via genuine local image statistics. The patch region is
        #     masked as "keep" here since LaMa has no img2img/strength concept to loosely
        #     guide from -- protecting it via mask is the only way to reference it at all.
        #   FLUX -- ordinary img2img over the same canvas (full mask, reference_strength,
        #     this stage's own FLUX pipeline with its LoRA temporarily disabled) -- loosely
        #     guided by the patch without hard-preserving it.
        # Either way, Pass 2 (FLUX + FLUX_SEAMLESS_TEXTURE LoRA) then repaints the *entire*
        # tile again -- including the original patch region -- to push toward genuine
        # tileability and the photorealistic macro finish.
        pass1_inpainting_type: str = "LAMA",
        # Gaussian feather radius (px) on the Pass-1 mask boundary (FLUX option only --
        # LaMa thresholds its own mask internally, so feathering there is a no-op).
        mask_feather_px: float = 12.0,
        # Diffusers `strength`: used for Pass 1 when pass1_inpainting_type is FLUX (ordinary
        # img2img strength), and for Pass 2's FLUX+LoRA repaint over the full Pass-1 tile
        # (full mask there -- this is the only thing keeping it anchored to Pass 1's real-
        # texture content rather than inventing something else entirely).
        reference_strength: float = 0.60,
        # Post-process LAB colour nudge toward the Pass-1 (real-texture-grounded) tile.
        color_transfer_strength: float = 0.60,
        # FLUX_SEAMLESS_TEXTURE LoRA strength. Lower than full (1.0) so its seamless-tiling
        # bias steers generation without overpowering the per-region material prompt or the
        # reference crop's genuine detail.
        lora_scale: float = 0.8,
        # Temporarily disabled -- was producing visible artifacts. When False, all FLUX
        # calls in this stage (Pass 1 fallback, Pass 2, seam-fix) run on plain
        # FLUX.1-Fill-dev with no LoRA loaded at all (not just zero-weighted).
        use_lora: bool = True,
        # Debug: save every intermediate image of the generation process (reference patch,
        # Pass 1 canvas/mask/result, Pass 2 raw/seam-fixed/sharpened/final) per region to
        # self.temp/texture_generation/, so each step can be inspected. No-op unless
        # self.temp is set.
        debug_save_steps: bool = True,
        # Real-material pattern texturing (see pipeline/terrain/pattern_texture.py) --
        # tried first for every region type; today's FLUX generation above is the
        # explicit last-resort fallback when this returns nothing (no usable real
        # reference for that type, or the terrain mesh isn't available yet).
        use_pattern_texture: bool = True,
        # A triangle "needs synthesis" (pattern-texture territory, not left to the
        # panorama layer) when _panorama_visibility_weight at its centroid is below
        # this. Matches the panorama layer's own certainty gate so the two never
        # fight over the same ground -- see _pattern_texture_context.
        pattern_certainty_threshold: float = 0.5,
        # Reference crops for the pattern-texture library reuse _extract_reference_patches
        # (same mechanism the FLUX path uses), just with different sizing --  library
        # samples are small triangular crops, not a whole tile's guidance seed.
        pattern_reference_patch_size: int = 512,
        pattern_num_reference_patches: int = 4,
        # Working resolution of one triangular library sample.
        pattern_sample_res: int = 128,
        # Fraction of a sample's inradius blended toward the shared border colour.
        # Wide and low-contrast (not a thin flat ring) reads as one continuous
        # material rather than a visibly "faceted" tiling -- see pattern_texture.py.
        pattern_border_width_frac: float = 0.45,
        # Image-space feather (px, at pattern_canvas_size) blending the synthesized
        # region back toward real photo content at its boundary with the panorama
        # layer's own territory.
        pattern_feather_band_px: float = 48.0,
        pattern_feather_sigma_px: float = 8.0,
        # Skip a region type's pattern-texture layer (fall back to FLUX) if its
        # needs-synthesis footprint is smaller than this fraction of the canvas --
        # not worth a dedicated bake.
        pattern_min_coverage_fraction: float = 0.002,
        # Resolution of the baked pattern-texture layer (a one-shot world-space
        # bake, tile_factor=1.0 -- not tiled like the FLUX layers, so this alone
        # sets its on-terrain texel density).
        pattern_canvas_size: int = 2048,
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
        self.seam_fix_context_px = seam_fix_context_px
        self.seam_fix_max_size = seam_fix_max_size
        self.seam_fix_feather_px = seam_fix_feather_px
        self.outpaint_border_fraction = outpaint_border_fraction
        self.use_panorama_layer = use_panorama_layer
        self.panorama_blend_power = panorama_blend_power
        self.nadir_cutoff_deg = nadir_cutoff_deg
        self.nadir_fade_deg = nadir_fade_deg
        self.panorama_upright_gate = panorama_upright_gate
        self.panorama_upright_feather_frac = panorama_upright_feather_frac
        self.panorama_upright_full_radius_m = panorama_upright_full_radius_m
        self.panorama_upright_falloff_radius_m = panorama_upright_falloff_radius_m
        self.horizon_fade_deg = horizon_fade_deg
        self.panorama_layer_max_resolution = panorama_layer_max_resolution
        # UV tiling factor for synthetic region tiles (panorama layer always uses 1.0).
        # At 50× over a 200 m grid one tile covers 4 m → ~0.4 cm/texel at 1024 px.
        self.synthetic_tile_factor = synthetic_tile_factor
        self.use_photo_reference = use_photo_reference
        self.use_point_cloud_unwarp = use_point_cloud_unwarp
        self.describe_material = describe_material
        self.num_reference_patches = num_reference_patches
        self.reference_patch_fraction = reference_patch_fraction
        self.use_intrinsic_delighting = use_intrinsic_delighting
        self.intrinsic_delight_strength = intrinsic_delight_strength
        self.intrinsic_resolution = intrinsic_resolution
        self.intrinsic_agg_num = intrinsic_agg_num
        self.use_patch_supersampling = use_patch_supersampling
        self.pass1_inpainting_type = pass1_inpainting_type
        self.mask_feather_px = mask_feather_px
        self.reference_strength = reference_strength
        self.color_transfer_strength = color_transfer_strength
        self.lora_scale = lora_scale
        self.use_lora = use_lora
        self.debug_save_steps = debug_save_steps
        self.use_pattern_texture = use_pattern_texture
        self.pattern_certainty_threshold = pattern_certainty_threshold
        self.pattern_reference_patch_size = pattern_reference_patch_size
        self.pattern_num_reference_patches = pattern_num_reference_patches
        self.pattern_sample_res = pattern_sample_res
        self.pattern_border_width_frac = pattern_border_width_frac
        self.pattern_feather_band_px = pattern_feather_band_px
        self.pattern_feather_sigma_px = pattern_feather_sigma_px
        self.pattern_min_coverage_fraction = pattern_min_coverage_fraction
        self.pattern_canvas_size = pattern_canvas_size


class TerrainTextureGenerationStage(PipelineStage):
    """
    Generates high-quality tileable region textures and packages them as a SplatMaterial.

    For each ground region type present in the REGION_MAP (above min_region_fraction):
      1. If a real photo is available (PANORAMA_TERRAIN + PANORAMA_REGION_TYPE_MAP_TERRAIN),
         crop num_reference_patches squares directly from the most solid,
         mutually distant locations of that region in the panorama's own
         pixels (_extract_reference_patches), reprojected to a perspective
         (rectilinear) view via Panorama.perspective_crop to remove
         equirectangular distortion — no top-down/orthographic reprojection,
         which produced radial/pinwheel artefacts near the camera. Each patch
         is one real, contiguous location — coherent lighting/exposure,
         genuine continuous texture — unlike stitching many small patches
         from potentially very different lighting conditions within the same
         region label, which produces a hard-seamed, visibly patchy mosaic;
         using several well-separated patches instead of just one also gives
         later steps more than one real colour/lighting anchor to match
         against.
      1b. If use_intrinsic_delighting, each patch is run through
         IntrinsicDiffusion (_delight_patch) and its RGB swapped for the
         model's predicted albedo — the raw photo patch's own baked-in sun/
         shadow/exposure would otherwise get stamped across the whole tile at
         synthetic_tile_factor repeats, which doesn't tile and fights the
         target tile's own flat-overcast prompt.
      2. Pass 1 (_pass1_extend_patch): tile those crops edge to edge across a
         grid that fills the tile's whole interior (_build_reference_canvas)
         — 1 patch fills it entirely, more are tiled with no gaps — leaving
         only a thin outpaint_border_fraction ring around the four edges for
         Pass 1 to actually generate. This is a genuine, bounded outpaint of
         real photo content past its own edge, not extrapolating most of the
         tile from a small inset patch. By default LaMa (classical texture-
         completion, no prompt/LoRA/hallucination — it propagates the
         patches' own real texture into that ring), or plain FLUX img2img
         with the LoRA disabled. A region with no pixels at all falls back
         to plain text generation.
      3. Pass 2: FLUX + FLUX_SEAMLESS_TEXTURE LoRA repaints the *entire* Pass-1
         tile again (full mask, moderate reference_strength) — including the
         original patch regions, not hard-preserved — pushing toward genuine
         tileability and the photorealistic macro finish. Sharpen/contrast-
         boost recovers detail softened by inpainting, and colour statistics
         nudge back toward the Pass-1 tile. Seam fix (circular-shift +
         crop/downscale/inpaint/upscale-back per seam arm, see _seam_fix) runs
         last, against this final sharpened/colour-graded tile, rather than
         in the middle where later passes could re-disturb it. A local
         micro-height map derived from the tile's own high-frequency detail
         is packed into its alpha channel (_pack_local_height_channel) so the
         terrain shader can do height-biased blending between layers instead
         of a flat linear cross-fade.

    SplatMaterial.from_region_map handles weight computation, normalisation, and
    RGBA blend map packing.  The first layer tile is also written to TERRAIN_TEXTURE
    so the terrain mesh can embed a preview in the GLB (with alpha stripped —
    see TerrainMeshGenerator.generate — since that channel holds height, not opacity).

    Reads:
      ContextKey.REGION_MAP              — top-down region type grid (optional)
      ContextKey.PANORAMA_TERRAIN        — real photo, source for the reference crop (optional)
      ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN — panorama-space per-pixel region type,
                                            classified from the same panorama_terrain (not the
                                            original-panorama PANORAMA_REGION_TYPE_MAP), source
                                            for picking reference crop locations (optional)
      ContextKey.PANORAMA_DEPTH          — panorama-aligned depth, used to flatten each
                                            reference crop from its own real 3D points
                                            (optional; falls back to perspective_crop only)
      ContextKey.PANORAMA_FOREGROUND_MASK — ObjectClear's dilated foreground-removal mask;
                                            excluded from reference-crop candidates so fabricated
                                            fill in PANORAMA_TERRAIN is never picked as a "real"
                                            texture source (optional)
      ContextKey.HEIGHT_MAP              — for the panorama blend layer's visibility ramp (optional)
      ContextKey.INPUT_CAPTION           — scene caption for prompt context (optional)

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
        self._lama_inpainter: Optional[InPainting] = None
        self._captioner: Optional[ImageCaptioning] = None
        self._intrinsics: Optional[ImageIntrinsics] = None
        self._samp: Optional[ImageSupersampling] = None

    def _init_inpainter(self) -> None:
        if self._inpainter is None:
            cfg: TerrainTextureGenerationConfiguration = self.config
            inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
            self._inpainter = InPainting(
                inpaint_device, inpaint_dtype, cfg.inpainting_type,
                lora_type=PanoramaLoraType.FLUX_SEAMLESS_TEXTURE if cfg.use_lora else None,
                lora_scale=cfg.lora_scale,
            )

    def _init_lama_inpainter(self) -> None:
        """Lazily load LaMa for Pass 1 (real-texture extension) — only if actually used."""
        if self._lama_inpainter is None:
            device, dtype = preferred_device(DeviceStrategy.MEMORY)
            self._lama_inpainter = InPainting(device, dtype, InPaintingType.LAMA)

    def _init_captioner(self) -> None:
        """Lazily load Florence-2 for material captioning — only if actually used."""
        if self._captioner is None:
            device, _ = preferred_device(DeviceStrategy.MEMORY)
            self._captioner = ImageCaptioning(device, model=CaptioningModel.FLORENCE2)

    def _init_intrinsics(self) -> None:
        """Lazily load IntrinsicDiffusion for reference-patch delighting — only if actually used."""
        if self._intrinsics is None:
            device, _ = preferred_device(DeviceStrategy.MEMORY)
            self._intrinsics = ImageIntrinsics(device)

    def _init_supersampler(self) -> None:
        """Lazily load Swin2SR for reference-patch supersampling — only if actually used."""
        if self._samp is None:
            device, _ = preferred_device(DeviceStrategy.MEMORY)
            self._samp = ImageSupersampling(device)

    # Each secondary helper below (LaMa, Florence-2, IntrinsicDiffusion, Swin2SR) is only
    # needed for one narrow step per region, sequentially -- unlike self._inpainter (the
    # Pass 2/seam-fix FLUX+LoRA pipeline), which is genuinely needed for every region and
    # stays loaded for the whole run. Closing each of these the moment its step finishes,
    # instead of caching it for the rest of the run like self._inpainter, keeps at most one
    # of them resident in memory alongside the always-loaded FLUX pipeline rather than all
    # four accumulating on top of it across the region loop -- the latter is what has caused
    # OOMs in the past. The _init_* methods above make re-creating one on the next region (or
    # the next call to _generate_tileable_tile_high_fidelity) a no-op cost beyond the reload
    # itself.

    def _close_lama_inpainter(self) -> None:
        if self._lama_inpainter is not None:
            self._lama_inpainter.close()
            self._lama_inpainter = None

    def _close_captioner(self) -> None:
        if self._captioner is not None:
            self._captioner.close()
            self._captioner = None

    def _close_intrinsics(self) -> None:
        if self._intrinsics is not None:
            self._intrinsics.close()
            self._intrinsics = None

    def _close_supersampler(self) -> None:
        if self._samp is not None:
            self._samp.close()
            self._samp = None

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
        pattern_ctx = self._pattern_texture_context(context, cfg) if cfg.use_pattern_texture else None

        # Real-content base layer for every region type's pattern texture,
        # computed once and shared (doesn't depend on region_val) -- see
        # bake_real_layer_from_mesh's own docstring for why this projects
        # from the terrain mesh's own (already correct, cached-preferring)
        # per-vertex panorama UV instead of re-deriving position/height from
        # a synthetic top-down world grid. `reference`'s panorama is the same
        # PANORAMA_TERRAIN TerrainMeshStage built panorama_uv against (see
        # that stage's own SemanticKey.PANORAMA resolution), so this stays in
        # the same coordinate space pattern_ctx["panorama_uv"] was computed in.
        real_everywhere = None
        if pattern_ctx is not None and reference is not None:
            real_everywhere = bake_real_layer_from_mesh(
                pattern_ctx["vertices"], pattern_ctx["faces"], pattern_ctx["panorama_uv"],
                reference[0], cfg.pattern_canvas_size, pattern_ctx["half"], pattern_ctx["half"],
            )

        tiles: dict[str, PIL.Image.Image] = {}
        tile_factors: dict[str, float] = {}
        try:
            for idx, rt in enumerate(present_types):
                pattern_layer = None
                if pattern_ctx is not None and reference is not None:
                    panorama, type_map, depth_arr, exclude_mask = reference
                    ref_patches = self._extract_reference_patches(
                        panorama, type_map, int(rt),
                        patch_size=cfg.pattern_reference_patch_size,
                        num_patches=cfg.pattern_num_reference_patches,
                        depth=depth_arr if cfg.use_point_cloud_unwarp else None,
                        exclude_mask=exclude_mask,
                    )
                    if ref_patches:
                        pattern_layer = self._synthesize_pattern_layer(
                            pattern_ctx, real_everywhere, int(rt), ref_patches, cfg, seed_offset=idx,
                        )

                if pattern_layer is not None:
                    tiles[rt.label] = pattern_layer
                    tile_factors[rt.label] = 1.0  # one-shot world-space bake, not a repeating tile
                    self.log_info(f"Generated {rt.label} layer via real-material pattern texturing")
                    self.advance_progress(task)
                    self.advance_progress(task)
                    continue

                # Last-resort fallback: no usable real reference for this region type
                # (or pattern texturing disabled / mesh not yet available) -- today's
                # FLUX generation path, unchanged.
                prompt = self._build_prompt(rt, caption)
                if reference is not None:
                    panorama, type_map, depth_arr, exclude_mask = reference
                    tiles[rt.label] = self._generate_tileable_tile_high_fidelity(
                        prompt, cfg, seed_offset=idx,
                        panorama=panorama,
                        type_map=type_map,
                        region_val=int(rt),
                        depth=depth_arr if cfg.use_point_cloud_unwarp else None,
                        exclude_mask=exclude_mask,
                        debug_label=rt.label,
                    )
                else:
                    tiles[rt.label] = self._generate_tileable_tile(
                        prompt, cfg, seed_offset=idx, debug_label=rt.label,
                    )
                tile_factors[rt.label] = cfg.synthetic_tile_factor
                self.log_info(
                    f"Generated {rt.label} tile ({cfg.tile_size}px, seamless"
                    f"{', photo-referenced (largest-region crop)' if reference is not None else ''})"
                )
                self.advance_progress(task)
                self.advance_progress(task)
        finally:
            self._inpainter.close()
            self._inpainter = None
            # Safety net -- each of these is normally already closed right after its one
            # narrow use per region (see the _close_* helpers), not held for the whole loop.
            self._close_lama_inpainter()
            self._close_intrinsics()
            self._close_supersampler()
            self._close_captioner()

        material = self._build_material(context, cfg, present_types, tiles, region_map_depth, tile_factors)

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

        self._texture_formations(context, cfg)

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    # ── Formation meshes ─────────────────────────────────────────────────────

    def _texture_formations(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
    ) -> None:
        """
        Bake and apply each non-primary ground component's own texture (see
        TerrainMeshStage / ContextKey.TERRAIN_FORMATIONS) directly onto its
        mesh's own UVs -- unlike the base terrain, a formation mesh has no
        separately-transmitted SplatMaterial to fall back on (confirmed
        against the Unity client: mesh/material association is name-matched
        to "terrain" specifically), so its final texture has to be baked
        into its own GLB, the same standard glTF texturing every other
        object mesh in this codebase already uses.

        A formation is, by construction, made entirely of directly-observed
        cells (component membership in
        HeightMapGenerator._label_ground_components requires real height
        data) -- essentially never the nadir hole, horizon band, or an
        occlusion gap a synthesized/library layer exists for. So this skips
        the library/assignment machinery entirely and just bakes real,
        dewarped, delit photo content (pattern_texture.bake_real_layer),
        the same operation the main terrain's own panorama layer performs.
        """
        formations = context.input_object(ContextKey.TERRAIN_FORMATIONS)
        if not formations:
            return

        panorama_terrain = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        height_map_params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        if panorama_terrain is None or height_map_depth is None:
            self.log_warning("Formations present but no panorama/height map — leaving them untextured")
            return
        grid_size = (height_map_params.get("grid_size_meters") if height_map_params else None) or 100.0
        terrain_half = grid_size / 2.0

        sky_mask = context.input_sky_mask()

        # Panorama-space region typing of the same image being sampled, used to
        # keep tree-line foliage out of a rock formation's bake -- see
        # vegetation_bleed_mask for why this is done on the sampled result rather
        # than by correcting the sampling angle.
        region_type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        region_type_map = region_type_depth.depth if region_type_depth is not None else None

        for i, formation in enumerate(formations):
            mesh = context.mesh(formation["mesh_key"])
            if mesh is None:
                continue

            exclude_mask = None
            if region_type_map is not None:
                exclude_mask = vegetation_bleed_mask(
                    panorama_terrain, height_map_depth.depth, region_type_map, terrain_half,
                    formation["x_half"], formation["z_half"], cfg.pattern_canvas_size,
                    formation["x_center"], formation["z_center"],
                )
                if exclude_mask is None:
                    self.log_info(
                        f"  formation {formation['id']}: mostly vegetation, keeping foliage as-is"
                    )

            layer = bake_real_layer(
                panorama_terrain, height_map_depth.depth, terrain_half,
                formation["x_half"], formation["z_half"], cfg.pattern_canvas_size,
                formation["x_center"], formation["z_center"],
                sky_mask=sky_mask,
                exclude_mask=exclude_mask,
            )
            tile = PIL.Image.fromarray((layer.clip(0.0, 1.0) * 255.0).astype("uint8"), "RGB")
            if cfg.use_intrinsic_delighting:
                tile = self._delight_patch(tile, cfg)

            TerrainMeshGenerator.apply_component_texture(
                mesh, tile,
                formation["x_center"], formation["z_center"],
                formation["x_half"], formation["z_half"],
            )
            context.add_mesh(formation["mesh_key"], mesh)

            self.log_info(f"Textured formation {formation['id']} ({formation['mesh_key']})")
            if self.temp is not None:
                tile.save(self.temp / f"terrain_formation_{formation['id']}_texture.png")

        self._close_intrinsics()

    # ── Material construction ─────────────────────────────────────────────────

    def _build_material(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
        present_types: list[RegionType],
        tiles: dict[str, PIL.Image.Image],
        region_map_depth,
        tile_factors: Optional[dict[str, float]] = None,
    ) -> SplatMaterial:
        """
        Assemble the SplatMaterial with an optional panorama layer prepended.

        When PANORAMA_TERRAIN and HEIGHT_MAP are both available and
        use_panorama_layer is True, the real (unbaked, equirect) panorama is
        inserted as layer 0 with a viewing-latitude visibility blend weight
        (see _panorama_visibility_weight) that keeps it dominant across
        almost the whole terrain. The synthetic region layer weights are
        scaled by (1 − panorama_weight) so all weights sum to 1 everywhere.

        tile_factors: per-label UV tiling factor override. A pattern-texture
        layer (see _synthesize_pattern_layer) is a one-shot world-space bake
        covering the whole terrain, not a small repeating tile, so it needs
        tile_factor=1.0 like the panorama layer; a FLUX-generated layer keeps
        cfg.synthetic_tile_factor. Falls back to cfg.synthetic_tile_factor
        for any label not present (e.g. if called by older/other code).
        """
        tile_factors = tile_factors or {}
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
        if panorama_terrain is not None:
            # Real ground colour where the removal mask ate the ground rather
            # than an occluder -- see _ground_colour_panorama.
            panorama_terrain = self._ground_colour_panorama(context, panorama_terrain)
        height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        height_map_params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)

        if panorama_terrain is not None and height_map_depth is not None:
            grid_size = (height_map_params.get("grid_size_meters") if height_map_params else None) or 100.0
            half = grid_size / 2.0

            pano_tile = self._panorama_tile(panorama_terrain, cfg.panorama_layer_max_resolution)
            # The ORIGINAL-photo typing, not the terrain-scoped one, because this has
            # to describe the pixels this layer actually paints. _ground_colour_panorama
            # above puts the original photograph's pixels back wherever foreground
            # removal ate the ground -- which in a ground-level capture is the entire
            # near field -- so the tile carries the real, upright wildflower meadow even
            # though the terrain panorama has fabricated gravel there and types it
            # GROUND. Asking the terrain pass gives 100% "ground" across the whole 0-5 m
            # disc and gates nothing; asking the original pass gives 1%, which is the
            # truth about what is being painted.
            upright_region_depth = (
                context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
                if cfg.panorama_upright_gate else None
            )
            if cfg.panorama_upright_gate and upright_region_depth is None:
                self.log_warning(
                    "panorama_upright_gate is on but PANORAMA_REGION_TYPE_MAP "
                    "is not in context — the panorama layer will keep painting upright "
                    "vegetation onto the ground plane"
                )
            pano_weight = self._panorama_visibility_weight(
                height_map_depth.depth, half, cfg.blend_map_size, cfg.panorama_blend_power,
                nadir_cutoff_deg=cfg.nadir_cutoff_deg,
                nadir_fade_deg=cfg.nadir_fade_deg,
                horizon_fade_deg=cfg.horizon_fade_deg,
                region_type_map=(
                    upright_region_depth.depth if upright_region_depth is not None else None
                ),
                upright_feather_frac=cfg.panorama_upright_feather_frac,
                upright_full_radius_m=cfg.panorama_upright_full_radius_m,
                upright_falloff_radius_m=cfg.panorama_upright_falloff_radius_m,
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
                    tile_factor=tile_factors.get(rt.label, cfg.synthetic_tile_factor),
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
                    tile_factor=tile_factors.get(rt.label, cfg.synthetic_tile_factor),
                    smoothness=_LAYER_SMOOTHNESS.get(rt, 0.1),
                )
                for rt in present_types if rt.label in tiles
            ]

        if not layers:
            label, tile = next(iter(tiles.items()))
            return SplatMaterial.from_single_layer(label, tile, cfg.blend_map_size)

        return SplatMaterial.from_weight_maps(layers=layers, weight_maps=weight_maps)

    def _ground_colour_panorama(self, context: PipelineContext, panorama_terrain):
        """PANORAMA_TERRAIN with real ground colour restored from the original photo.

        PANORAMA_TERRAIN is the right *geometry* source -- occluders removed, so
        nothing punches a hole in the ridgeline or anchors a fake crest -- and it
        is what the height/region chain is built against. It is the wrong
        *colour* source, because PanoramaForegroundInpaintingStage's removal test
        is `depth < foreground_distance_m`, and in a ground-level capture that
        catches the ground itself across the whole lower hemisphere: on the
        Rainier capture it replaced a wildflower meadow with fabricated gravel,
        which the panorama layer (dominant across almost the entire terrain, see
        _panorama_visibility_weight) then painted over everything the user walks
        on.

        So this takes the original panorama's own pixels back wherever all three
        hold:
          - the ORIGINAL panorama's region typing calls it vegetation or ground
            (it is a photograph of the surface, not of something standing on it),
          - it is below the horizon (anything rising above eye level is an
            occluder or a distant landform, neither of which is ground here),
          - PANORAMA_FOREGROUND_OCCLUDER_MASK does not cover it (that is the
            genuine tree/building removal, which must stay removed -- restoring
            it would put canopy colour on the ground the canopy was hiding).

        Returns panorama_terrain unchanged if the inputs for that test aren't
        available, which is the pre-existing behaviour.
        """
        type_map_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        panorama_original = context.input_panorama(ContextKey.PANORAMA)
        if type_map_depth is None or panorama_original is None:
            return panorama_terrain

        original = panorama_original.image.convert("RGB")
        terrain = panorama_terrain.image.convert("RGB")
        if original.size != terrain.size:
            self.log_info(
                f"Original panorama {original.size} != terrain panorama {terrain.size}; "
                "keeping the object-removed panorama as the colour source"
            )
            return panorama_terrain

        type_map = type_map_depth.depth
        width, height = terrain.size
        if type_map.shape != (height, width):
            type_map = np.array(PIL.Image.fromarray(type_map.astype(np.uint8)).resize(
                (width, height), PIL.Image.NEAREST,
            ))

        restore = np.isin(type_map, [int(RegionType.VEGETATION), int(RegionType.GROUND)])
        restore[: height // 2, :] = False

        occluder_image = context.input_image(ContextKey.PANORAMA_FOREGROUND_OCCLUDER_MASK)
        if occluder_image is not None:
            occluder = np.array(occluder_image.image.convert("L"))
            if occluder.shape != (height, width):
                occluder = np.array(PIL.Image.fromarray(occluder).resize(
                    (width, height), PIL.Image.NEAREST,
                ))
            restore &= occluder <= 127

        if not restore.any():
            return panorama_terrain

        # Feather the boundary -- a hard switch between a real photo and
        # ObjectClear's fill shows up as a visible outline once it is baked into
        # the terrain texture and magnified across several metres of ground.
        alpha = PIL.Image.fromarray((restore * 255).astype(np.uint8), "L").filter(
            PIL.ImageFilter.GaussianBlur(6)
        )
        self.log_info(
            f"Ground colour: restored real photo over {restore.mean():.1%} of the panorama "
            f"({restore[height // 2:].mean():.1%} of the lower half)"
        )
        return Panorama(Image(PIL.Image.composite(original, terrain, alpha)))

    @staticmethod
    def _panorama_tile(panorama, max_resolution: int) -> PIL.Image.Image:
        """
        Prepare the full equirectangular panorama as a texture for storage.

        The full image is kept (not cropped to the lower half) so mountain detail
        near and above the horizon is preserved.  The shader uses standard equirect
        UV projection:
            U = atan2(X, Z) / (2π) + 0.5
            V = 0.5 − φ / π   where φ = atan2(Y, √(X²+Z²))
        Mountains at φ > 0 (above camera level) sit at V < 0.5 in the tile.
        Terrain below the horizon sits at V > 0.5.  Nadir maps to V = 1.0.

        Unlike the synthetic FLUX tiles, this layer is just a resize of a real
        photo with no generation-resolution constraint, so it keeps its native
        ~2:1 equirect aspect ratio (U and V are sampled independently in [0,1]
        anyway — a square texture was never required) and is only downscaled if
        its longest edge exceeds max_resolution. This is also what the
        ground_point_cloud.glb debug export samples color from (at a pixel
        stride, but no resize) — matching resolution here keeps the mesh's
        real-photo layer as close to that reference as reasonable.
        """
        img = panorama.image.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_resolution / max(w, h))
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), PIL.Image.LANCZOS)
        return img

    @staticmethod
    def _panorama_visibility_weight(
        height_map: np.ndarray,
        half: float,
        blend_map_size: int,
        blend_power: float,
        nadir_cutoff_deg: float = -85.0,
        nadir_fade_deg: float = 4.0,
        horizon_fade_deg: float = 1.5,
        region_type_map: Optional[np.ndarray] = None,
        upright_feather_frac: float = 0.01,
        upright_full_radius_m: float = 6.0,
        upright_falloff_radius_m: float = 25.0,
    ) -> np.ndarray:
        """
        Per-texel visibility weight for the panorama layer, based on the
        viewing latitude from the camera (at the world origin), and on whether
        the panorama pixel each ground texel would sample is actually GROUND.

        An earlier version weighted by surface-normal facing instead, but for
        typical rolling/flat terrain viewed from a ~1-2 m camera height, the
        facing dot product collapses to near-zero within a few metres of the
        camera (it decays as camera_height / distance) — so nearly the whole
        terrain fell back to synthetic FLUX tiles instead of the real photo.

        Latitude-based weighting (delegated to Panorama.mesh_visibility_weight,
        the one shared implementation of this ramp) keeps the real panorama
        dominant across the terrain.
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

        This used to also gate on HEIGHT_MAP_OBSERVED_MASK, dropping weight
        wherever a cell had no direct point-cloud measurement, on the theory
        that unobserved terrain's height is unreliable enough to misdirect a
        world-position-based panorama lookup. That reasoning no longer
        applies: the shader doesn't do a live position-based lookup for this
        layer any more (see TerrainSplat.shader's LayerUV/panoUV) — it
        samples the server-baked panoUV (TEXCOORD_1), which already prefers
        each vertex's own true observed panorama pixel and only falls back to
        a position-derived one where there's no observation, so the
        uncertainty this gate was defending against is already handled at
        the UV level, once, upstream. Re-gating here on top of that just
        threw away most of the real photo for no remaining benefit.

        The latitude ramp alone is not enough, and this is the failure it misses.
        Projecting an equirect panorama onto a horizontal ground plane assumes the
        content at elevation phi IS the ground at radius h/tan(-phi). That holds for
        bare terrain and fails completely for anything UPRIGHT: a 0.5 m flower spans
        a range of elevations, and each of those elevations lands at a different
        ground radius, so one flower is smeared into a streak metres long. Rendering
        the baked result top-down on the Rainier capture gives a textbook radial
        pinwheel -- the near field is a starburst of stretched wildflowers, and by
        20 m the mid-ground conifers have become huge dark spokes crossing everything.
        It is not confined to the nadir: it happens wherever the panorama holds
        vertical structure, which in a meadow-and-forest scene is nearly everywhere.
        Meanwhile the panorama layer's weight was 1.00 across the whole 0-5 m disc,
        so pattern_texture.py -- which exists precisely to supply real ground material
        without any projective smear -- contributed exactly nothing where the viewer
        actually looks.

        The scene also places those same trees and flowers as real 3D instances on
        top of this terrain, so a smeared copy baked into the ground is not just
        wrong, it is double-counted.

        RegionType.ground_valid already encodes exactly the needed distinction, for
        exactly this reason ("Vegetation canopies and built structures occlude the
        ground"). So where a texel's own panorama pixel is VEGETATION or BUILT, the
        panorama is not showing that texel's ground and its weight is dropped,
        handing the texel to the synthetic/pattern layers. Feathered rather than
        binary, so the handover doesn't produce a hard-edged stencil of the canopy.

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

        grid_verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        weight = Panorama.mesh_visibility_weight(
            grid_verts,
            nadir_cutoff_deg=nadir_cutoff_deg,
            nadir_fade_deg=nadir_fade_deg,
            horizon_fade_deg=horizon_fade_deg,
        ).reshape(blend_map_size, blend_map_size)

        if region_type_map is not None:
            # Same projection HeightMapGenerator._panorama_uv_from_height and
            # Panorama.mesh_uvs use, so this asks about the very pixel the shader
            # will sample for this texel -- not an approximation of it.
            lon = np.arctan2(X, Z)
            r_xz = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
            lat = np.arctan2(Y, r_xz)
            u = (lon + np.pi) / (2.0 * np.pi)
            v = 0.5 + lat / np.pi

            rt_h, rt_w = region_type_map.shape
            # Row from (1 - v): mesh_uvs stores V flipped for glTF, so v = 0 is the
            # bottom of the image, not the top.
            rt_row = ((1.0 - v) * (rt_h - 1)).clip(0, rt_h - 1)
            rt_col = (u * (rt_w - 1)).clip(0, rt_w - 1)
            sampled = map_coordinates(
                region_type_map, [rt_row.ravel(), rt_col.ravel()], order=0, mode="nearest"
            ).reshape(blend_map_size, blend_map_size)

            # VEGETATION/BUILT only, rather than "everything not ground_valid".
            # SKY deliberately does not gate here: Panorama.mesh_uvs already snaps a
            # mountainside vertex's UV down off sky to the nearest real content in its
            # own column, so a texel this function believes samples sky usually does
            # not. Penalising it here would strip the real photo off exactly the
            # distant terrain that snap exists to keep -- an error in this estimate,
            # not a defect in the texture.
            upright = np.isin(
                np.rint(sampled).astype(np.int32),
                [int(RegionType.VEGETATION), int(RegionType.BUILT)],
            ).astype(np.float32)

            # Distance-weighted, because the artifact is: the panorama layer is a
            # view-dependent texture that is EXACT from the capture point and decays
            # as the viewer leaves it. A radial smear pointing away from the origin is
            # foreshortened to nothing when seen from the origin, so what makes it
            # visible is parallax from the viewer's own movement -- roughly d/r in
            # angular terms for a head movement d. At the ~2 m of movement this scene
            # is built for that is 40% at 5 m and 4% at 50 m. The smear itself is
            # proportional to radius (dr = r*v/(h-v) for an upright object of height v
            # under a camera h above ground, so dr/r is constant), but its visibility
            # is not, and stripping the far field would cost real photographic detail
            # to fix something nobody can see.
            if upright_falloff_radius_m > upright_full_radius_m:
                t = (r_xz - upright_full_radius_m) / (upright_falloff_radius_m - upright_full_radius_m)
                strength = np.clip(1.0 - t, 0.0, 1.0).astype(np.float32)
            else:
                strength = (r_xz <= upright_full_radius_m).astype(np.float32)

            sigma = max(1.0, upright_feather_frac * blend_map_size)
            keep = np.clip(gaussian_filter(1.0 - upright, sigma=sigma), 0.0, 1.0)
            # strength 1 -> the gate applies in full; strength 0 -> weight unchanged.
            weight = weight * (1.0 - strength * (1.0 - keep))

        return (weight ** blend_power).astype(np.float32)

    # ── Photo reference (weak img2img seed) ─────────────────────────────────────

    def _bake_reference(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
    ) -> Optional[tuple[Panorama, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Fetch the real photo, its panorama-space region typing, and its
        aligned depth, for use as a weak img2img seed per region tile.
        Returns (panorama, type_map, depth, exclude_mask), or None if the
        source imagery is missing. depth is None if PANORAMA_DEPTH isn't
        available (callers fall back to plain perspective_crop for reference
        patches in that case).

        exclude_mask (bool, True = don't crop reference patches from here) is
        PANORAMA_FOREGROUND_MASK — ObjectClear's own dilated removal mask —
        resized onto panorama_terrain's grid. panorama_terrain has that same
        region filled in with fabricated (if plausible) content, not a real
        photograph; it's there so the *region classification and geometry*
        chain has a with-objects-removed version to work from, not so that
        fill can stand in as ground truth. Left unfiltered, that fabricated
        region is often the single most homogeneous ("solid") patch of
        whatever it got inpainted as, so _extract_reference_patches's own
        distance-transform peak search would preferentially pick it over
        genuine, more textured real photo content elsewhere in the same
        region type — the exact way a real wildflower meadow removed as
        foreground ended up replaced by flat inpainted dirt as the terrain's
        actual reference texture.

        Deliberately does *not* reproject the whole scene to a top-down/
        orthographic grid — that reprojection collapses to a radial/pinwheel
        singularity right around the camera (the single largest "ground"
        blob is almost always the area immediately around it) and stretches
        distant ground into blocky artefacts near the horizon. Reference
        crops are instead cut straight out of the panorama's own
        equirectangular pixels (_extract_reference_patches), then flattened
        per-crop from that one crop's own real depth points
        (Panorama.unwarp_box) — local in scope, so it never touches the
        camera-adjacent singularity that broke the whole-scene version.
        """
        panorama_terrain = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        type_map_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        if panorama_terrain is None or type_map_depth is None:
            return None
        # Reference patches are what every synthetic layer's material is built
        # from, so they need the same real-ground colour restoration the panorama
        # layer gets -- otherwise the fallback material for the ground is cut from
        # the fabricated gravel this exists to stop using.
        panorama_terrain = self._ground_colour_panorama(context, panorama_terrain)
        depth_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        depth_arr = (
            depth_depth.depth
            if depth_depth is not None and depth_depth.depth.shape == (panorama_terrain.height, panorama_terrain.width)
            else None
        )

        exclude_mask = None
        foreground_mask_img = context.input_image(ContextKey.PANORAMA_FOREGROUND_MASK)
        if foreground_mask_img is not None:
            mask_arr = np.array(foreground_mask_img.image.convert("L"))
            if mask_arr.shape != type_map_depth.depth.shape:
                mask_arr = np.array(
                    PIL.Image.fromarray(mask_arr).resize(
                        (type_map_depth.depth.shape[1], type_map_depth.depth.shape[0]),
                        PIL.Image.NEAREST,
                    )
                )
            exclude_mask = mask_arr > 127

        return panorama_terrain, type_map_depth.depth, depth_arr, exclude_mask

    def _pattern_texture_context(
        self,
        context: PipelineContext,
        cfg: "TerrainTextureGenerationConfiguration",
    ) -> Optional[dict]:
        """
        Precompute what every region type's pattern-texture layer needs, once,
        shared across all of them: the actual terrain mesh (so the triangles
        used for library assignment are exactly the render mesh's own, per
        the pattern-texture plan -- no separate texture-mesh resolution to
        keep in sync), which of its triangles the panorama layer already
        rejects (needs_synthesis -- reusing _panorama_visibility_weight's own
        certainty formula, evaluated at each triangle's centroid instead of a
        blend-map grid, so the two layers never fight over the same ground),
        and which region type each triangle belongs to (top-down REGION_MAP
        sampled at the same centroids).

        Returns None if the terrain mesh, height map, or region map isn't
        available yet (e.g. Terrain Mesh stage was skipped) -- callers fall
        back to FLUX.
        """
        terrain_mesh = context.mesh(ContextKey.TERRAIN_MESH)
        height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        height_map_params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        if terrain_mesh is None or height_map_depth is None or region_map_depth is None:
            return None
        # extra_uv (TEXCOORD_1): the same already-correct, cached-preferring,
        # per-vertex panorama UV TerrainMeshGenerator.generate() computed once
        # for the live panorama layer -- see bake_real_layer_from_mesh, which
        # reuses it directly instead of re-deriving a second, less trustworthy
        # position/height-based projection of its own. No panorama was ever
        # supplied to generate() if this is None (e.g. no PANORAMA_TERRAIN),
        # in which case there's nothing this whole module can usefully bake.
        panorama_uv = getattr(terrain_mesh, "extra_uv", None)
        if panorama_uv is None:
            return None

        grid_size = (height_map_params.get("grid_size_meters") if height_map_params else None) or 100.0
        half = grid_size / 2.0

        vertices = np.asarray(terrain_mesh.mesh.vertices, dtype=np.float64)
        faces = np.asarray(terrain_mesh.mesh.faces, dtype=np.int64)
        if len(faces) == 0 or len(panorama_uv) != len(vertices):
            return None
        centroids = vertices[faces].mean(axis=1)  # (M, 3) world XYZ

        pano_weight_grid = self._panorama_visibility_weight(
            height_map_depth.depth, half, cfg.blend_map_size, cfg.panorama_blend_power,
            nadir_cutoff_deg=cfg.nadir_cutoff_deg,
            nadir_fade_deg=cfg.nadir_fade_deg,
            horizon_fade_deg=cfg.horizon_fade_deg,
        )
        row_c = ((centroids[:, 2] + half) / (2.0 * half) * (cfg.blend_map_size - 1)).clip(0, cfg.blend_map_size - 1)
        col_c = ((centroids[:, 0] + half) / (2.0 * half) * (cfg.blend_map_size - 1)).clip(0, cfg.blend_map_size - 1)
        face_pano_weight = map_coordinates(pano_weight_grid, [row_c, col_c], order=1, mode="nearest")
        face_needs_synthesis = face_pano_weight < cfg.pattern_certainty_threshold

        rm = region_map_depth.depth
        rm_h, rm_w = rm.shape
        row_rm = ((centroids[:, 2] + half) / (2.0 * half) * (rm_h - 1)).clip(0, rm_h - 1)
        col_rm = ((centroids[:, 0] + half) / (2.0 * half) * (rm_w - 1)).clip(0, rm_w - 1)
        face_region_idx = map_coordinates(rm, [row_rm, col_rm], order=0, mode="nearest").astype(np.int32)

        self.log_info(
            f"Pattern texture: {int(face_needs_synthesis.sum())}/{len(faces)} triangles "
            f"need synthesis (panorama weight < {cfg.pattern_certainty_threshold:.2f})"
        )

        return {
            "vertices": vertices,
            "faces": faces,
            "face_needs_synthesis": face_needs_synthesis,
            "face_region_idx": face_region_idx,
            "panorama_uv": panorama_uv,
            "half": half,
        }

    def _synthesize_pattern_layer(
        self,
        pattern_ctx: dict,
        real_everywhere: np.ndarray,
        region_val: int,
        ref_patches: list[PIL.Image.Image],
        cfg: "TerrainTextureGenerationConfiguration",
        seed_offset: int,
    ) -> Optional[PIL.Image.Image]:
        """
        Delight each real reference crop (same operation the FLUX path's own
        patches go through -- see _delight_patch) before handing them to
        pattern_texture.build_library, so library samples match the same
        flat-lighting convention as the panorama layer's own observed-mask-
        gated content and don't visually clash at the boundary between them
        ("use the un-lit panorama, not hallucination" applies here too).
        """
        delit_patches = (
            [self._delight_patch(p, cfg) for p in ref_patches]
            if cfg.use_intrinsic_delighting else ref_patches
        )

        face_region_mask = pattern_ctx["face_region_idx"] == region_val
        return synthesize_region_layer(
            vertices=pattern_ctx["vertices"],
            faces=pattern_ctx["faces"],
            face_needs_synthesis=pattern_ctx["face_needs_synthesis"],
            face_region_mask=face_region_mask,
            reference_patches=delit_patches,
            real_everywhere=real_everywhere,
            x_half=pattern_ctx["half"],
            z_far=pattern_ctx["half"],
            canvas_size=cfg.pattern_canvas_size,
            seed=cfg.seed + seed_offset,
            sample_res=cfg.pattern_sample_res,
            border_width_frac=cfg.pattern_border_width_frac,
            feather_band_px=cfg.pattern_feather_band_px,
            feather_sigma_px=cfg.pattern_feather_sigma_px,
            min_coverage_fraction=cfg.pattern_min_coverage_fraction,
        )

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
        return self._material_prompt(base)

    def _material_prompt(self, description: str) -> str:
        if not self.config.use_lora:
            # No LoRA loaded -- its "smlstxtr"/"seamless texture" trigger wording is
            # meaningless (and potentially confusing) to plain FLUX.1-Fill-dev.
            return f"{description}{_TILE_SUFFIX}"
        lora = PanoramaLoraType.FLUX_SEAMLESS_TEXTURE
        return f"{lora_prompt_prefix(lora)}{description}{_TILE_SUFFIX}{lora_prompt_suffix(lora)}"

    def _describe_material(self, patch: PIL.Image.Image) -> Optional[str]:
        """
        Caption the real reference crop with Florence-2 to get a concrete
        material description grounded in what's actually in the photo (e.g.
        "a close-up of cracked, sun-bleached mud with small pebbles") instead
        of guessing from the coarse region label alone.

        patch may be RGBA (see _extract_reference_patches): ImageCaptioning's
        own alpha-aware conversion fills anything outside the region mask
        with the mean colour of the region itself before captioning, so
        neighbouring materials at the crop's edges don't leak into the
        description.

        Returns None (caller falls back to the generic per-region prompt) if
        there's no usable text or captioning fails outright.
        """
        self._init_captioner()
        try:
            text = self._captioner.caption(Image(patch)).strip()
        except Exception as exc:
            self.log_info(f"Material captioning failed, falling back to generic prompt: {exc}")
            return None
        return text.rstrip(". ") or None

    def _delight_patch(
        self,
        patch: PIL.Image.Image,
        cfg: "TerrainTextureGenerationConfiguration",
    ) -> PIL.Image.Image:
        """
        Blend `patch`'s RGB toward IntrinsicDiffusion's predicted albedo (delit,
        lighting-free reflectance) by cfg.intrinsic_delight_strength, keeping its
        own region-mask alpha channel (see _extract_reference_patches) untouched.
        Falls back to the raw patch if the model errors on this particular crop.
        """
        self._init_intrinsics()
        try:
            result = self._intrinsics.intrinsic_images(
                Image(patch.convert("RGB")),
                temp_path=self.temp,
                resolution=cfg.intrinsic_resolution,
                agg_num=cfg.intrinsic_agg_num,
            )
            albedo = result.albedo_image()
        except Exception as exc:
            self.log_info(f"IntrinsicDiffusion delighting failed, using raw patch: {exc}")
            return patch

        strength = cfg.intrinsic_delight_strength
        if strength < 1.0:
            albedo_rgb = np.asarray(albedo.convert("RGB"), dtype=np.float32)
            original_rgb = np.asarray(patch.convert("RGB"), dtype=np.float32)
            blended = albedo_rgb * strength + original_rgb * (1.0 - strength)
            albedo = PIL.Image.fromarray(blended.clip(0, 255).astype(np.uint8), "RGB")

        if patch.mode == "RGBA":
            albedo = albedo.convert("RGBA")
            albedo.putalpha(patch.split()[-1])
        return albedo

    def _supersample_patch(
        self,
        patch: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Swin2SR 2x `patch`'s RGB, then LANCZOS it back down to its own original
        size -- the same "render bigger, downsample for quality" trick used for
        Flux fills elsewhere (see _supersample_flux in panorama_inpainting/
        generation.py). Run right before compositing so the sharper, denoised
        result -- not the raw (optionally delit) patch -- is what gets pasted
        into the Pass 1 canvas and what the canvas's own background fill
        (_build_reference_canvas, which LANCZOS-stretches a copy of one patch up
        to the full tile) stretches from. Keeps the patch's own alpha (region
        mask) and size unchanged, so placement/canvas geometry is unaffected.
        Falls back to the input patch if supersampling errors on this crop.
        """
        self._init_supersampler()
        size = patch.size
        try:
            hq = self._samp.supersample(Image(patch.convert("RGB")), self.temp).image
        except Exception as exc:
            self.log_info(f"Patch supersampling failed, using un-supersampled patch: {exc}")
            return patch
        result = hq.resize(size, PIL.Image.LANCZOS)
        if patch.mode == "RGBA":
            result = result.convert("RGBA")
            result.putalpha(patch.split()[-1])
        return result

    def _generate_tileable_tile(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
        inpainter: Optional[InPainting] = None,
        debug_label: str = "tile",
    ) -> PIL.Image.Image:
        """
        Pure text-to-image tile generation for regions with no usable photo
        reference at all (either the whole scene has none, or this specific
        region has no trustworthy pixels in the baked photo).

        inpainter: override the stage's own self._inpainter. Defaults to
        self._inpainter.
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
        self._save_debug_step(cfg, debug_label, "05_pass2_raw", generated)

        result_img = self._seam_fix(generated, cfg, prompt, seed, inpainter=inpainter)
        self._save_debug_step(cfg, debug_label, "06_pass2_seamfixed", result_img)
        return self._pack_local_height_channel(result_img)

    def _generate_tileable_tile_high_fidelity(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
        panorama: Panorama,
        type_map: np.ndarray,
        region_val: int,
        depth: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
        inpainter: Optional[InPainting] = None,
        debug_label: str = "tile",
    ) -> PIL.Image.Image:
        """
        Two-pass tile generation guided by a few real crops of this region cut
        directly from the panorama's own pixels (_extract_reference_patches)
        — each a single contiguous, coherently-lit location — tiled edge to
        edge across the tile's interior (_build_reference_canvas) so Pass 1
        only has to genuinely outpaint a thin border ring, not extrapolate
        most of the tile from one small inset patch.

        If cfg.describe_material, the first patch is also captioned
        (_describe_material) and the prompt's subject is swapped for that
        description — e.g. "a close-up of cracked, sun-bleached mud with
        small pebbles" — in place of the generic per-region-type phrase in
        _BASE_PROMPTS, so generation is grounded in what this specific crop
        actually shows rather than just its coarse region label.

        Pass 1 (_pass1_extend_patch) extends them to fill the whole tile using
        them purely as guidance, not something to hard-preserve. Pass 2
        (below) then repaints the *entire* Pass-1 tile with FLUX + the
        FLUX_SEAMLESS_TEXTURE LoRA — including the original patch regions — to
        push toward genuine tileability and the photorealistic macro finish.
        Falls back to plain text generation if the region has no pixels at all.

        type_map: panorama-space per-pixel RegionType index array (see
        ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN), used to locate crop candidates
        (see _extract_reference_patches).
        depth: optional panorama-aligned depth (ContextKey.PANORAMA_DEPTH),
        used to flatten each crop from its own real 3D points
        (Panorama.unwarp_box) instead of a plain perspective reprojection.
        None falls back to perspective_crop for every patch.
        inpainter: override the stage's own self._inpainter for Pass 2. Defaults
        to self._inpainter.
        debug_label: filename prefix for step-by-step debug images (see
        _save_debug_step), typically the region label (e.g. "ground").
        """
        pass2_inpainter = inpainter if inpainter is not None else self._inpainter
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        n = max(1, cfg.num_reference_patches)
        patch_size = max(16, int(T * cfg.reference_patch_fraction / math.sqrt(n)))
        patches = self._extract_reference_patches(
            panorama=panorama,
            type_map=type_map,
            region_val=region_val,
            patch_size=patch_size,
            num_patches=n,
            depth=depth,
            exclude_mask=exclude_mask,
        )
        if not patches:
            return self._generate_tileable_tile(prompt, cfg, seed_offset, inpainter=pass2_inpainter, debug_label=debug_label)
        for i, p in enumerate(patches):
            self._save_debug_step(cfg, debug_label, f"01_reference_patch_{i}", p)

        if cfg.use_intrinsic_delighting:
            patches = [self._delight_patch(p, cfg) for p in patches]
            for i, p in enumerate(patches):
                self._save_debug_step(cfg, debug_label, f"01b_reference_patch_delit_{i}", p)
            self._close_intrinsics()

        if cfg.use_patch_supersampling:
            patches = [self._supersample_patch(p) for p in patches]
            for i, p in enumerate(patches):
                self._save_debug_step(cfg, debug_label, f"01c_reference_patch_supersampled_{i}", p)
            self._close_supersampler()

        if cfg.describe_material:
            material_description = self._describe_material(patches[0])
            if material_description:
                prompt = self._material_prompt(material_description)
                self.log_info(f"{debug_label}: material caption -> \"{material_description}\"")
            self._close_captioner()

        pass1_image = self._pass1_extend_patch(patches, prompt, cfg, seed, debug_label)
        self._save_debug_step(cfg, debug_label, "04_pass1_result", pass1_image)

        # Pass 2: FLUX + LoRA repaints the whole tile again, full mask — free to change
        # anything, including the original patch region, since Pass 1 already gave the
        # whole canvas plausible real-texture coverage. reference_strength is the only
        # thing anchoring this to Pass 1's content rather than inventing something else.
        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = pass2_inpainter.inpaint(
            input_image=pass1_image,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.reference_strength,
            seed=seed,
        )
        self._save_debug_step(cfg, debug_label, "05_pass2_raw", generated)

        # Pass 3: detail recovery — the two inpainting passes can soften the
        # reference crop's crisp stones/grass structure, so restore local contrast.
        result_img = generated.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
        result_img = PIL.ImageEnhance.Contrast(result_img).enhance(1.15)
        result_img = PIL.ImageEnhance.Sharpness(result_img).enhance(1.20)
        self._save_debug_step(cfg, debug_label, "06_pass2_sharpened", result_img)

        # Pass 4: nudge final colour statistics back toward the Pass-1 (real-texture-
        # grounded) tile, not just the small patch, since Pass 1 now covers the whole tile.
        if cfg.color_transfer_strength > 0.0:
            result_img = lab_color_transfer(
                source=pass1_image, target=result_img, strength=cfg.color_transfer_strength,
            )
        self._save_debug_step(cfg, debug_label, "07_pass2_final", result_img)

        # Pass 5: seam fix — last, deliberately, so it repairs the actual final tile
        # (post sharpen/contrast/colour-transfer) rather than being run in the middle and
        # then potentially re-disturbed by those later passes.
        result_img = self._seam_fix(result_img, cfg, prompt, seed, inpainter=pass2_inpainter)
        self._save_debug_step(cfg, debug_label, "08_pass2_seamfixed", result_img)

        return self._pack_local_height_channel(result_img)

    def _pass1_extend_patch(
        self,
        patches: list[PIL.Image.Image],
        prompt: str,
        cfg: "TerrainTextureGenerationConfiguration",
        seed: int,
        debug_label: str,
    ) -> PIL.Image.Image:
        """
        Extend `patches` to fill a whole tile_size canvas, using them purely
        as guidance (see cfg.pass1_inpainting_type for the LAMA vs FLUX
        choice).
        """
        T = cfg.tile_size
        use_lama = cfg.pass1_inpainting_type == "LAMA"
        canvas, mask, placements = self._build_reference_canvas(
            patches, T,
            feather_px=0.0 if use_lama else cfg.mask_feather_px,
            border_fill_px=int(T * cfg.outpaint_border_fraction),
        )
        self._save_debug_step(cfg, debug_label, "02_pass1_canvas", canvas)
        self._save_debug_step(cfg, debug_label, "03_pass1_mask", mask)

        if use_lama:
            self._init_lama_inpainter()
            result = self._lama_inpainter.inpaint(input_image=canvas, mask_image=mask, temp_path=self.temp)
            # LaMa has no prompt to anchor colour/exposure and can drift away from its real
            # anchors, especially over the larger distances between multiple spread-out
            # patches -- nudge its whole fill back toward the real patches' own colour
            # statistics before compositing them back over it.
            if cfg.color_transfer_strength > 0.0:
                patches_ref = PIL.Image.fromarray(
                    np.concatenate([np.array(p.convert("RGB")) for p in patches], axis=0)
                )
                result = lab_color_transfer(source=patches_ref, target=result, strength=cfg.color_transfer_strength)
            self._close_lama_inpainter()
            # LaMa is trained to reproduce the unmasked region closely but isn't guaranteed
            # pixel-exact — composite the real patches back to be sure Pass 2 actually starts
            # from genuine material, not a LaMa approximation of it.
            return self._composite_patches_over(result, patches, placements, mask, T)

        # Plain FLUX img2img with the LoRA disabled: loosely guided by the patches (via
        # `image` + reference_strength) without hard-preserving them, and no seamless-
        # texture LoRA bias at this stage. Full mask -- nothing here is meant to be kept
        # verbatim, unlike the LaMa path where the mask is the only way to reference the
        # patches at all.
        self._inpainter.set_lora_enabled(False)
        try:
            full_mask = PIL.Image.new("L", (T, T), 255)
            return self._inpainter.inpaint(
                input_image=canvas,
                mask_image=full_mask,
                temp_path=self.temp,
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                strength=cfg.reference_strength,
                seed=seed,
            )
        finally:
            self._inpainter.set_lora_enabled(True)

    @staticmethod
    def _build_reference_canvas(
        patches: list[PIL.Image.Image],
        tile_size: int,
        feather_px: float,
        border_fill_px: int = 0,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image, list[tuple[int, int, int, int]]]:
        """
        Tile `patches` across a simple row-major grid sized to fill the
        tile's whole interior (tile_size minus a border_fill_px ring on
        every side) edge to edge, each resized up to its own cell — 1 patch
        fills the entire interior; more are tiled with no gaps between them.
        Only border_fill_px's ring is left for Pass 1 to actually generate.

        This makes Pass 1 a genuine, bounded outpaint (extend real photo
        content a little past its own edge) instead of extrapolating most of
        the tile from a small inset island of real content while marking the
        rest "generate" regardless of how plausible a background preview
        looks there — both LaMa and diffusion inpainting zero out or heavily
        denoise whatever sits under a "generate" pixel before it's ever
        conditioned on, so a bigger genuinely-real interior is the only fix
        that actually changes what the model sees, not better pixels under a
        mask it never looks at.

        The mask is black (0, keep) over each cell's genuine same-region
        pixels (from the resized patch's own alpha channel — see
        _extract_reference_patches), white (255, generate) everywhere else —
        any non-region sliver inside a cell, plus the whole outer ring —
        boundary Gaussian-feathered (when feather_px > 0) so the transition
        blends rather than leaving a hard rectangular seam. Used as-is by the
        LaMa Pass-1 path (real inpainting mask); the FLUX Pass-1 path only
        uses the canvas and builds its own full-white mask, since it never
        asks the model to hard-preserve anything.

        The outer ring's own background (before Pass 1 fills it) is a
        reflect-padded extension of the interior mosaic, not a fresh blurry
        stretch of one patch across the whole tile — relevant only to the
        FLUX Pass-1 fallback (ordinary img2img actually uses it as a
        starting point); LaMa ignores background pixels under "generate"
        entirely and reconstructs purely from context.

        Returns (canvas RGB, mask L, placements — (offset_y, offset_x, w, h)
        per patch in `patches`' own order, for _composite_patches_over to
        restore them verbatim afterwards).
        """
        T = tile_size
        n = len(patches)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        interior = max(1, T - 2 * border_fill_px)
        base_cell_w = max(1, interior // cols)
        base_cell_h = max(1, interior // rows)

        canvas_arr = np.zeros((T, T, 3), dtype=np.uint8)
        mask_arr = np.full((T, T), 255, dtype=np.uint8)
        placements: list[tuple[int, int, int, int]] = []

        for i, patch in enumerate(patches):
            r, c = divmod(i, cols)
            # Last row/column absorbs the integer-division remainder so the grid
            # exactly fills `interior` with no gap or overshoot.
            w = base_cell_w if c < cols - 1 else interior - base_cell_w * (cols - 1)
            h = base_cell_h if r < rows - 1 else interior - base_cell_h * (rows - 1)
            oy = border_fill_px + r * base_cell_h
            ox = border_fill_px + c * base_cell_w

            resized = patch.resize((w, h), PIL.Image.LANCZOS)
            canvas_arr[oy:oy + h, ox:ox + w] = np.array(resized.convert("RGB"))
            keep = (
                np.array(resized.split()[-1]) > 127
                if resized.mode == "RGBA" else np.ones((h, w), dtype=bool)
            )
            mask_arr[oy:oy + h, ox:ox + w][keep] = 0
            placements.append((oy, ox, w, h))

        if border_fill_px > 0:
            b = border_fill_px
            interior_arr = canvas_arr[b:T - b, b:T - b]
            canvas_arr = np.pad(interior_arr, ((b, b), (b, b), (0, 0)), mode="reflect")

        canvas = PIL.Image.fromarray(canvas_arr)
        mask = PIL.Image.fromarray(mask_arr, "L")
        if feather_px > 0:
            mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px))

        return canvas, mask, placements

    @staticmethod
    def _composite_patches_over(
        generated: PIL.Image.Image,
        patches: list[PIL.Image.Image],
        placements: list[tuple[int, int, int, int]],
        mask: PIL.Image.Image,
        tile_size: int,
    ) -> PIL.Image.Image:
        """
        Paste the real reference patches back over the generated tile, at the
        same grid cells and with the same feathered mask used to build the
        input canvas (_build_reference_canvas) — `1 - mask` is exactly the
        right blend weight, since `mask` is generate(255)/keep(0). This is the
        actual guarantee that the real patches survive and blend smoothly;
        see the call site for why the model's own mask handling can't be
        trusted to do this itself.
        """
        T = tile_size
        keep_weight = 1.0 - (np.array(mask, dtype=np.float32) / 255.0)
        gen_arr = np.array(generated.convert("RGB"), dtype=np.float32)
        patches_placed = np.zeros((T, T, 3), dtype=np.float32)
        for patch, (oy, ox, w, h) in zip(patches, placements):
            resized = patch.resize((w, h), PIL.Image.LANCZOS).convert("RGB")
            patches_placed[oy:oy + h, ox:ox + w] = np.array(resized, dtype=np.float32)

        blended = keep_weight[..., None] * patches_placed + (1.0 - keep_weight[..., None]) * gen_arr
        return PIL.Image.fromarray(blended.clip(0, 255).astype(np.uint8))

    def _seam_fix(
        self,
        generated: PIL.Image.Image,
        cfg: TerrainTextureGenerationConfiguration,
        prompt: str,
        seed: int,
        inpainter: Optional[InPainting] = None,
    ) -> PIL.Image.Image:
        """
        Circular-shift the tile by half its size so all four original tile
        edges land at the centre as a "+"-shaped seam, repair each arm of
        that cross separately (_seam_fix_band), then shift back.
        """
        inpainter = inpainter if inpainter is not None else self._inpainter
        T = cfg.tile_size
        arr = np.array(generated.convert("RGB"), dtype=np.uint8)
        half = T // 2
        shifted = np.roll(np.roll(arr, half, axis=0), half, axis=1)

        sw = max(2, int(T * cfg.seam_width_fraction / 2))
        shifted = self._seam_fix_band(
            shifted, axis=0, center=half, half_width=sw, cfg=cfg,
            prompt=prompt, seed=seed + 1000, inpainter=inpainter,
        )
        shifted = self._seam_fix_band(
            shifted, axis=1, center=half, half_width=sw, cfg=cfg,
            prompt=prompt, seed=seed + 1001, inpainter=inpainter,
        )

        result = np.roll(np.roll(shifted, -half, axis=0), -half, axis=1)
        return PIL.Image.fromarray(result)

    def _seam_fix_band(
        self,
        arr: np.ndarray,
        axis: int,
        center: int,
        half_width: int,
        cfg: TerrainTextureGenerationConfiguration,
        prompt: str,
        seed: int,
        inpainter: InPainting,
    ) -> np.ndarray:
        """
        Repair one arm of the seam cross -- axis=0: a horizontal band
        spanning the tile's full width at row `center`; axis=1: a vertical
        band spanning its full height at column `center`.

        Unlike inpainting the whole tile, this crops tightly around just the
        band (plus seam_fix_context_px of real surrounding pixels so FLUX has
        genuine texture to condition on, not a crop edge that reads as a hard
        image border), downscales that crop to seam_fix_max_size before
        inpainting, then LANCZOS's the result back up and feathers it into
        the original crop. Running FLUX directly at the tile's own full
        resolution on a strip this narrow tends to stamp a crisp, mismatched
        patch of new high-frequency noise right on the seam; downscaling
        first produces a smoother fill that actually blends into its real
        neighbours on both sides -- the same "crop, downscale, inpaint,
        upscale back" trick used for the skybox panorama's own wrap-seam
        repair (see skybox_inpainting's _sky_seam_inpaint /
        util.seam_repair.heal_wrap_seam's crop_context_px). Feathering the
        composite back over the original crop (rather than pasting the FLUX
        result verbatim) means real detail everywhere outside the band is
        untouched.

        arr: full tile, already rolled so the seam sits at `center`.
        Returns the tile with this one band repaired (same shape as `arr`).
        """
        T = arr.shape[0]
        context = cfg.seam_fix_context_px
        lo = max(0, center - half_width - context)
        hi = min(T, center + half_width + context)
        crop = arr[lo:hi, :] if axis == 0 else arr[:, lo:hi]
        crop_h, crop_w = crop.shape[:2]

        band_lo = (center - half_width) - lo
        band_hi = (center + half_width) - lo
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if axis == 0:
            mask[max(0, band_lo):min(crop_h, band_hi), :] = 255
        else:
            mask[:, max(0, band_lo):min(crop_w, band_hi)] = 255
        if cfg.seam_dilation_px > 0:
            mask = (binary_dilation(mask, iterations=cfg.seam_dilation_px).astype(np.uint8) * 255)

        scale = min(1.0, cfg.seam_fix_max_size / max(crop_w, crop_h))
        fw = max(16, int(round(crop_w * scale)))
        fh = max(16, int(round(crop_h * scale)))
        crop_pil = PIL.Image.fromarray(crop)
        mask_pil = PIL.Image.fromarray(mask, "L")
        flux_input = crop_pil.resize((fw, fh), PIL.Image.LANCZOS) if scale < 1.0 else crop_pil
        flux_mask = mask_pil.resize((fw, fh), PIL.Image.NEAREST) if scale < 1.0 else mask_pil

        healed = inpainter.inpaint(
            input_image=flux_input,
            mask_image=flux_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=seed,
        )
        if scale < 1.0:
            healed = healed.resize((crop_w, crop_h), PIL.Image.LANCZOS)
        healed_arr = np.array(healed.convert("RGB"), dtype=np.float32)

        alpha = mask.astype(np.float32) / 255.0
        if cfg.seam_fix_feather_px > 0:
            alpha = gaussian_filter(alpha, sigma=cfg.seam_fix_feather_px)
            # A feather this wide relative to the (deliberately narrow) band washes its
            # whole peak down below 1.0 -- renormalise back up so the seam's own centre
            # still gets the fix at full strength; only the taper shape (the actual "wide,
            # soft transition") is what the blur is for. Same idiom as heal_seam.
            peak = float(alpha.max())
            if 0 < peak < 1.0:
                alpha = np.clip(alpha / peak, 0.0, 1.0)
        alpha = alpha[..., None]

        blended = crop.astype(np.float32) * (1.0 - alpha) + healed_arr * alpha
        result = arr.copy()
        if axis == 0:
            result[lo:hi, :] = blended.clip(0, 255).astype(np.uint8)
        else:
            result[:, lo:hi] = blended.clip(0, 255).astype(np.uint8)
        return result

    def _extract_reference_patches(
        self,
        panorama: Panorama,
        type_map: np.ndarray,
        region_val: int,
        patch_size: int,
        num_patches: int = 1,
        depth: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> list[PIL.Image.Image]:
        """
        Crop up to num_patches squares directly from the most solid, mutually
        distant areas of this region in the panorama's own equirectangular
        pixels. Each crop's bounding box is picked here in panorama pixel
        space; the box itself is then flattened one of two ways:

          - depth given (Panorama.unwarp_box): the box's own pixels are
            unprojected forward into real 3D points, a local plane is fit to
            that small point cluster, and the box is rasterized onto it —
            removes true oblique-view foreshortening (the ground here was
            genuinely photographed at an angle), not just equirect
            distortion. Falls back to perspective_crop for a given patch if
            there aren't enough valid depth points in its box (see
            unwarp_box's min_points) or its points are too close to
            collinear to fit a plane.
          - depth is None: Panorama.perspective_crop (rectilinear
            reprojection) — removes the equirect stretch only, not
            foreshortening from viewing angle.

        Either way, nothing is ever resampled onto a world-space grid larger
        than this one box. These are guidance patches pasted into a larger
        tile canvas (see _build_reference_canvas), not necessarily the full
        tile.

        A patch's bounding box is picked for being deep inside this region,
        but a square box isn't guaranteed to be *entirely* this region right
        up to its edges (e.g. right at a boundary with another material).
        The same region mask used to find the box is reprojected alongside
        the colour (via perspective_crop's own `mask` argument) and returned
        as each patch's alpha channel, so _build_reference_canvas can treat
        only genuine same-region pixels as "real" and let anything else
        within the box's own footprint be regenerated instead of hard-
        preserved.

        An earlier version baked the *entire* panorama top-down (one
        world-space XZ grid spanning the whole terrain, one sample per grid
        cell) before cropping. That reprojection collapses to a radial/
        pinwheel singularity right around the camera — the single largest
        "ground" blob is almost always the area immediately around it — and
        stretches distant ground near the horizon into blocky artefacts,
        because a global grid has to back-sample the panorama from every
        cell, including ones arbitrarily close to that singularity.
        unwarp_box's local flattening doesn't have this problem even though
        it also rasterizes onto a grid: its grid only ever spans one already-
        solid, well-separated box, and it forward-projects that box's own
        pixels rather than back-sampling toward a singularity.

        A version before that scattered many small patches from anywhere in
        the region into a collage. That produced a hard-seamed, visibly
        patchy mosaic whenever the region spanned meaningfully different
        lighting or exposure (sunlit rock vs. shadow vs. a warm sunset glow,
        all the same region label) — each patch only feathered against its
        own mask boundary, never against its neighbours on the canvas, so
        adjacent patches could clash violently. A version after that used a
        single crop from one real, contiguous patch — inherently coherent,
        but with only one real colour/lighting anchor, generated fill can
        drift far from it. This version keeps the "one coherent crop per
        patch" property but samples num_patches of them, each from a
        distinct, well-separated location (via a distance transform over the
        region's panorama-space mask — the most "interior," safest point of
        whatever solid material is left, not just a bounding-box centre),
        so Pass 1 has more than one real anchor to match against wherever
        it's generating. Each area is excluded before picking the next one so
        patches don't overlap or crowd each other.

        Returns fewer than num_patches (down to an empty list) if there isn't
        enough distinct material of this type — the caller falls back to
        plain text-to-image generation if the list ends up empty.

        exclude_mask (bool, True = never pick from here — see _bake_reference):
        typically ObjectClear's own foreground-removal mask. Applied before the
        search starts, not just as a tie-breaker — a fabricated-fill region is
        often the most homogeneous ("solid") blob of its type in the whole
        panorama, so left unfiltered it would consistently win the distance-
        transform peak search over genuine, more textured real photo content.
        """
        H, W = type_map.shape
        region_mask = (type_map == region_val).astype(np.float32)
        remaining = region_mask > 0.5
        if exclude_mask is not None:
            remaining &= ~exclude_mask
        patches: list[PIL.Image.Image] = []

        for _ in range(max(1, num_patches)):
            if not remaining.any():
                break
            dist = distance_transform_edt(remaining)
            peak = float(dist.max())
            if peak < 4.0:   # nothing left big enough to be worth a patch
                break
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            side = min(max(16, int(2 * peak)), H, W)
            half = side // 2
            sr0 = int(np.clip(cy - half, 0, H - side))
            sc0 = int(np.clip(cx - half, 0, W - side))

            box = [sc0, sr0, side, side]
            crop = (
                panorama.unwarp_box(depth, box, mask=region_mask, out_size=patch_size)
                if depth is not None else None
            )
            if crop is None:
                crop = panorama.perspective_crop(box, mask=region_mask).resize(
                    (patch_size, patch_size), PIL.Image.LANCZOS,
                )
            patches.append(crop)

            # Exclude this area (out to its own safe radius) so the next pick
            # lands somewhere distinct rather than immediately adjacent.
            yy, xx = np.ogrid[:H, :W]
            remaining &= (yy - cy) ** 2 + (xx - cx) ** 2 > peak ** 2

        return patches

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

    def _save_debug_step(
        self,
        cfg: "TerrainTextureGenerationConfiguration",
        debug_label: str,
        step_name: str,
        image: PIL.Image.Image,
    ) -> None:
        """
        Save one intermediate image of the generation process, if debugging is
        on. No-op unless both cfg.debug_save_steps and self.temp are set, so
        this is free to call liberally at every step without extra guards at
        each call site.
        """
        if not cfg.debug_save_steps or self.temp is None:
            return
        step_dir = self.temp / "texture_generation"
        step_dir.mkdir(parents=True, exist_ok=True)
        image.convert("RGB" if image.mode != "L" else "L").save(step_dir / f"{debug_label}_{step_name}.png")

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
        names = InPainting.model_names(
            self.config.inpainting_type,
            lora_type=PanoramaLoraType.FLUX_SEAMLESS_TEXTURE if self.config.use_lora else None,
        )
        if self.config.pass1_inpainting_type == "LAMA":
            names = names + InPainting.model_names(InPaintingType.LAMA)
        if self.config.describe_material and self.config.use_photo_reference:
            names = names + [CaptioningModel.FLORENCE2.value]
        if self.config.use_intrinsic_delighting and self.config.use_photo_reference:
            names = names + ImageIntrinsics.model_names()
        if self.config.use_patch_supersampling and self.config.use_photo_reference:
            names = names + ImageSupersampling.model_names()
        return names

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        # Normally already closed right after their one narrow use per region (see the
        # _close_* helpers above) -- these remain as a safety net for early-exit/exception
        # paths that skip past that point.
        self._close_lama_inpainter()
        self._close_intrinsics()
        self._close_supersampler()
        self._close_captioner()
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
