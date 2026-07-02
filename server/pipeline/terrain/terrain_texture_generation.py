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
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.terrain.terrain_generator import TerrainMeshGenerator
from scene.splat_material import SplatLayer, SplatMaterial
from util.image_utils import Image, lab_color_transfer
from util.device_utils import DeviceStrategy, preferred_device


_GROUND_TYPES: frozenset[RegionType] = frozenset({
    RegionType.GROUND,
    RegionType.TERRAIN,
    RegionType.VEGETATION,
    RegionType.WATER,
    RegionType.ROAD,
    RegionType.TRAIL,
    RegionType.BUILT,
})

# Short on purpose: at FLUX Fill's guidance_scale (30, see TerrainTextureGenerationConfiguration)
# the text prompt strongly dominates the img2img seed, so an elaborate, hyper-specific
# description would fight the actual photo/collage reference instead of following it. These
# name the material and its 1-2 defining traits only — the scattered-patch collage seed
# (_generate_scattered_texture_base) and per-region LAB colour transfer carry the rest.
_BASE_PROMPTS: dict[RegionType, str] = {
    RegionType.GROUND: "Ultra-detailed close-up photograph of loose soil, crisp dirt grains, small sharp broken stones and pebbles embedded in ground, high-contrast micro-shadows, hyper-realistic, game engine asset",
    RegionType.TERRAIN: "RAW unedited photograph of fractured grey mountain scree, jagged broken rock with lichen patches",
    RegionType.VEGETATION: "Ultra-detailed macro overhead photograph of dense wild ground foliage, crisp individual blades of grass, sharp moss structures, small clover patches, high-frequency natural details",
    RegionType.WATER: "RAW unedited photograph of still lake water, rippling reflective surface",
    RegionType.ROAD: "RAW unedited photograph of worn dark asphalt pavement, fine cracks and stone aggregate",
    RegionType.TRAIL: "RAW unedited photograph of hard-packed dirt hiking trail, pale compacted earth",
    RegionType.BUILT: "RAW unedited photograph of old granite cobblestone pavement, mortared stone blocks",
}

# PBR smoothness per region: 0 = perfectly rough/matte, 1 = mirror-smooth.
# Water is high (reflective surface); soil/grass/gravel are near-zero (diffuse).
_LAYER_SMOOTHNESS: dict[RegionType, float] = {
    RegionType.GROUND:      0.05,
    RegionType.TERRAIN:     0.08,
    RegionType.VEGETATION:  0.04,
    RegionType.WATER:       0.88,
    RegionType.ROAD:        0.18,
    RegionType.TRAIL:       0.05,
    RegionType.BUILT:       0.28,
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
        seed: int = 0,
        tile_size: int = 1024,
        blend_map_size: int = 1024,
        blend_sigma: float = 0.05,
        min_region_fraction: float = 0.02,
        inpainting_type: str = "FLUX",
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        seam_width_fraction: float = 0.08,
        seam_dilation_px: int = 8,
        use_panorama_layer: bool = True,
        panorama_blend_power: float = 2.0,
        # 200 m terrain / 4 m per tile = 50 repeats → ~4 cm/texel at 1024 px
        synthetic_tile_factor: float = 50.0,
        use_photo_reference: bool = True,
        reference_tex_size: int = 2048,
        # Diffusers `strength` semantics: 1.0 = maximum added noise = ignores `image`
        # entirely; lower = stays closer to the seed. Higher than a plain photo-crop
        # reference would use, because the scattered-patch collage seed
        # (_generate_scattered_texture_base) is already packed with sharp, genuine
        # same-region detail — FLUX mainly needs to unify lighting/seams and follow
        # the text prompt rather than invent texture from scratch.
        collage_strength: float = 0.65,
        # Post-process LAB colour nudge toward the collage seed's stats.
        color_transfer_strength: float = 0.35,
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
        # UV tiling factor for synthetic region tiles (panorama layer always uses 1.0).
        # At 50× over a 200 m grid one tile covers 4 m → ~0.4 cm/texel at 1024 px.
        self.synthetic_tile_factor = synthetic_tile_factor
        self.use_photo_reference = use_photo_reference
        self.reference_tex_size = reference_tex_size
        self.collage_strength = collage_strength
        self.color_transfer_strength = color_transfer_strength


class TerrainTextureGenerationStage(PipelineStage):
    """
    Generates high-quality tileable region textures and packages them as a SplatMaterial.

    For each ground region type present in the REGION_MAP (above min_region_fraction):
      1. If a real photo is available (PANORAMA_TERRAIN + HEIGHT_MAP), bake it
         top-down and build a high-frequency "collage" seed: dozens of small,
         feather-edged crisp patches sampled only from that region's own pixels
         are scattered across a blank canvas (_generate_scattered_texture_base).
         Because every seed pixel is genuine same-material detail, this avoids
         both cross-region bleed and the smeared/stretched look of cropping a
         single (possibly small or irregular) footprint. The collage drives
         generation as a moderate-`strength` img2img seed — the short,
         material-identity-only text prompt (_BASE_PROMPTS) mainly unifies
         lighting and fixes seams rather than reinventing the scene. A region
         with no trustworthy pixels at all falls back to plain text generation.
      2. Generate a photorealistic seamlessly tileable tile with FLUX (two-pass:
         collage-seeded/full-mask generation → circular-shift seam inpainting),
         then sharpen/contrast-boost to recover detail softened by inpainting
         and nudge colour statistics back toward the collage seed. A local
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
      ContextKey.PANORAMA_TERRAIN     — real photo, source for the collage seed (optional)
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
            self._inpainter = InPainting(inpaint_device, inpaint_dtype, cfg.inpainting_type)

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
                    f"{', photo-referenced (collage)' if reference is not None else ''})"
                )
                self.advance_progress(task)
                self.advance_progress(task)
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
        use_panorama_layer is True, the panorama-baked texture is inserted as
        layer 0 with a facing-based visibility blend weight (surface normal
        vs. direction to camera, see _panorama_visibility_weight). The
        synthetic region layer weights are scaled by (1 − panorama_weight) so
        all weights sum to 1 everywhere.
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
                height_map_depth.depth, half, cfg.blend_map_size, cfg.panorama_blend_power
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
        nadir_cutoff_deg: float = -80.0,
        nadir_fade_deg: float = 5.0,
    ) -> np.ndarray:
        """
        Per-texel visibility weight for the panorama layer, based on how
        directly each terrain face is oriented toward the camera (at the
        world origin) rather than on depression angle alone.

        The surface normal is derived from the height-map gradient and
        dotted with the direction back to the camera: a face pointing at the
        camera — e.g. a distant mountain slope viewed head-on, even though
        it sits near the horizon — gets high weight and keeps the real
        panorama photo. A face that's near edge-on to the view ray (steep or
        perpendicular, regardless of distance) fades toward the generated
        layer instead, since an equirectangular sample of it is stretched
        and unreliable. A narrow nadir dead-zone directly under the camera
        is excluded separately — that's an equirectangular pole singularity,
        not a facing problem, so it isn't captured by the normal dot product.

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

        # Surface normal from the height-field gradient, in world-unit spacing.
        texel_size = (2.0 * half) / max(blend_map_size - 1, 1)
        dY_dX = np.gradient(Y, texel_size, axis=1)
        dY_dZ = np.gradient(Y, texel_size, axis=0)
        normal = np.stack([-dY_dX, np.ones_like(Y), -dY_dZ], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)

        # Direction from each terrain point back to the camera at the origin.
        point = np.stack([X, Y, Z], axis=-1).astype(np.float64)
        dist = np.linalg.norm(point, axis=-1).clip(1e-6)
        to_camera = (-point / dist[..., None]).astype(np.float32)

        facing = np.einsum("...c,...c->...", normal, to_camera).clip(0.0, 1.0)

        r_xz = np.sqrt(X.astype(np.float64) ** 2 + Z.astype(np.float64) ** 2).clip(1e-6)
        lat = np.arctan2(Y.astype(np.float64), r_xz)
        min_lat = np.radians(nadir_cutoff_deg)
        nadir_fade = (
            (lat - min_lat) / max(np.radians(nadir_fade_deg), 1e-6)
        ).clip(0.0, 1.0).astype(np.float32)

        weight = facing * nadir_fade
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
        return f"{base}{_TILE_SUFFIX}"

    def _generate_tileable_tile(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
    ) -> PIL.Image.Image:
        """
        Pure text-to-image tile generation for regions with no usable photo
        reference at all (either the whole scene has none, or this specific
        region has no trustworthy pixels in the baked photo).
        """
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        base_image = PIL.Image.new("RGB", (T, T), (128, 128, 128))
        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = self._inpainter.inpaint(
            input_image=base_image,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=1.0,
            seed=seed,
        )

        result_img = self._seam_fix(generated, cfg, prompt, seed)
        return self._pack_local_height_channel(result_img)

    def _generate_tileable_tile_high_fidelity(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
        region_map: np.ndarray,
        baked_color: PIL.Image.Image,
        region_val: int,
    ) -> PIL.Image.Image:
        """
        Tile generation seeded from a scattered same-region texture collage
        (_generate_scattered_texture_base) rather than a single stretched
        reference crop. Because the seed is already packed with sharp,
        genuine material detail everywhere, FLUX mainly needs to unify
        lighting and smooth the patch seams rather than invent texture from
        scratch — hence the higher `collage_strength` compared to a
        low-strength photo-crop reference. Falls back to plain text
        generation if the region has no trustworthy pixels at all.
        """
        T = cfg.tile_size
        seed = cfg.seed + seed_offset
        rng = np.random.default_rng(seed)

        base_image = self._generate_scattered_texture_base(
            baked_color=baked_color,
            region_map=region_map,
            region_val=region_val,
            target_size=T,
            rng=rng,
        )
        if base_image is None:
            return self._generate_tileable_tile(prompt, cfg, seed_offset)

        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = self._inpainter.inpaint(
            input_image=base_image,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.collage_strength,
            seed=seed,
        )

        result_img = self._seam_fix(generated, cfg, prompt, seed)

        # Pass 3: detail recovery — the two inpainting passes can soften the
        # collage's crisp stones/grass structure, so restore local contrast.
        result_img = result_img.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
        result_img = PIL.ImageEnhance.Contrast(result_img).enhance(1.15)
        result_img = PIL.ImageEnhance.Sharpness(result_img).enhance(1.20)

        # Pass 4: nudge final colour statistics back toward the collage seed.
        if cfg.color_transfer_strength > 0.0:
            result_img = lab_color_transfer(
                source=base_image, target=result_img, strength=cfg.color_transfer_strength,
            )

        return self._pack_local_height_channel(result_img)

    def _seam_fix(
        self,
        generated: PIL.Image.Image,
        cfg: TerrainTextureGenerationConfiguration,
        prompt: str,
        seed: int,
    ) -> PIL.Image.Image:
        """Circular-shift the tile by half its size and inpaint over the new seam, then shift back."""
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

        fixed = self._inpainter.inpaint(
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

    def _generate_scattered_texture_base(
        self,
        baked_color: PIL.Image.Image,
        region_map: np.ndarray,
        region_val: int,
        target_size: int,
        rng: np.random.Generator,
        num_patches: int = 24,
        patch_sizes: tuple[int, ...] = (128, 256),
        feather_px: float = 6.0,
    ) -> Optional[PIL.Image.Image]:
        """
        Build a high-frequency texture "collage" seed for a region by
        scattering small crisp sub-patches — sampled only from that region's
        own pixels in the baked top-down photo — across a blank canvas. Each
        patch's opacity is feathered to zero within feather_px of its own
        region-mask boundary (or crop edge, if clipped by the source image
        bounds), so pasted patches blend into the canvas instead of leaving
        hard rectangular seams. This produces a seed that is uniformly
        high-frequency and true to the source material everywhere, unlike a
        single stretched crop of the region's (possibly small or irregular)
        footprint.

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
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return None

        color_arr = np.array(baked_color.convert("RGB"))
        mean_color = color_arr[mask].mean(axis=0).astype(np.uint8)
        canvas = np.tile(mean_color, (target_size, target_size, 1)).astype(np.float32)

        for _ in range(num_patches):
            p_size = int(rng.choice(patch_sizes))
            anchor = int(rng.integers(0, rows.size))
            cr, cc = int(rows[anchor]), int(cols[anchor])

            r0 = max(0, min(cr - p_size // 2, max(0, H - p_size)))
            c0 = max(0, min(cc - p_size // 2, max(0, W - p_size)))
            r1, c1 = min(H, r0 + p_size), min(W, c0 + p_size)

            src_patch = color_arr[r0:r1, c0:c1]
            src_mask = mask[r0:r1, c0:c1]
            ph, pw = src_patch.shape[:2]
            if ph == 0 or pw == 0 or not src_mask.any():
                continue

            max_oy, max_ox = target_size - ph, target_size - pw
            if max_oy < 0 or max_ox < 0:
                continue
            out_y = int(rng.integers(0, max_oy + 1))
            out_x = int(rng.integers(0, max_ox + 1))

            alpha = np.clip(distance_transform_edt(src_mask) / feather_px, 0.0, 1.0)[..., None]
            region = canvas[out_y:out_y + ph, out_x:out_x + pw]
            region[:] = src_patch.astype(np.float32) * alpha + region * (1.0 - alpha)

        return PIL.Image.fromarray(canvas.clip(0, 255).astype(np.uint8))

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
        return InPainting.model_names(self.config.inpainting_type)

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
