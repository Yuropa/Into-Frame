from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import torch
from scipy.ndimage import binary_dilation, gaussian_filter, zoom
from scipy.ndimage import map_coordinates

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from scene.splat_material import SplatLayer, SplatMaterial
from util.image_utils import Image
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

_BASE_PROMPTS: dict[RegionType, str] = {
    RegionType.GROUND: (
        "close-up macro top-down photo of natural bare earth and soil, "
        "loose dirt with fine grit and small pebbles, subtle mineral color variation, "
        "scattered organic fragments and root fibers, weathered natural ground surface"
    ),
    RegionType.TERRAIN: (
        "close-up macro top-down photo of mountain rock and gravel surface, "
        "fractured stone pieces in varied sizes from pebble to fist-sized, "
        "mineral color variation from grey to rust-brown, lichen patches, rough weathered rock"
    ),
    RegionType.VEGETATION: (
        "close-up macro top-down photo of dense mixed grass ground cover, "
        "intertwined grass blades of varying length and shade of green, "
        "tiny wildflowers and clover scattered throughout, rich lush natural vegetation"
    ),
    RegionType.WATER: (
        "close-up macro top-down photo of calm water surface viewed from directly above, "
        "subtle concentric ripple patterns, clear translucent water with pale blue-green tint, "
        "gentle surface undulation, fine caustic light patterns on the bottom"
    ),
    RegionType.ROAD: (
        "close-up macro top-down photo of weathered asphalt road surface, "
        "grey tarmac with exposed stone aggregate and tar binder, "
        "fine hairline cracks, age patina and wear, rough granular road texture"
    ),
    RegionType.TRAIL: (
        "close-up macro top-down photo of compacted dirt hiking trail, "
        "hard-packed earth with fine gravel, small embedded stones and exposed roots, "
        "subtle bootprint impressions and erosion channels, natural trail surface"
    ),
    RegionType.BUILT: (
        "close-up macro top-down photo of weathered stone cobblestone paving, "
        "rectangular stones with mortar joints, varied grey and brown stone colors, "
        "worn rounded edges and minor surface chips, aged urban ground surface"
    ),
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


class TerrainTextureGenerationStage(PipelineStage):
    """
    Generates high-quality tileable region textures and packages them as a SplatMaterial.

    For each ground region type present in the REGION_MAP (above min_region_fraction):
      1. Generate a photorealistic seamlessly tileable tile with FLUX (two-pass:
         full-mask generation → circular-shift seam inpainting).

    SplatMaterial.from_region_map handles weight computation, normalisation, and
    RGBA blend map packing.  The first layer tile is also written to TERRAIN_TEXTURE
    so the terrain mesh can embed a preview in the GLB.

    Reads:
      ContextKey.REGION_MAP     — top-down region type grid (optional)
      ContextKey.INPUT_CAPTION  — scene caption for prompt context (optional)

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

        inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
        self._inpainter = InPainting(inpaint_device, inpaint_dtype, cfg.inpainting_type)

        tiles: dict[str, PIL.Image.Image] = {}
        for idx, rt in enumerate(present_types):
            prompt = self._build_prompt(rt, caption)
            tiles[rt.label] = self._generate_tileable_tile(prompt, cfg, seed_offset=idx)
            self.log_info(f"Generated {rt.label} tile ({cfg.tile_size}px, seamless)")
            self.advance_progress(task)
            self.advance_progress(task)

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
        layer 0 with a view-angle blend weight (sin(depression)^power). The
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
            pano_weight = self._view_angle_weight(
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

            layers = [SplatLayer(name="panorama", tile=pano_tile, tile_factor=1.0, equirect=True)]
            layers += [
                SplatLayer(name=rt.label, tile=tiles[rt.label], tile_factor=cfg.synthetic_tile_factor)
                for rt in present_types if rt.label in tiles
            ]
        else:
            weight_maps = synth_weight_maps
            layers = [
                SplatLayer(name=rt.label, tile=tiles[rt.label], tile_factor=cfg.synthetic_tile_factor)
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
    def _view_angle_weight(
        height_map: np.ndarray,
        half: float,
        blend_map_size: int,
        blend_power: float,
        nadir_cutoff_deg: float = -35.0,
        nadir_fade_deg: float = 10.0,
        horizon_fade_deg: float = 5.0,
    ) -> np.ndarray:
        """
        Per-texel view-angle quality weight for the panorama layer.

        Computed purely from height-map geometry — no panorama sampling needed.
        Returns a float32 array of shape (blend_map_size, blend_map_size) in [0, 1].

        Weight is high when the camera looks steeply down at the terrain point
        (reliable, low-distortion equirectangular sample) and fades to 0 at:
          • grazing / near-horizon angles (equirect pixels are stretched thin)
          • the nadir dead-zone directly below the camera (high distortion)
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

        r_xz = np.sqrt(X.astype(np.float64) ** 2 + Z.astype(np.float64) ** 2).clip(1e-6).astype(np.float32)
        lat = np.arctan2(Y.astype(np.float64), r_xz.astype(np.float64)).astype(np.float32)

        min_lat = np.radians(nadir_cutoff_deg)
        fade_in  = ((lat - min_lat) / max(np.radians(nadir_fade_deg), 1e-6)).clip(0.0, 1.0)
        fade_out = ((-lat)          / max(np.radians(horizon_fade_deg), 1e-6)).clip(0.0, 1.0)
        valid    = (lat < 0.0) & (lat >= min_lat)
        weight   = np.where(valid, np.minimum(fade_in, fade_out), 0.0).astype(np.float32)
        return weight ** blend_power

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
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        # Pass 1: pure generation via 100%-masked neutral image
        gray = PIL.Image.new("RGB", (T, T), (128, 128, 128))
        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = self._inpainter.inpaint(
            input_image=gray,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=seed,
        )

        # Pass 2: circular-shift seam fix
        arr = np.array(generated.convert("RGB"), dtype=np.uint8)
        half = T // 2
        shifted = np.roll(np.roll(arr, half, axis=0), half, axis=1)

        seam_mask = np.zeros((T, T), dtype=np.uint8)
        sw = max(2, int(T * cfg.seam_width_fraction / 2))
        cx = half
        seam_mask[max(0, cx - sw): min(T, cx + sw), :] = 255
        seam_mask[:, max(0, cx - sw): min(T, cx + sw)] = 255
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
