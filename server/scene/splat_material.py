import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, zoom


MAX_LAYERS = 8
MAX_BLEND_MAPS = 2
CHANNELS_PER_BLEND_MAP = 4  # RGBA


@dataclass
class SplatLayer:
    """One named region: a tileable texture tile."""
    name: str
    tile: PILImage.Image
    # How many times this layer's tile repeats across the full terrain UV space.
    # 1.0 = the tile covers the terrain exactly once (used for the panorama layer).
    # Higher values (e.g. 50.0) tile the texture for synthetic region layers.
    tile_factor: float = 1.0
    # When True the shader derives UVs from the vertex world position using
    # standard equirectangular projection rather than from TEXCOORD_0 × tile_factor.
    # Used for the panorama layer: U = atan2(X,Z)/(2π) + 0.5
    #                              V = 0.5 − φ/π   where φ = atan2(Y, √(X²+Z²))
    # The tile is the full equirectangular image resized to square.
    # V = 0.5 at the horizon; V < 0.5 for above-horizon (mountains); V = 1.0 at nadir.
    equirect: bool = False
    # PBR smoothness for this layer's surface in [0, 1].
    # 0 = perfectly rough/matte (soil, grass); 1 = perfectly mirror-smooth.
    # Water sits at ~0.88; concrete ~0.25; rock/trail/vegetation < 0.1.
    smoothness: float = 0.1


class SplatMaterial:
    """
    N tileable region textures paired with RGBA blend maps.

    Channel assignment is positional:
      blend_maps[0]  R → layers[0]   G → layers[1]   B → layers[2]   A → layers[3]
      blend_maps[1]  R → layers[4]   G → layers[5]   B → layers[6]   A → layers[7]

    Supports up to 8 layers across 2 blend maps.

    Construction
    ------------
    from_region_map   — full pipeline: raw integer region map + tiles dict → weights + blend maps
    from_weight_maps  — lower level: pre-computed float weight maps + layers → blend maps
    from_single_layer — trivial case: one tile, no blending needed

    Wire format (encode / decode)
    -----------------------------
    {
      "tile_size": int,
      "layers": [{"name": str, "tile": "<base64 PNG>"}, ...],
      "blend_maps": ["<base64 PNG>", ...]
    }

    File layout (save / load)
    -------------------------
    splat_material.json   — manifest with tile_size and ordered layer names
    blend_map_0.png       — RGBA
    blend_map_1.png       — RGBA (only present when more than 4 layers)
    tile_<name>.png       — one file per layer
    """

    def __init__(
        self,
        layers: list[SplatLayer],
        blend_maps: list[PILImage.Image],
        tile_size: int,
    ) -> None:
        if len(layers) > MAX_LAYERS:
            raise ValueError(f"SplatMaterial supports at most {MAX_LAYERS} layers, got {len(layers)}")
        if len(blend_maps) > MAX_BLEND_MAPS:
            raise ValueError(f"SplatMaterial supports at most {MAX_BLEND_MAPS} blend maps, got {len(blend_maps)}")
        self.layers = layers
        self.blend_maps = blend_maps
        self.tile_size = tile_size

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_region_map(
        cls,
        region_map: np.ndarray,
        index_to_label: dict[int, str],
        tiles: dict[str, PILImage.Image],
        blend_map_size: int = 1024,
        blend_sigma: float = 0.05,
        min_fraction: float = 0.02,
    ) -> "SplatMaterial":
        """
        Build a SplatMaterial from a raw integer region map.

        region_map      — H×W array of integer type indices
        index_to_label  — maps each integer value to a layer name
        tiles           — label → tileable tile image (labels not in this dict are skipped)
        blend_map_size  — output resolution of the blend maps
        blend_sigma     — gaussian smoothing width as a fraction of blend_map_size
        min_fraction    — regions covering less than this fraction of cells are skipped
        """
        total = region_map.size

        # Collect eligible labels ordered by coverage (largest first)
        candidates: list[tuple[int, str, int]] = []
        for idx, label in index_to_label.items():
            if label not in tiles:
                continue
            count = int((region_map == idx).sum())
            if count / total >= min_fraction:
                candidates.append((idx, label, count))

        candidates.sort(key=lambda t: -t[2])
        candidates = candidates[:MAX_LAYERS]

        if not candidates:
            # Fallback: use the first available tile as a uniform cover
            label, tile = next(iter(tiles.items()))
            return cls.from_single_layer(label, tile, blend_map_size)

        sigma_px = max(4.0, blend_sigma * blend_map_size)
        weight_maps: dict[str, np.ndarray] = {}
        for idx, label, _ in candidates:
            mask = zoom(
                (region_map == idx).astype(np.float32),
                (blend_map_size / region_map.shape[0], blend_map_size / region_map.shape[1]),
                order=1,
            )
            weight_maps[label] = gaussian_filter(mask, sigma=sigma_px)

        # Normalise so weights sum to 1 everywhere
        total_w = sum(weight_maps.values()) + 1e-6
        for label in weight_maps:
            weight_maps[label] /= total_w

        ordered_labels = [label for _, label, _ in candidates]
        layers = [SplatLayer(name=label, tile=tiles[label]) for label in ordered_labels]
        blend_maps = cls._pack_blend_maps(ordered_labels, weight_maps)
        tile_size = tiles[ordered_labels[0]].width
        return cls(layers=layers, blend_maps=blend_maps, tile_size=tile_size)

    @classmethod
    def from_weight_maps(
        cls,
        layers: list[SplatLayer],
        weight_maps: dict[str, np.ndarray],
    ) -> "SplatMaterial":
        """
        Build a SplatMaterial from pre-computed float weight maps.

        layers      — ordered layer list (defines channel assignment)
        weight_maps — label → float32 array in [0, 1], all the same size
        """
        ordered_labels = [layer.name for layer in layers]
        blend_maps = cls._pack_blend_maps(ordered_labels, weight_maps)
        tile_size = layers[0].tile.width if layers else 0
        return cls(layers=layers, blend_maps=blend_maps, tile_size=tile_size)

    @classmethod
    def from_single_layer(
        cls,
        name: str,
        tile: PILImage.Image,
        blend_map_size: int = 1024,
    ) -> "SplatMaterial":
        """Build a SplatMaterial with one layer covering the entire surface."""
        bm = np.zeros((blend_map_size, blend_map_size, 4), dtype=np.uint8)
        bm[:, :, 0] = 255
        return cls(
            layers=[SplatLayer(name=name, tile=tile)],
            blend_maps=[PILImage.fromarray(bm, "RGBA")],
            tile_size=tile.width,
        )

    # ── Blend map packing (internal) ──────────────────────────────────────────

    @staticmethod
    def _pack_blend_maps(
        ordered_labels: list[str],
        weight_maps: dict[str, np.ndarray],
    ) -> list[PILImage.Image]:
        """Pack float weight maps into RGBA blend map images."""
        h, w = next(iter(weight_maps.values())).shape
        blend_maps = []
        for map_idx in range(MAX_BLEND_MAPS):
            start = map_idx * CHANNELS_PER_BLEND_MAP
            slot = ordered_labels[start: start + CHANNELS_PER_BLEND_MAP]
            if not slot:
                break
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            for ch, label in enumerate(slot):
                rgba[:, :, ch] = (weight_maps[label] * 255).clip(0, 255).astype(np.uint8)
            blend_maps.append(PILImage.fromarray(rgba, "RGBA"))
        return blend_maps

    # ── Encode / decode ───────────────────────────────────────────────────────

    def encode(self) -> dict:
        def _b64(img: PILImage.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "tile_size": self.tile_size,
            "layers": [
                {
                    "name": layer.name,
                    "tile_factor": layer.tile_factor,
                    "equirect": layer.equirect,
                    "smoothness": layer.smoothness,
                    "tile": _b64(layer.tile),
                }
                for layer in self.layers
            ],
            "blend_maps": [_b64(bm) for bm in self.blend_maps],
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        def _from_b64(b64: str) -> PILImage.Image:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).copy()

        layers = [
            SplatLayer(
                name=entry["name"],
                tile=_from_b64(entry["tile"]),
                tile_factor=entry.get("tile_factor", 1.0),
                equirect=entry.get("equirect", False),
                smoothness=entry.get("smoothness", 0.1),
            )
            for entry in data["layers"]
        ]
        blend_maps = [_from_b64(bm) for bm in data["blend_maps"]]
        return cls(layers=layers, blend_maps=blend_maps, tile_size=data["tile_size"])

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "tile_size": self.tile_size,
            "layers": [
                {"name": layer.name, "tile_factor": layer.tile_factor, "equirect": layer.equirect, "smoothness": layer.smoothness}
                for layer in self.layers
            ],
        }
        (path / "splat_material.json").write_text(json.dumps(manifest, indent=2))
        for i, bm in enumerate(self.blend_maps):
            bm.save(path / f"blend_map_{i}.png")
        for layer in self.layers:
            layer.tile.save(path / f"tile_{layer.name}.png")

    @classmethod
    def load(cls, path: Path) -> Self:
        manifest = json.loads((path / "splat_material.json").read_text())
        tile_size = manifest["tile_size"]
        layers = [
            SplatLayer(
                name=entry["name"],
                tile=PILImage.open(path / f"tile_{entry['name']}.png").copy(),
                tile_factor=entry.get("tile_factor", 1.0),
                equirect=entry.get("equirect", False),
                smoothness=entry.get("smoothness", 0.1),
            )
            for entry in manifest["layers"]
        ]
        blend_maps = []
        for i in range(MAX_BLEND_MAPS):
            p = path / f"blend_map_{i}.png"
            if p.exists():
                blend_maps.append(PILImage.open(p).copy())
        return cls(layers=layers, blend_maps=blend_maps, tile_size=tile_size)
