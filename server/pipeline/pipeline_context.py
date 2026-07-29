from pipeline.context_value import ContextValue, ValueKeys
from pathlib import Path
from typing import Literal, TypeAlias, Optional, Any
import logging
import numpy as np
from util.json_utils import write_json, parse_json

_STAGE_ORDER_FILE = "_stage_order.json"

_log = logging.getLogger("pipeline")
from scene.mesh import Mesh
from util.depth_utils import Depth
from util.intrinsic_utils import IntrinsicImages
from util.image_utils import Image
from util.cubemap_utils import CubeMap
from util.panorama_utils import Panorama
from util.video_utils import Video
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics, CameraExtrinsics
from scene.lighting import SceneLighting

class ContextKey:
    """Well-known string keys for values stored in PipelineContext."""
    INPUT = "input"
    DEPTH = "depth"
    INTRINSIC_IMAGES = "intrinsic_images"
    SCENE = "scene"
    INTRINSICS = "intrinsics"
    EXTRINSICS = "extrinsics"
    PANORAMA = "panorama"
    INPUT_CAPTION = "input_caption"
    PANAORAMA_CUBENAME = "panorama_cubemap"
    OBJECT_COUNT = "count"
    FOREGROUND_MASKED_IMAGE = "foreground_masked_image"
    PANORAMA_DEPTH = "panorama_depth"
    PANORAMA_OBJECT_DEPTH = "panorama_object_depth"
    PANORAMA_TERRAIN = "panorama_terrain"
    PANORAMA_FOREGROUND_REMOVED = "panorama_foreground_removed"
    PANORAMA_FOREGROUND_MASK = "panorama_foreground_mask"
    # The occluder-only half of PANORAMA_FOREGROUND_MASK: just the pixels covered
    # by a genuine occluder crop (tree/building/...), without the near-field ground
    # the foreground_distance_m depth test also sweeps in. Written by
    # PanoramaForegroundInpaintingStage, read by TerrainTextureGenerationStage to
    # recover real ground colour without recovering the occluders with it.
    PANORAMA_FOREGROUND_OCCLUDER_MASK = "panorama_foreground_occluder_mask"
    PANORAMA_SKY_MASK = "panorama_sky_mask"
    PANORAMA_SKY = "panorama_sky"
    HEIGHT_MAP = "height_map"
    HEIGHT_MAP_PARAMS = "height_map_params"
    TERRAIN_MESH = "terrain_mesh"
    TERRAIN_PHYSICS_MESH = "terrain_physics_mesh"
    WATER_MESH = "water_mesh"
    TERRAIN_FORMATIONS = "terrain_formations"
    LIGHTING = "lighting"
    RECOGNIZE_TAGS = "recognize_tags"
    OBJECT_CORRELATION = "object_correlation"
    OBJECT_DISTRIBUTION = "object_distribution"
    LINEAR_GRAPH = "linear_graph"
    PANORAMA_REGIONS = "panorama_regions"
    PANORAMA_REGION_TYPE_MAP = "panorama_region_type_map"
    PANORAMA_REGION_TYPE_MAP_RAW = "panorama_region_type_map_raw"
    PANORAMA_REGION_CONFIDENCE_MAP = "panorama_region_confidence_map"
    PANORAMA_REGION_RUNNERUP_TYPE_MAP = "panorama_region_runnerup_type_map"
    PANORAMA_REGION_AMBIGUOUS_MASK = "panorama_region_ambiguous_mask"
    # Terrain-scoped mirror of the five keys above: same PanoramaRegionStage,
    # run a second time against ContextKey.PANORAMA_TERRAIN (object-removed +
    # LoRA-corrected) instead of the original ContextKey.PANORAMA, exactly like
    # PANORAMA_DEPTH already mirrors PANORAMA_OBJECT_DEPTH. Anything that shapes
    # or textures the terrain itself (RegionMapStage, HeightMapStage,
    # TerrainTextureGenerationStage) should read these, not the originals --
    # object clutter that was removed from the photo shouldn't still be
    # classified as ridge/vegetation/etc. in the map that shapes the mesh.
    PANORAMA_REGIONS_TERRAIN = "panorama_regions_terrain"
    PANORAMA_REGION_TYPE_MAP_TERRAIN = "panorama_region_type_map_terrain"
    PANORAMA_REGION_TYPE_MAP_RAW_TERRAIN = "panorama_region_type_map_raw_terrain"
    PANORAMA_REGION_CONFIDENCE_MAP_TERRAIN = "panorama_region_confidence_map_terrain"
    PANORAMA_REGION_RUNNERUP_TYPE_MAP_TERRAIN = "panorama_region_runnerup_type_map_terrain"
    PANORAMA_REGION_AMBIGUOUS_MASK_TERRAIN = "panorama_region_ambiguous_mask_terrain"
    REGION_MAP = "region_map"
    MOUNTAIN_SILHOUETTE = "mountain_silhouette"
    MOUNTAIN_RIDGE_CHAINS = "mountain_ridge_chains"
    PANORAMA_HORIZON = "panorama_horizon"
    WATER_CHAINS = "water_chains"
    INTERIOR_PEAKS = "interior_peaks"
    WATER_SKELETON = "water_skeleton"
    ROAD_SKELETON = "road_skeleton"
    TRAIL_SKELETON = "trail_skeleton"
    HEIGHT_MAP_CERTAINTY = "height_map_certainty"
    HEIGHT_MAP_CELL_RELIEF = "height_map_cell_relief"
    HEIGHT_MAP_CELL_SLOPE = "height_map_cell_slope"
    HEIGHT_MAP_OBSERVED_MASK = "height_map_observed_mask"
    # Narrower than HEIGHT_MAP_OBSERVED_MASK: excludes cells the mountain-ridge
    # envelope overrode in TerrainReconstructionStage. Written by that stage,
    # consumed only by TerrainMeshStage/TerrainMeshGenerator.generate() to
    # decide whether a vertex's cached pre-reconstruction panorama UV is still
    # trustworthy -- see the write site in terrain_reconstruction.py for why
    # this must stay separate from HEIGHT_MAP_OBSERVED_MASK rather than
    # replacing it.
    HEIGHT_MAP_PANO_UV_TRUST_MASK = "height_map_pano_uv_trust_mask"
    # Broader than HEIGHT_MAP_OBSERVED_MASK: also True for the nadir disc's ramp
    # band, which HEIGHT_MAP_OBSERVED_MASK excludes for elevation hard-pinning
    # purposes even though those cells came from a genuine depth sample. Written by
    # HeightMapGenerator, consumed by TerrainReconstructionStage when it builds
    # HEIGHT_MAP_PANO_UV_TRUST_MASK -- a ramp-band cell's cached panorama UV is
    # still trustworthy (it came from a real sample), so it shouldn't be starved of
    # that cached UV just because its height isn't trusted as ground truth.
    HEIGHT_MAP_REAL_SAMPLE_MASK = "height_map_real_sample_mask"
    HEIGHT_MAP_COMPONENT_ID = "height_map_component_id"
    HEIGHT_MAP_PANO_U = "height_map_pano_u"
    HEIGHT_MAP_PANO_V = "height_map_pano_v"
    REGION_MAP_CERTAINTY = "region_map_certainty"
    TERRAIN_TEXTURE = "terrain_texture"
    TERRAIN_TEXTURE_CERTAINTY = "terrain_texture_certainty"
    TERRAIN_TEXTURE_TILES = "terrain_texture_tiles"
    TERRAIN_TEXTURE_TILE_FACTOR = "terrain_texture_tile_factor"
    TERRAIN_MATERIAL = "terrain_material"
    CLIFF_MASK = "cliff_mask"
    GENERATED_VIDEO = "generated_video"
    OBJECT_VIDEO_COUNT = "object_video_count"
    OBJECT_MOTION_COUNT = "object_motion_count"
    CATEGORY_MESH_RIGGING_COUNT = "category_mesh_rigging_count"
    SCENE_ANIMATION_COUNT = "scene_animation_count"
    Type = Literal[
        "input",
        "depth",
        "intrinsic_images",
        "scene",
        "intrinsics",
        "panorama",
        "panorama_terrain",
        "panorama_sky",
        "input_caption",
        "panorama_cubemap",
        "count",
        "foreground_masked_image",
        "panorama_depth",
        "panorama_object_depth",
        "panorama_foreground_mask",
        "panorama_sky_mask",
        "height_map",
        "height_map_params",
        "terrain_mesh",
        "water_mesh",
        "terrain_formations",
        "lighting",
        "recognize_tags",
        "object_correlation",
        "object_distribution",
        "linear_graph",
        "panorama_regions",
        "panorama_region_type_map",
        "panorama_region_type_map_raw",
        "panorama_region_confidence_map",
        "panorama_region_runnerup_type_map",
        "panorama_region_ambiguous_mask",
        "region_map",
        "mountain_silhouette",
        "mountain_ridge_chains",
        "panorama_horizon",
        "water_chains",
        "interior_peaks",
        "water_skeleton",
        "road_skeleton",
        "trail_skeleton",
        "height_map_certainty",
        "height_map_cell_relief",
        "height_map_observed_mask",
        "height_map_pano_uv_trust_mask",
        "height_map_component_id",
        "height_map_pano_u",
        "height_map_pano_v",
        "region_map_certainty",
        "terrain_texture",
        "terrain_texture_certainty",
        "terrain_texture_tiles",
        "terrain_texture_tile_factor",
        "terrain_material",
        "cliff_mask",
        "generated_video",
        "object_video_count",
        "object_motion_count",
        "category_mesh_rigging_count",
        "scene_animation_count",
    ]

ContextKeyName: TypeAlias = ContextKey.Type | str

class PipelineContext():
    """
    Shared key-value store that flows through every pipeline stage.

    Values are namespaced per stage: each stage writes into its own slot, and the
    context walks stages in reverse order when looking up a key, so later stages
    naturally shadow earlier ones. Use input_*() accessors to read values written
    by a prior stage; use the plain accessors (image(), depth(), …) to read values
    that may have been written by the current stage (e.g. for cache checks).
    """

    def __init__(self) -> None:
        self._stage_state = {}
        self._state = {}
        self._current_stage = ""
        self._previous_stage = ""
        self._stage_order = []
        self._dirty_state: set[str] = set()
        self._dirty_stage_state: dict[str, set[str]] = {}
        self._report_sections: list = []

    def push_stage(self, name: str):
        self._current_stage = name
        if name not in self._stage_order:
            self._stage_order.append(name)

    def pop_stage(self):
        self._previous_stage = self._current_stage
        self._current_stage = ""

    def _value(self, name: ContextKeyName, search_stage: Optional[str] = None) -> ContextValue:
        if search_stage is None:
            search_stage = self._current_stage

        # Build the search list: all stages up to and including search_stage, in reverse
        if search_stage and search_stage in self._stage_order:
            idx = self._stage_order.index(search_stage)
            stages_to_search = list(reversed(self._stage_order[:idx + 1]))
        else:
            stages_to_search = list(reversed(self._stage_order))

        # Walk stages in reverse order looking for the value
        for stage in stages_to_search:
            if stage in self._stage_state and name in self._stage_state[stage]:
                return self._stage_state[stage][name]

        # Fall back to global state
        if name in self._state:
            return self._state[name]

        return ContextValue("")

    def _set_value(self, name: ContextKeyName, value: ContextValue):
        if len(self._current_stage) == 0:
            self._state[name] = value
            self._dirty_state.add(name)
            return
    
        if self._current_stage not in self._stage_state:
            self._stage_state[self._current_stage] = {}
        if self._current_stage not in self._dirty_stage_state:
            self._dirty_stage_state[self._current_stage] = set()

        self._stage_state[self._current_stage][name] = value
        self._dirty_stage_state[self._current_stage].add(name)

    # Type
    def type_for(self, name: ContextKeyName) -> Optional[ValueKeys]:
        return self._value(name).type

    # Image
    def add_image(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_image(input)
        self._set_value(name, value)

    def image(self, name: ContextKeyName) -> Optional[Image]:
        return self._value(name).image()
    
    def input_image(self, name: ContextKeyName) -> Optional[Image]:
        return self._value(name, self._previous_stage).image()
    
    # Object
    def add_object(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_object(input)
        self._set_value(name, value)

    def object(self, name: ContextKeyName) -> Optional[Any]:
        return self._value(name).object()
    
    def input_object(self, name: ContextKeyName) -> Optional[Any]:
        return self._value(name, self._previous_stage).object()

    def input_sky_mask(self, name: ContextKeyName = ContextKey.PANORAMA_SKY_MASK) -> Optional[np.ndarray]:
        """
        (H, W) bool sky mask (True = sky) at whatever native resolution it was
        written at, preferring a dedicated segmentation stored as an Image
        (see SkyboxInpaintingStage) over a coarser byproduct some models also
        store as a plain object under the same key (see PanoramaDepthStage's
        DAP sky_mask, a rough side-output of the *depth* model, not a real
        segmentation). Both are legitimate values for this name written by
        different stages -- _value() only ever returns whichever stage wrote
        most recently, with no notion of which is more trustworthy, so a
        caller that only ever calls input_object() silently gets the worse
        one whenever Panorama Depth happens to have run after Skybox
        Inpainting (which it always does in the default pipeline order).
        Returns None if neither is present.
        """
        image = self.input_image(name)
        if image is not None:
            return np.array(image.image.convert("L")) > 127
        obj = self.input_object(name)
        if obj is None:
            return None
        return np.asarray(obj, dtype=bool)

    # Mesh 
    def add_mesh(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_mesh(input)
        self._set_value(name, value)

    def mesh(self, name: ContextKeyName) -> Optional[Mesh]:
        return self._value(name).mesh()
    
    def input_mesh(self, name: ContextKeyName) -> Optional[Mesh]:
        return self._value(name, self._previous_stage).mesh()

    # Depth
    def add_depth(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_depth(input)
        self._set_value(name, value)

    def depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name).depth()
    
    def input_depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name, self._previous_stage).depth()

    # IntrinsicImages (albedo + normal)
    def add_intrinsic_images(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_intrinsic_images(input)
        self._set_value(name, value)

    def intrinsic_images(self, name: ContextKeyName) -> Optional[IntrinsicImages]:
        return self._value(name).intrinsic_images()

    def input_intrinsic_images(self, name: ContextKeyName) -> Optional[IntrinsicImages]:
        return self._value(name, self._previous_stage).intrinsic_images()

    # Object3D
    def add_object3d(self, name: ContextKeyName, input: Object3D):
        value = ContextValue(name=name)
        value.set_object3d(input)
        self._set_value(name, value)

    def object3d(self, name: ContextKeyName) -> Optional[Object3D]:
        return self._value(name).object3d()
    
    def input_object3d(self, name: ContextKeyName) -> Optional[Object3D]:
        return self._value(name, self._previous_stage).object3d()

    # Scene
    def add_scene(self, name: ContextKeyName, input: Scene):
        value = ContextValue(name=name)
        value.set_scene(input)
        self._set_value(name, value)

    def scene(self, name: ContextKeyName) -> Optional[Scene]:
        return self._value(name).scene()
    
    def input_scene(self, name: ContextKeyName) -> Optional[Scene]:
        return self._value(name, self._previous_stage).scene()
    
    # Intrinsics
    def add_intrinsics(self, name: ContextKeyName, input: CameraIntrinsics):
        value = ContextValue(name=name)
        value.set_intrinsics(input)
        self._set_value(name, value)

    def intrinsics(self, name: ContextKeyName) -> Optional[CameraIntrinsics]:
        return self._value(name).intrinsics()
    
    def input_intrinsics(self, name: ContextKeyName) -> Optional[CameraIntrinsics]:
        return self._value(name, self._previous_stage).intrinsics()
    
    # Extrinsics
    def add_extrinsics(self, name: ContextKeyName, input: CameraExtrinsics):
        value = ContextValue(name=name)
        value.set_extrinsics(input)
        self._set_value(name, value)

    def extrinsics(self, name: ContextKeyName) -> Optional[CameraExtrinsics]:
        return self._value(name).extrinsics()
    
    def input_extrinsics(self, name: ContextKeyName) -> Optional[CameraExtrinsics]:
        return self._value(name, self._previous_stage).extrinsics()

    # CubeMap
    def add_cubemap(self, name: ContextKeyName, input: CubeMap):
        value = ContextValue(name=name)
        value.set_cubemap(input)
        self._set_value(name, value)

    def cubemap(self, name: ContextKeyName) -> Optional[CubeMap]:
        return self._value(name).cubemap()
    
    def input_cubemap(self, name: ContextKeyName) -> Optional[CubeMap]:
        return self._value(name, self._previous_stage).cubemap()

    # Panorama
    def add_panorama(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_panorama(input)
        self._set_value(name, value)

    def panorama(self, name: ContextKeyName) -> Optional[Panorama]:
        return self._value(name).panorama()

    def input_panorama(self, name: ContextKeyName) -> Optional[Panorama]:
        return self._value(name, self._previous_stage).panorama()

    # Lighting
    def add_lighting(self, name: ContextKeyName, input: SceneLighting):
        value = ContextValue(name=name)
        value.set_lighting(input)
        self._set_value(name, value)

    def lighting(self, name: ContextKeyName) -> Optional[SceneLighting]:
        return self._value(name).lighting()

    def input_lighting(self, name: ContextKeyName) -> Optional[SceneLighting]:
        return self._value(name, self._previous_stage).lighting()

    # ObjectCorrelationResult
    def add_object_correlation(self, name: ContextKeyName, input: "ObjectCorrelationResult"):
        from pipeline.object_correlation.object_correlation_result import ObjectCorrelationResult
        value = ContextValue(name=name)
        value.set_object_correlation(input)
        self._set_value(name, value)

    def object_correlation(self, name: ContextKeyName) -> Optional["ObjectCorrelationResult"]:
        return self._value(name).object_correlation()

    def input_object_correlation(self, name: ContextKeyName) -> Optional["ObjectCorrelationResult"]:
        return self._value(name, self._previous_stage).object_correlation()

    # ObjectDistributionResult
    def add_object_distribution(self, name: ContextKeyName, input: "ObjectDistributionResult"):
        from pipeline.object_distribution.object_distribution_result import ObjectDistributionResult
        value = ContextValue(name=name)
        value.set_object_distribution(input)
        self._set_value(name, value)

    def object_distribution(self, name: ContextKeyName) -> Optional["ObjectDistributionResult"]:
        return self._value(name).object_distribution()

    def input_object_distribution(self, name: ContextKeyName) -> Optional["ObjectDistributionResult"]:
        return self._value(name, self._previous_stage).object_distribution()

    # PanoramaRegionResult
    def add_panorama_regions(self, name: ContextKeyName, input: "PanoramaRegionResult"):
        from pipeline.panorama_segmentation.panorama_region_result import PanoramaRegionResult
        value = ContextValue(name=name)
        value.set_panorama_regions(input)
        self._set_value(name, value)

    def panorama_regions(self, name: ContextKeyName) -> Optional["PanoramaRegionResult"]:
        return self._value(name).panorama_regions()

    def input_panorama_regions(self, name: ContextKeyName) -> Optional["PanoramaRegionResult"]:
        return self._value(name, self._previous_stage).panorama_regions()

    def add_splat_material(self, name: ContextKeyName, input: "SplatMaterial"):
        from scene.splat_material import SplatMaterial
        value = ContextValue(name=name)
        value.set_splat_material(input)
        self._set_value(name, value)

    def splat_material(self, name: ContextKeyName) -> Optional["SplatMaterial"]:
        return self._value(name).splat_material()

    def input_splat_material(self, name: ContextKeyName) -> Optional["SplatMaterial"]:
        return self._value(name, self._previous_stage).splat_material()

    # Video
    def add_video(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_video(input)
        self._set_value(name, value)

    def video(self, name: ContextKeyName) -> Optional[Video]:
        return self._value(name).video()

    def input_video(self, name: ContextKeyName) -> Optional[Video]:
        return self._value(name, self._previous_stage).video()

    # Report sections — in-memory only, not persisted
    def add_report_section(self, section) -> None:
        self._report_sections.append(section)

    def report_sections(self) -> list:
        return list(self._report_sections)

    def has_stage_output(self, name: ContextKeyName) -> bool:
        """True only if the current stage has already written this key (cache hit for this stage)."""
        stage = self._current_stage
        return stage in self._stage_state and name in self._stage_state[stage]

    # Persistence
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for name in self._dirty_state:
            self._state[name].write(path)
        self._dirty_state.clear()

        for stage_name, dirty_keys in self._dirty_stage_state.items():
            stage_path = path / stage_name
            stage_path.mkdir(parents=True, exist_ok=True)
            for name in dirty_keys:
                self._stage_state[stage_name][name].write(stage_path)
        self._dirty_stage_state.clear()

        # Persist the true, as-run stage order alongside the cached data itself.
        # load() needs this to correctly place a stage that's since been disabled
        # (or dropped from config.yaml) but still has stale cached output on disk.
        # Without it, load() has no record of where that stage actually ran and
        # falls back to appending its directory wherever filesystem iteration
        # happens to encounter it (alphabetically) -- which can shuffle an early
        # stage's data to the very end of the search order. Every downstream
        # input_*() lookup for that stage's keys then silently comes back empty,
        # which reads as "nothing cached" and forces every stage after it to
        # rerun, even though nothing relevant to them actually changed.
        with (path / _STAGE_ORDER_FILE).open("w") as f:
            write_json(self._stage_order, f)

    def save_object(self, name: ContextKeyName, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return self._value(name).write(path=path)

    def _insert_in_declared_order(self, name: str, full_stage_order: Optional[list[str]]):
        """
        Add `name` to self._stage_order, placed at its correct relative position
        rather than always at the end.

        Used for a stage whose data is found on disk but that isn't already in
        self._stage_order (e.g. a stage disabled/removed since the cache was
        written, on a cache old enough to have no persisted _stage_order.json at
        all -- see load() below). Blindly appending in that case would put an
        early stage's data after every other stage's, making every later stage's
        input_*() lookups for its keys silently come back empty -- read as
        "nothing cached", forcing an unwanted full rerun of everything after it.
        full_stage_order (config.yaml's complete, declared stage list, including
        disabled entries) gives us a true reference order to place it by; without
        one, there's no way to do better than appending at the end.
        """
        if not full_stage_order or name not in full_stage_order:
            self._stage_order.append(name)
            return
        target = full_stage_order.index(name)
        insert_pos = len(self._stage_order)
        for i, existing in enumerate(self._stage_order):
            existing_idx = full_stage_order.index(existing) if existing in full_stage_order else None
            if existing_idx is not None and existing_idx > target:
                insert_pos = i
                break
        self._stage_order.insert(insert_pos, name)

    def load(self, path: Path, stage_order: list[str], full_stage_order: Optional[list[str]] = None):
        if not path.exists():
            return

        order_file = path / _STAGE_ORDER_FILE
        if order_file.exists():
            # The saved order records where stages actually ran last time. It is
            # authoritative ONLY for stages the current config no longer declares
            # (disabled or removed since): those must keep their original position
            # rather than be treated as unrecognised and appended at the end (see
            # save() above). For every stage the current config DOES declare, the
            # current config wins.
            #
            # Letting the saved order win outright silently defeated reordering a
            # stage in config.yaml: _value() bounds its search at the *current*
            # stage's index in this list, so a stage moved earlier in the config
            # still sat late here and stayed invisible to everything that now runs
            # after it. Observed on the Rainier capture -- Object Distribution /
            # Distribution Synthesis / Grass Cover were moved ahead of Scene
            # Generation, but the cache still ordered them after Scene Animation,
            # so Scene Generation read OBJECT_COUNT = 417 (Object Instance
            # Refinement's) instead of 7078 and dropped all 6,661 painted grass
            # instances plus every synthesized distribution point on the floor.
            saved = list(parse_json(order_file.read_text()))
            self._stage_order = list(stage_order)
            placed = set(self._stage_order)
            for i, name in enumerate(saved):
                if name in placed:
                    continue
                # Anchor a saved-only stage against the next stage after it in the
                # saved order that we've already placed, so it lands in the same
                # relative spot it originally ran in. Previously-inserted saved-only
                # stages count as anchors too, which keeps runs of them in order.
                anchor = next((s for s in saved[i + 1:] if s in placed), None)
                position = self._stage_order.index(anchor) if anchor is not None else len(self._stage_order)
                self._stage_order.insert(position, name)
                placed.add(name)
        else:
            # No saved order (cache predates this fix). Best effort: use the
            # currently-enabled order as a base, then place anything unrecognised
            # (found on disk below) by its position in the full declared config
            # instead of wherever filesystem iteration happens to encounter it.
            self._stage_order = list(stage_order)

        self._load_directory(path, self._state)
        for stage_path in sorted(path.iterdir()):
            if stage_path.is_dir():
                stage_name = stage_path.name
                if stage_name not in self._stage_state:
                    self._stage_state[stage_name] = {}
                if stage_name not in self._stage_order:
                    self._insert_in_declared_order(stage_name, full_stage_order)
                self._load_directory(stage_path, self._stage_state[stage_name])

    def _load_directory(self, path: Path, target: dict):
        for meta_file in path.glob("*.meta"):
            name = meta_file.stem
            try:
                value = ContextValue(name)
                value.read(path)
                target[name] = value
            except Exception as e:
                _log.warning(f"Skipping '{name}' in {path}: {e}")

    def log_state(self):
        def _emit_values(values: dict, indent: str):
            items = sorted(values.items())
            for i, (name, value) in enumerate(items):
                connector = "└──" if i == len(items) - 1 else "├──"
                _log.info(f"{indent}{connector} {name}: {value.describe()}")

        _log.info("PipelineContext:")
        if self._state:
            has_stages = bool(self._stage_state)
            connector = "├──" if has_stages else "└──"
            _log.info(f" {connector} [global]")
            _emit_values(self._state, " │   " if has_stages else "     ")

        stages = list(self._stage_state.items())
        for i, (stage_name, values) in enumerate(stages):
            connector = "└──" if i == len(stages) - 1 else "├──"
            _log.info(f" {connector} [{stage_name}]")
            _emit_values(values, "     " if i == len(stages) - 1 else " │   ")
        