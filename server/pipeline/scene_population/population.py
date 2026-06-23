import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.object import Object3D

# Categories that should not be placed on roads or rivers
_AVOID_ROAD   = {"tree", "bush", "animal", "bench"}
_AVOID_RIVER  = {"tree", "bush", "person", "vehicle", "furniture", "bench", "sign", "other"}
# Categories preferentially placed near roads
_NEAR_ROAD    = {"bench", "sign", "person", "vehicle"}


class ScenePopulationConfiguration(PipelineStageConfiguration):
    def __init__(self, *args, fill_factor: float = 2.5, max_per_category: int = 50,
                 scale_jitter: float = 0.2, seed: int = 42, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        # A new object is placed when its nearest existing neighbour is farther
        # than fill_factor * typical_spacing for that category.
        self.fill_factor = fill_factor
        self.max_per_category = max_per_category
        # Uniform scale variation: final scale = 1 ± scale_jitter
        self.scale_jitter = scale_jitter


class ScenePopulationStage(PipelineStage):
    """
    Extrapolates detected objects into uncovered terrain regions using the
    spatial distribution built by ObjectDistributionStage.

    For each semantic category it:
      1. Identifies gaps — terrain grid cells whose nearest existing object
         of that category is farther than fill_factor * typical_spacing.
      2. Samples candidate positions inside those gaps via Poisson-disc-like
         filtering on a uniform grid.
      3. Places cloned Object3D instances (referencing an existing mesh_{i} key)
         with random Y-rotation and mild scale jitter.  Heights are sampled from
         the terrain height map.

    Reads:  ContextKey.OBJECT_DISTRIBUTION, ContextKey.SCENE,
            ContextKey.HEIGHT_MAP, ContextKey.HEIGHT_MAP_PARAMS
    Writes: ContextKey.SCENE  (Object3D entries appended in-place)
    """

    @classmethod
    def config_class(cls):
        return ScenePopulationConfiguration

    def _sample_terrain_height(self, x: float, z: float, height_map, params: dict) -> float:
        if height_map is None or params is None:
            return 0.0
        grid_size = params.get("grid_size_meters", 100.0)
        x_half = grid_size / 2.0
        z_far  = grid_size / 2.0
        hm = height_map.depth
        h_hm, w_hm = hm.shape
        row = ((z + z_far)  / (2.0 * z_far)   * (h_hm - 1))
        col = ((x + x_half) / grid_size        * (w_hm - 1))
        row = np.clip(row, 0, h_hm - 1)
        col = np.clip(col, 0, w_hm - 1)
        return float(map_coordinates(hm, [[row], [col]], order=1, mode="nearest")[0])

    def _is_blocked(self, x: float, z: float, category: str, graph) -> bool:
        """Return True if this position should be rejected given the linear graph."""
        if graph is None:
            return False
        if category in _AVOID_ROAD and graph.is_near(x, z, 1.5, types={"road"}):
            return True
        if category in _AVOID_RIVER and graph.is_near(x, z, 2.0, types={"river"}):
            return True
        return False

    def _road_bias(self, candidates: np.ndarray, category: str, graph) -> np.ndarray:
        """
        For road-affine categories, up-weight candidates near roads by
        returning a filtered subset that prefers proximity to roads.
        """
        if graph is None or category not in _NEAR_ROAD or not graph.structures:
            return candidates
        road_dists = np.array([
            graph.nearest(float(pt[0]), float(pt[1]), types={"road"})[0]
            for pt in candidates
        ], dtype=np.float32)
        # Keep candidates within 8 m of a road (or fall back to all if none qualify)
        near = candidates[road_dists < 8.0]
        return near if len(near) >= 3 else candidates

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: ScenePopulationConfiguration = self.config
        rng = np.random.default_rng(cfg.seed)

        distribution = context.input_object(ContextKey.OBJECT_DISTRIBUTION)
        scene        = context.input_scene(ContextKey.SCENE)
        height_map   = context.input_depth(ContextKey.HEIGHT_MAP)
        params       = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        graph        = context.input_object(ContextKey.LINEAR_GRAPH)

        if distribution is None or scene is None:
            self.log_warning("Missing distribution or scene — skipping population")
            return context

        if graph is not None:
            self.log_info(f"Grounding population against linear graph: {graph.summary()}")

        grid_size = (params or {}).get("grid_size_meters", 100.0)
        x_half = grid_size / 2.0
        z_far  = grid_size / 2.0

        task = self.create_progress(len(distribution), "Populating scene...")
        total_placed = 0

        for category, data in distribution.items():
            self.advance_progress(task)

            existing_xz = np.array(data["positions_xz"], dtype=np.float32)
            spacing     = data["typical_spacing"]
            mean_w      = data["mean_width"]
            mean_h      = data["mean_height"]
            indices     = data["indices"]
            threshold   = spacing * cfg.fill_factor

            # Build a candidate grid at typical_spacing resolution
            step = max(spacing, 0.5)
            xs = np.arange(-x_half + step / 2, x_half, step, dtype=np.float32)
            zs = np.arange(-z_far  + step / 2, z_far,  step, dtype=np.float32)
            gx, gz = np.meshgrid(xs, zs)
            candidates = np.stack([gx.ravel(), gz.ravel()], axis=-1)

            # Keep only candidates that are far enough from all existing objects
            tree = KDTree(existing_xz)
            dists, _ = tree.query(candidates)
            gap_candidates = candidates[dists > threshold]

            if len(gap_candidates) == 0:
                self.log_info(f"  {category}: no gaps to fill")
                continue

            # Apply road-proximity bias for road-affine categories
            gap_candidates = self._road_bias(gap_candidates, category, graph)

            # Greedy Poisson-disc pass: place points at least spacing apart
            rng.shuffle(gap_candidates)
            placed_xz: list[np.ndarray] = []
            placed_tree_pts: list[list] = list(existing_xz.tolist())

            for pt in gap_candidates:
                if len(placed_xz) >= cfg.max_per_category:
                    break
                x, z = float(pt[0]), float(pt[1])
                # Reject positions that conflict with linear structures
                if self._is_blocked(x, z, category, graph):
                    continue
                if placed_tree_pts:
                    d, _ = KDTree(placed_tree_pts).query(pt)
                    if d < spacing:
                        continue
                placed_xz.append(pt)
                placed_tree_pts.append(pt.tolist())

            for pt in placed_xz:
                x, z = float(pt[0]), float(pt[1])
                y = self._sample_terrain_height(x, z, height_map, params)

                src_idx = int(rng.choice(indices))
                mesh_key = f"mesh_{src_idx}"

                scale = float(1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter))
                y_rot = float(rng.uniform(0.0, 360.0))

                obj = Object3D.mesh(mesh_key, x=x, y=y, z=z)
                obj.set_scale(scale * mean_w, scale * mean_h, scale * mean_w)
                obj.set_rotation(0.0, y_rot, 0.0)
                scene.add_object(obj)

            total_placed += len(placed_xz)
            self.log_info(f"  {category}: placed {len(placed_xz)} extra objects")

        self.finish_progress(task)
        self.log_info(f"Scene population: {total_placed} objects added")

        context.add_scene(ContextKey.SCENE, scene)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.scene(ContextKey.SCENE) is not None

    def model_names(self) -> list[str]:
        return []
