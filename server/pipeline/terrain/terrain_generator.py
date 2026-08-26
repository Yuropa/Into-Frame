import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import Delaunay
from typing import Optional
import trimesh
import trimesh.visual.material
import PIL.Image

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from scene.camera import CameraIntrinsics
from scene.mesh import Mesh
from pipeline.panorama_segmentation.panorama_region_result import RegionType


def _prune_small_water(water: np.ndarray, *, grid_size_meters: float, min_area_m2: float) -> np.ndarray:
    """Drop water components too small to be a body of water.

    Region typing hands this stage every cell it called WATER, with no notion of
    how big the resulting body is, and a handful of cells is enough to produce a
    real flat water surface in the scene. Measured on the Mount Rainier capture --
    an alpine meadow with no lake in it -- the region map carried 402 WATER cells,
    0.98 m2 in total, ALL of them between 3.0 m and 6.0 m from the camera. They
    became an 894-vertex water mesh spanning x[-4.3, 5.2] z[-4.7, 4.9] sitting at
    y -1.70..-2.18 against terrain at -1.92: a puddle wrapped around the viewer's
    feet, interpenetrating the ground it was drawn on.

    Their source was panorama rows 1218-1362 -- below the horizon, so
    PanoramaRegionStage's _clean_nadir_band never looks at them, and above the
    nadir cutoff, so nothing else did either. It is melting snow typed as water,
    which is a perfectly reasonable per-pixel call and a nonsensical lake.

    Area is the honest discriminator: a lake, river or sea covers tens of square
    metres at minimum, so anything under min_area_m2 is a typing artifact. Applied
    per connected component rather than in total, so a genuine lake is never
    dropped because some speckle elsewhere shares its class -- and 8-connectivity,
    so a body pinched to a diagonal thread at one point stays one body.
    """
    if not water.any() or min_area_m2 <= 0.0:
        return water
    from scipy import ndimage

    cell_area = (grid_size_meters / water.shape[0]) * (grid_size_meters / water.shape[1])
    labels, count = ndimage.label(water, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return water
    areas = np.bincount(labels.ravel()) * cell_area
    too_small = np.flatnonzero(areas < min_area_m2)
    if too_small.size == 0:
        return water
    return water & ~np.isin(labels, too_small)


class TerrainMeshGenerator:
    @staticmethod
    def generate(
        height_map: Depth,
        grid_size_meters: float,
        inner_min_dist: float = 1.5,
        outer_min_dist: float = 6.0,
        n_boundary: int = 12,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        noise_blend_floor: float = 0.15,
        noise_seed: int = 42,
        panorama: Optional[Panorama] = None,
        texture: Optional[PIL.Image.Image] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        precomputed_texture: Optional[PIL.Image.Image] = None,
        texture_tile_factor: float = 1.0,
        region_map: Optional[Depth] = None,
        water_depression_m: float = 0.5,
        min_water_area_m2: float = 12.0,
        water_level_percentile: float = 25.0,
        observed_mask: Optional[Depth] = None,
        component_id: Optional[Depth] = None,
        formation_depression_m: float = 0.5,
        sky_mask: Optional[np.ndarray] = None,
        pano_uv_u: Optional[Depth] = None,
        pano_uv_v: Optional[Depth] = None,
    ) -> tuple[Mesh, Optional[Mesh], Optional[np.ndarray]]:
        """
        Build a variable-density terrain mesh from a height map using Poisson
        disc sampling and Delaunay triangulation.

        A dense Poisson disc pass covers the inner region; a sparser pass covers
        the full domain.  Boundary points anchor the rectangle edges.  All points
        are triangulated with Delaunay, giving a natural, non-axis-biased LOD.

        panorama  : Panorama for equirectangular vertex-colour baking (full 360° coverage).
        texture   : PIL image for pinhole UV mapping via CameraIntrinsics (FOV-limited).
        intrinsics: required when texture is supplied.
        region_map: optional top-down RegionType grid (same convention as height_map).
                    Where present, WATER vertices are depressed below the water
                    surface (so an animated water plane never clips through the
                    lakebed) and a separate flat water Mesh is returned, built from
                    the exact same triangulation so its shoreline matches the
                    terrain mesh's water hole precisely.
        water_depression_m: how far below the water surface the lakebed is carved.
        water_level_percentile: which percentile of a water body's own reconstructed
                    terrain height becomes its single flat surface level. See
                    _level_water for why it is a low percentile and not the median.
        observed_mask: optional HEIGHT_MAP_OBSERVED_MASK (same grid as height_map,
                    True where a cell has a genuine direct point-cloud measurement,
                    already restored verbatim through Reconstruction and Noise
                    Refinement). When supplied, the vertex noise below is
                    suppressed on real cells and only applied on synthetic/
                    interpolated ones -- otherwise this stage's own noise would be
                    the one place in the pipeline that displaces real geometry
                    regardless of how confidently it was observed.
        component_id: optional HEIGHT_MAP_COMPONENT_ID (same grid as height_map;
                    see HeightMapGenerator._label_ground_components). Cells with
                    id > 1 belong to a connected ground component other than the
                    largest/base one -- a separate landmass, an isolated rock
                    formation, anything real but genuinely disconnected from this
                    mesh's own ground. Those get extracted as their own separate
                    mesh elsewhere (see generate_component_mesh), so *this* --
                    the base terrain -- mirrors the water-depression pattern
                    above and pushes its own vertices there below the real
                    height by formation_depression_m, instead of showing the
                    same geometry twice or leaving a hole an extracted mesh with
                    any gap of its own could be seen through.
        formation_depression_m: how far below its real height the base terrain
                    is carved wherever a separate component's own mesh covers it.
        sky_mask: optional PANORAMA_SKY_MASK, forwarded to the panorama UV
                    projection (Panorama.mesh_uvs) so a vertex whose
                    reconstructed elevation angle overshoots into the sky --
                    e.g. a mountainside vertex sitting on a heuristic slope
                    envelope rather than real measured geometry -- gets
                    snapped back down to the nearest real (non-sky) content
                    in the same panorama column instead of literally
                    texturing sky onto the mesh.
        pano_uv_u, pano_uv_v: optional HEIGHT_MAP_PANO_U/_V (same grid as
                    height_map; see HeightMapGenerator._panorama_uv_from_height).
                    Each observed vertex's own true panorama UV, preferred over
                    re-deriving UV from this stage's own (noised/reconstructed)
                    Y_pos -- see the panorama-UV block below for why that
                    matters. No effect unless panorama is also supplied.

        Returns (terrain_mesh, water_mesh, panorama_uv, uv_folds_fixed).
        uv_folds_fixed is how many triangles _fix_uv_discontinuities collapsed --
        a direct measure of how much of this terrain the panorama never saw.
        water_mesh is None
        when region_map is not supplied or the panorama has no detected water.
        panorama_uv is the final per-vertex panorama UV (None unless panorama
        is supplied) aligned 1:1 with terrain_mesh's own (possibly seam-
        duplicated) vertices -- intended for embedding as a second UV channel
        so Unity's live shader can sample it directly instead of recomputing
        an equirect projection from world position every frame.
        """
        z_far  = z_far if z_far is not None else grid_size_meters / 2.0
        x_half = grid_size_meters / 2.0
        hm     = height_map.depth  # (H, W) float32

        # ── Poisson disc sampling ─────────────────────────────────────────
        all_xz = TerrainMeshGenerator._poisson_disc_xz(
            x_half=x_half,
            z_far=z_far,
            inner_min_dist=inner_min_dist,
            outer_min_dist=outer_min_dist,
            n_boundary=n_boundary,
            seed=noise_seed,
        )
        all_xz = np.unique(np.round(all_xz, 4), axis=0)

        X_pos = all_xz[:, 0].astype(np.float32)
        Z_pos = all_xz[:, 1].astype(np.float32)

        # ── Delaunay triangulation ────────────────────────────────────────
        faces = Delaunay(all_xz).simplices[:, ::-1].astype(np.int32)

        # ── Sample height map at every vertex ─────────────────────────────
        h_hm, w_hm = hm.shape
        row_coords = ((Z_pos + z_far)  / (2.0 * z_far)        * (h_hm - 1)).clip(0, h_hm - 1)
        col_coords = ((X_pos + x_half) / grid_size_meters      * (w_hm - 1)).clip(0, w_hm - 1)

        Y_pos = map_coordinates(
            hm, [row_coords, col_coords], order=1, mode="nearest",
        ).astype(np.float32)
        Y_pos = np.nan_to_num(Y_pos, nan=0.0)

        # ── Per-vertex observed weight ──────────────────────────────────────
        # Reused below for both panorama-UV selection (prefer each observed
        # vertex's own true UV) and noise suppression (don't add cosmetic
        # relief on top of real point-cloud geometry).
        observed_field = None
        observed_weight = None
        if observed_mask is not None and observed_mask.depth.shape == hm.shape:
            # sigma=1 matches the feather TerrainReconstructionStage/
            # TerrainNoiseRefinementStage already use when restoring observed
            # cells, so the transition doesn't leave a sudden seam.
            observed_field = gaussian_filter(observed_mask.depth.astype(np.float64), sigma=1.0)
            observed_weight = map_coordinates(
                observed_field, [row_coords, col_coords], order=1, mode="nearest",
            ).astype(np.float32).clip(0.0, 1.0)

        # ── Panorama UV ──────────────────────────────────────────────────────
        # This -- not the embedded-preview texture branch further down -- is
        # what Unity's terrain shader actually samples live for the panorama
        # layer, so it has to be right regardless of which texture ends up
        # embedded as the GLB preview. Prefer each observed vertex's own true
        # panorama UV (HeightMapGenerator._panorama_uv_from_height, captured
        # from the height as actually measured, before Terrain Reconstruction/
        # Noise Refinement/this function's own noise below could perturb it --
        # see that function's docstring for why re-deriving UV from a possibly-
        # perturbed Y instead amplifies tiny height errors into a visible
        # mismatch between the mesh's own silhouette and the photographed
        # ridge line) over re-deriving UV from this stage's own Y_pos, which
        # has no such guarantee. Unobserved (interpolated/synthetic) vertices
        # have no true UV to prefer, so they keep the position-derived one.
        pano_uv = None
        # How many triangles _fix_uv_discontinuities had to collapse. Reported by
        # TerrainMeshStage rather than logged here (this class has no logger), and
        # worth watching: it is a direct measure of how much of the terrain the
        # panorama could not actually see.
        uv_folds_fixed = 0
        if panorama is not None:
            pano_uv = panorama.mesh_uvs(
                np.stack([X_pos, Y_pos, Z_pos], axis=-1), sky_mask=sky_mask,
            )
            if (
                pano_uv_u is not None and pano_uv_v is not None
                and pano_uv_u.depth.shape == hm.shape and pano_uv_v.depth.shape == hm.shape
                and observed_weight is not None
            ):
                # order=0 (nearest), not bilinear: U wraps around (it's derived
                # from longitude), so linearly blending stored U values straddling
                # that wrap averages two opposite-side values into one nonsense
                # one in the middle -- e.g. 0.99 and 0.01 blend to ~0.5, the wrong
                # side of the panorama entirely. Nearest-neighbour never blends
                # across the seam; at this grid's resolution (~5 cm/cell) picking
                # one side's exact stored value instead of an interpolated one
                # between neighbours is not a visible tradeoff.
                obs_u = map_coordinates(
                    pano_uv_u.depth, [row_coords, col_coords], order=0, mode="nearest",
                ).astype(np.float32)
                obs_v = map_coordinates(
                    pano_uv_v.depth, [row_coords, col_coords], order=0, mode="nearest",
                ).astype(np.float32)
                use_stored = (observed_weight > 0.5) & np.isfinite(obs_u) & np.isfinite(obs_v)
                if use_stored.any():
                    pano_uv = pano_uv.copy()
                    pano_uv[use_stored, 0] = obs_u[use_stored]
                    pano_uv[use_stored, 1] = obs_v[use_stored]

            # Seam-fix now (not in the texture branch below) so every
            # per-vertex array computed after this point -- water/formation
            # masks, noise, the final UV0 branch -- naturally operates on the
            # right (possibly larger) vertex/face set. Everything below here
            # resamples height_map/region_map/component_id fresh from each
            # vertex's own (X, Z), so duplicated vertices need no special
            # handling beyond X_pos/Y_pos/Z_pos/faces/row_coords/col_coords
            # themselves being refreshed to match.
            vertices_seam = np.stack([X_pos, Y_pos, Z_pos], axis=-1).astype(np.float32)
            vertices_seam, faces, pano_uv = TerrainMeshGenerator._fix_uv_seam(
                vertices_seam, faces, pano_uv,
            )
            # After the seam fix, never before: that one unwraps longitude, and this
            # one must read an already-unwrapped U to tell a wrap from a real step.
            vertices_seam, faces, pano_uv, uv_folds_fixed = (
                TerrainMeshGenerator._fix_uv_discontinuities(
                    vertices_seam, faces, pano_uv, panorama.height,
                )
            )
            X_pos, Y_pos, Z_pos = vertices_seam[:, 0], vertices_seam[:, 1], vertices_seam[:, 2]
            row_coords = ((Z_pos + z_far)  / (2.0 * z_far)   * (h_hm - 1)).clip(0, h_hm - 1)
            col_coords = ((X_pos + x_half) / grid_size_meters * (w_hm - 1)).clip(0, w_hm - 1)
            if observed_field is not None:
                observed_weight = map_coordinates(
                    observed_field, [row_coords, col_coords], order=1, mode="nearest",
                ).astype(np.float32).clip(0.0, 1.0)

        # ── Water mask ──────────────────────────────────────────────────────
        # Sampled from the same reconstructed grid the DEM solve already pinned
        # flat/sloped at the water surface (see TerrainReconstructionStage's
        # WATER_CHAINS handling), so Y_pos at water vertices is already the
        # correct water-surface elevation — captured here, before noise and the
        # lakebed depression below are applied to the terrain copy.
        is_water = None
        water_Y = None
        if region_map is not None:
            rm = region_map.depth
            h_rm, w_rm = rm.shape
            row_rm = ((Z_pos + z_far)  / (2.0 * z_far)   * (h_rm - 1)).clip(0, h_rm - 1)
            col_rm = ((X_pos + x_half) / grid_size_meters * (w_rm - 1)).clip(0, w_rm - 1)
            water_grid = _prune_small_water(
                rm.astype(np.int16) == int(RegionType.WATER),
                grid_size_meters=grid_size_meters,
                min_area_m2=min_water_area_m2,
            )
            # order=0 so this stays a nearest-neighbour lookup of a boolean field;
            # interpolating a mask would produce fractional "water" along every
            # shoreline and then threshold it back at an arbitrary place.
            region_idx = map_coordinates(
                water_grid.astype(np.uint8), [row_rm, col_rm], order=0, mode="nearest"
            )
            is_water = region_idx.astype(bool)
            if is_water.any():
                # rm is the region grid; hm is the height grid the water sits on.
                # Both are the same (H, W) top-down convention, so a water cell in
                # one indexes the same ground in the other.
                water_Y = TerrainMeshGenerator._level_water(
                    water_grid, hm, is_water, Y_pos, row_rm, col_rm,
                    water_level_percentile,
                )

        # ── Formation mask ────────────────────────────────────────────────────
        # Same idea as the water mask above: capture each formation vertex's real
        # height before noise/depression, so the base terrain can be pushed below
        # it later without disturbing the value generate_component_mesh needs to
        # build that formation's own separate mesh.
        is_formation = None
        formation_Y = None
        if component_id is not None and component_id.depth.shape == hm.shape:
            cid = component_id.depth
            h_c, w_c = cid.shape
            row_c = ((Z_pos + z_far)  / (2.0 * z_far)   * (h_c - 1)).clip(0, h_c - 1)
            col_c = ((X_pos + x_half) / grid_size_meters * (w_c - 1)).clip(0, w_c - 1)
            vertex_component_id = map_coordinates(
                cid, [row_c, col_c], order=0, mode="nearest"
            ).round().astype(np.int32)
            is_formation = vertex_component_id > 1
            if is_formation.any():
                formation_Y = Y_pos.copy()

        # ── Noise, blended in with distance from origin ───────────────────
        noise_tex = TerrainMeshGenerator._smooth_noise((256, 256), seed=noise_seed)
        nr = (row_coords / (h_hm - 1) * 255).clip(0, 255)
        nc = (col_coords / (w_hm - 1) * 255).clip(0, 255)
        noise_vals = map_coordinates(noise_tex, [nr, nc], order=1, mode="wrap").astype(np.float32)

        # blend ramps from noise_blend_floor at the origin (camera position) up to
        # 1.0 at the domain corner. A floor of 0 would leave the ground directly
        # under the user perfectly noise-free, which reads as an artificially flat
        # disc even where the reconstructed height map itself has real relief.
        r_end = np.hypot(x_half, z_far)
        blend = noise_blend_floor + (1.0 - noise_blend_floor) * (np.hypot(X_pos, Z_pos) / r_end).clip(0.0, 1.0)
        noise_add = noise_vals * noise_amplitude * blend

        if observed_weight is not None:
            # Suppress noise on vertices sitting on real point-cloud geometry --
            # observed_weight already reflects the (possibly seam-duplicated)
            # vertex set computed above.
            noise_add *= (1.0 - observed_weight)

        if is_water is not None:
            # A lakebed shouldn't get the same surface-relief noise as dry
            # ground — zero it under water so the depression below stays clean.
            noise_add = np.where(is_water, 0.0, noise_add)
        if is_formation is not None:
            # Hidden beneath an extracted formation's own mesh -- no point
            # texturing/relief-shaping ground nobody will ever see.
            noise_add = np.where(is_formation, 0.0, noise_add)
        Y_pos += noise_add

        if formation_Y is not None:
            # Push the base terrain below each extracted formation's own real
            # height by a fixed margin, mirroring the water depression below --
            # never a literal hole, just hidden geometry underneath, so nothing
            # pokes through and no gap is visible from an angle that looks past
            # the extracted mesh's own edge.
            Y_pos = np.where(is_formation, formation_Y - formation_depression_m, Y_pos)

        if water_Y is not None:
            # Carve the lakebed a fixed margin below the (already correct,
            # noise-free) water surface so an animated water plane's wave
            # amplitude never exposes the terrain underneath it. Applied after
            # the formation depression so real water always wins in the rare
            # case a cell is classified as both (e.g. the submerged base of a
            # rock formation that also pokes above the surface elsewhere).
            Y_pos = np.where(is_water, water_Y - water_depression_m, Y_pos)

        # ── Vertex array ──────────────────────────────────────────────────
        vertices = np.stack([X_pos, Y_pos, Z_pos], axis=-1).astype(np.float32)

        # ── Colour / texture ──────────────────────────────────────────────
        if precomputed_texture is not None:
            # UVs scaled by texture_tile_factor so the tile repeats that many
            # times across the full terrain grid. Values > 1 are valid in glTF
            # (default sampler wrap = REPEAT) and give higher texel density.
            # trimesh's GLTF exporter unconditionally flips V on export (assumes OBJ-style
            # V-up input); pre-flip here so the exported glTF V lands at row 0 = image top,
            # matching how the precomputed texture's rows are laid out — otherwise the
            # texture renders upside-down in any spec-conformant glTF viewer (Unity included).
            u = ((X_pos + x_half) / (2.0 * x_half) * texture_tile_factor).astype(np.float32)
            v = (1.0 - (Z_pos + z_far) / (2.0 * z_far) * texture_tile_factor).astype(np.float32)
            uv = np.stack([u, v], axis=-1)
            tri_mesh = TerrainMeshGenerator._textured_mesh(vertices, faces, uv, precomputed_texture)
        elif panorama is not None:
            # pano_uv (computed above, already seam-fixed against the final
            # vertex/face set) projects the mesh directly onto the panorama's
            # own pixel grid instead of resampling the panorama into a
            # top-down raster first — no intermediate bake, so no world-space
            # <-> equirectangular resampling distortion (an earlier top-down
            # bake pinwheeled near the origin and went blocky near the
            # horizon; removed entirely now that every panorama-textured
            # branch goes through this same corrected per-vertex UV instead).
            tri_mesh = TerrainMeshGenerator._textured_mesh(vertices, faces, pano_uv, panorama.image)
        elif texture is not None and intrinsics is not None:
            uv = TerrainMeshGenerator._uvs_pinhole(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], intrinsics
            )
            tri_mesh = TerrainMeshGenerator._textured_mesh(vertices, faces, uv, texture)
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # ── Separate flat water mesh ─────────────────────────────────────────
        # Built from the same triangulation as the terrain mesh (faces whose
        # three vertices are all WATER), so its shoreline lines up exactly with
        # the depressed lakebed above — no gap, no overlap. Uses the pre-noise,
        # pre-depression water elevation, so it's flat wherever the DEM solve's
        # water-chain pinning made it flat.
        water_mesh: Optional[Mesh] = None
        if water_Y is not None:
            water_face_mask = is_water[faces].all(axis=1)
            if water_face_mask.any():
                water_vertices = np.stack([X_pos, water_Y, Z_pos], axis=-1).astype(np.float32)
                # Mirrored by Unity's IntoFrame/WaterSurface shader, which replaces
                # this material outright and re-declares the same two numbers as its
                # own defaults -- keep the two in step. It is also what the exported
                # glTF carries, and what renders unaided in any viewer with no water
                # shader of its own, so this is not merely a fallback.
                #
                # Alpha 0.90, raised from 0.75: the lakebed under this sheet is
                # carved water_depression_m below it and painted with the panorama's
                # own water pixels, so a quarter transmission showed a second,
                # darker, wrong-scale copy of the water through the water. That is
                # what read as tinted glass. Real water is near-opaque at the grazing
                # angles that make up almost all of a standing viewer's sightlines.
                # roughness 0.08 == the shader's smoothness 0.92, raised with it from
                # 0.15/0.85. Water is close to optically smooth; the old value was
                # broadening the specular into a sheen instead of a reflection, which
                # on a client with no water shader is the only specular there is.
                water_material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.10, 0.30, 0.45, 0.90],
                    metallicFactor=0.0,
                    roughnessFactor=0.08,
                    alphaMode="BLEND",
                )
                water_tri_mesh = trimesh.Trimesh(
                    vertices=water_vertices,
                    faces=faces[water_face_mask],
                    visual=trimesh.visual.TextureVisuals(material=water_material),
                    process=False,
                )
                water_tri_mesh.remove_unreferenced_vertices()
                water_mesh = Mesh(water_tri_mesh)

        return Mesh(tri_mesh), water_mesh, pano_uv, uv_folds_fixed

    # ── Physics (collision) mesh ────────────────────────────────────────────────

    @staticmethod
    def generate_physics_mesh(
        height_map: Depth,
        grid_size_meters: float,
        inner_min_dist: float = 0.5,
        outer_min_dist: float = 8.0,
        n_boundary: int = 24,
        noise_seed: int = 42,
        region_map: Optional[Depth] = None,
        water_depression_m: float = 0.5,
        min_water_area_m2: float = 12.0,
        observed_mask: Optional[Depth] = None,
        component_id: Optional[Depth] = None,
        formation_depression_m: float = 0.5,
    ) -> Mesh:
        """
        Build a coarse, geometry-only collision mesh for the terrain.

        Just generate() again -- same Poisson-disc/Delaunay sampling (so it
        keeps the near-camera density bias, and scales with grid_size_meters
        the same way) and the same water/formation depression (so the
        collider is correctly carved under a lake or wherever an extracted
        formation's own mesh/physics mesh sits) -- just no texture/panorama
        (geometry-only branch) and no cosmetic noise, and with much sparser
        min_dist defaults than the visual mesh's own (default inner/outer
        0.10/2.0 m): a MeshCollider only needs to be close enough to walk/
        land on, not the visual mesh's full density.
        """
        # Four values, not three: generate() also reports how many triangles
        # _fix_uv_discontinuities collapsed. Irrelevant here -- this branch passes
        # no panorama, so there is no UV to fix and the count is always 0.
        mesh, _water, _uv, _folds = TerrainMeshGenerator.generate(
            height_map=height_map,
            grid_size_meters=grid_size_meters,
            inner_min_dist=inner_min_dist,
            outer_min_dist=outer_min_dist,
            n_boundary=n_boundary,
            z_far=grid_size_meters / 2.0,
            noise_amplitude=0.0,
            noise_blend_floor=0.0,
            noise_seed=noise_seed,
            region_map=region_map,
            water_depression_m=water_depression_m,
            # Must match generate()'s own value or the collider is carved for a
            # different set of lakes than the visual mesh shows.
            min_water_area_m2=min_water_area_m2,
            observed_mask=observed_mask,
            component_id=component_id,
            formation_depression_m=formation_depression_m,
        )
        return mesh

    # ── Component (formation) meshes ────────────────────────────────────────────

    @staticmethod
    def generate_component_mesh(
        height_map: Depth,
        component_id: Depth,
        target_id: int,
        grid_size_meters: float,
        min_dist: float = 0.5,
        n_boundary: int = 8,
        seed: int = 42,
        footprint_margin: float = 2.0,
    ) -> Optional[tuple[Mesh, float, float, float, float]]:
        """
        Build an independent geometry-only mesh for one non-primary ground
        component (see HeightMapGenerator._label_ground_components) -- a
        separate landmass, an isolated rock formation, anything real but
        disconnected from the base terrain. Uses real height directly, no
        depression (that's the base terrain's own job -- see the
        component_id/formation_depression_m parameters on generate() above,
        which hide the base terrain's copy of this same footprint below it).

        Scoped to the component's own bounding box, not the whole grid: a
        UNIFORM-density Poisson disc (no near-camera falloff -- a formation
        doesn't have its own obvious "near" reference point the way the main
        terrain has the camera) fills the bounding box, then both points and
        triangles are filtered down to the component's own footprint mask --
        Delaunay triangulates the convex hull of whatever points survive, so
        without also dropping faces whose centroid falls outside the
        footprint, an irregular or concave component (a crescent-shaped
        sandbar, say) would get bridged over with incorrect flat triangles
        cutting across the concave part.

        No texture/UV is set here -- TerrainTextureGenerationStage bakes and
        applies each formation's own texture separately (see
        pipeline/terrain/pattern_texture.py's x_center/z_center support),
        using the returned (x_center, z_center, x_half, z_half) to align its
        canvas and UVs with this mesh's own local footprint.

        Returns None if the component doesn't exist, or its footprint is too
        small/degenerate to mesh (fewer than 3 surviving points, or a
        bounding half-extent under min_dist).
        """
        hm = height_map.depth
        cid = component_id.depth
        h, w = cid.shape
        if hm.shape != cid.shape:
            return None
        half = grid_size_meters / 2.0

        footprint = np.round(cid).astype(np.int32) == target_id
        if not footprint.any():
            return None

        rows, cols = np.where(footprint)
        z_coords = rows.astype(np.float64) / (h - 1) * grid_size_meters - half
        x_coords = cols.astype(np.float64) / (w - 1) * grid_size_meters - half
        x_center = float((x_coords.min() + x_coords.max()) / 2.0)
        z_center = float((z_coords.min() + z_coords.max()) / 2.0)
        x_half = float((x_coords.max() - x_coords.min()) / 2.0) + footprint_margin
        z_half = float((z_coords.max() - z_coords.min()) / 2.0) + footprint_margin
        if x_half < min_dist or z_half < min_dist:
            return None

        local_xz = TerrainMeshGenerator._poisson_disc_xz(
            x_half=x_half, z_far=z_half,
            inner_min_dist=min_dist, outer_min_dist=min_dist,
            n_boundary=n_boundary, seed=seed,
        )
        local_xz = np.unique(np.round(local_xz, 4), axis=0)

        world_x = local_xz[:, 0] + x_center
        world_z = local_xz[:, 1] + z_center

        # Shrink-wrap to the component's real (possibly irregular) footprint:
        # drop any point whose nearest grid cell isn't actually this component.
        row_i = np.clip(((world_z + half) / grid_size_meters * (h - 1)).round().astype(np.int64), 0, h - 1)
        col_i = np.clip(((world_x + half) / grid_size_meters * (w - 1)).round().astype(np.int64), 0, w - 1)
        in_footprint = footprint[row_i, col_i]
        local_xz = local_xz[in_footprint]
        world_x, world_z = world_x[in_footprint], world_z[in_footprint]
        row_i, col_i = row_i[in_footprint], col_i[in_footprint]
        if len(local_xz) < 3:
            return None

        try:
            faces = Delaunay(local_xz).simplices[:, ::-1].astype(np.int32)
        except Exception:
            return None  # degenerate point set (e.g. all collinear)

        # Drop faces bridging across a concave part of the footprint (see
        # docstring) -- centroid, in the same nearest-cell sense used above.
        centroid_x = world_x[faces].mean(axis=1)
        centroid_z = world_z[faces].mean(axis=1)
        c_row = np.clip(((centroid_z + half) / grid_size_meters * (h - 1)).round().astype(np.int64), 0, h - 1)
        c_col = np.clip(((centroid_x + half) / grid_size_meters * (w - 1)).round().astype(np.int64), 0, w - 1)
        faces = faces[footprint[c_row, c_col]]
        if len(faces) == 0:
            return None

        world_y = hm[row_i, col_i].astype(np.float32)
        world_y = np.nan_to_num(world_y, nan=0.0)
        vertices = np.stack([world_x, world_y, world_z], axis=-1).astype(np.float32)

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tri_mesh.remove_unreferenced_vertices()
        if len(tri_mesh.faces) == 0:
            return None

        return Mesh(tri_mesh), x_center, z_center, x_half, z_half

    @staticmethod
    def apply_component_texture(
        mesh: Mesh,
        tile: PIL.Image.Image,
        x_center: float,
        z_center: float,
        x_half: float,
        z_half: float,
    ) -> None:
        """
        Apply a formation mesh's own baked texture (see
        pipeline/terrain/pattern_texture.py, called with this same
        x_center/z_center/x_half/z_half so the canvas lines up) with simple
        orthographic top-down UVs local to the formation's own footprint --
        the standard "one texture, one UV per vertex" glTF case every other
        object mesh in this codebase already uses (confirmed against the
        Unity client: no per-triangle/texture-array support, so this -- not
        the terrain's own separately-transmitted SplatMaterial -- is how a
        formation mesh needs to carry its texture, baked directly into its
        own GLB).
        """
        verts = mesh.mesh.vertices
        u = ((verts[:, 0] - x_center + x_half) / (2.0 * x_half)).astype(np.float32)
        v = (1.0 - (verts[:, 2] - z_center + z_half) / (2.0 * z_half)).astype(np.float32)
        uv = np.stack([u, v], axis=-1)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tile.convert("RGB"),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        )
        mesh.mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    # ── Point generation ──────────────────────────────────────────────────────

    @staticmethod
    def _poisson_disc_xz(
        x_half: float,
        z_far: float,
        inner_min_dist: float,
        outer_min_dist: float,
        n_boundary: int,
        seed: int = 42,
        k: int = 30,
    ) -> np.ndarray:
        """
        Bridson's Poisson disc sampling with linearly varying radius.
        Spacing grows from inner_min_dist at the origin to outer_min_dist
        at the domain corner, giving a smooth continuous density falloff.
        """
        rng   = np.random.default_rng(seed)
        d_max = np.hypot(x_half, z_far)

        def radius_at(x: float, z: float) -> float:
            t = min(1.0, np.hypot(x, z) / d_max)
            return inner_min_dist + (outer_min_dist - inner_min_dist) * t

        # Background grid sized to the smallest possible radius
        cell = inner_min_dist / np.sqrt(2.0)
        cols = int(np.ceil(2.0 * x_half / cell)) + 2
        rows = int(np.ceil(2.0 * z_far  / cell)) + 2
        grid = np.full((rows, cols), -1, dtype=np.int32)

        pts_x: list[float] = []
        pts_z: list[float] = []
        active: list[int]  = []

        def to_grid(x: float, z: float):
            c = int((x + x_half) / cell)
            r = int((z + z_far)  / cell)
            return np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)

        def try_add(x: float, z: float) -> bool:
            r  = radius_at(x, z)
            gr, gc = to_grid(x, z)
            hw = int(np.ceil(r / cell)) + 1
            for dr in range(-hw, hw + 1):
                for dc in range(-hw, hw + 1):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] >= 0:
                        i = grid[nr, nc]
                        if np.hypot(x - pts_x[i], z - pts_z[i]) < r:
                            return False
            idx = len(pts_x)
            pts_x.append(x)
            pts_z.append(z)
            active.append(idx)
            grid[gr, gc] = idx
            return True

        try_add(0.0, 0.0)

        while active:
            i      = int(rng.integers(len(active)))
            ax, az = pts_x[active[i]], pts_z[active[i]]
            r      = radius_at(ax, az)
            placed = False
            for _ in range(k):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                dist  = rng.uniform(r, 2.0 * r)
                nx    = ax + dist * np.cos(angle)
                nz    = az + dist * np.sin(angle)
                if -x_half <= nx <= x_half and -z_far <= nz <= z_far:
                    if try_add(nx, nz):
                        placed = True
                        break
            if not placed:
                active.pop(i)

        pts = np.column_stack([pts_x, pts_z]).astype(np.float32)

        n  = n_boundary
        ex = np.linspace(-x_half, x_half, n, dtype=np.float32)
        ez = np.linspace(-z_far,  z_far,  n, dtype=np.float32)
        boundary_pts = np.concatenate([
            np.column_stack([ex,                                np.full(n, -z_far,  dtype=np.float32)]),
            np.column_stack([ex,                                np.full(n,  z_far,  dtype=np.float32)]),
            np.column_stack([np.full(n, -x_half, dtype=np.float32), ez]),
            np.column_stack([np.full(n,  x_half, dtype=np.float32), ez]),
        ]).astype(np.float32)

        return np.concatenate([pts, boundary_pts])

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _textured_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: np.ndarray,
        image: PIL.Image.Image,
    ) -> trimesh.Trimesh:
        """Shared UV-textured Trimesh construction for every texture-source branch."""
        material = trimesh.visual.material.PBRMaterial(
            # Splat layer tiles may carry a local micro-height channel packed into
            # alpha for shader blending (see terrain_texture_generation.py); strip it
            # here so the GLB preview material doesn't render as partially transparent.
            baseColorTexture=image.convert("RGB"),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        )
        visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        _ = tri_mesh.vertex_normals
        return tri_mesh

    @staticmethod
    def _fix_uv_seam(
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Duplicate vertices for triangles whose corners straddle the
        panorama's longitude seam (u≈0 vs u≈1, directly behind the camera —
        a full radial wedge from near the origin out to the far boundary,
        not a small localized patch), so linear UV interpolation across the
        triangle doesn't smear the texture from one edge of the image to
        the other.

        Any real terrain triangle is far too small in angular extent to
        legitimately span more than half the panorama's width in
        longitude, so a corner-to-corner U delta greater than 0.5 can only
        be seam wraparound — round(delta) gives the number of whole wraps
        to correct for. Only corners 1 and 2 of each triangle are ever
        shifted (relative to corner 0, an arbitrary but consistent local
        reference); shifted corners get a new, duplicated vertex (same
        XYZ, shifted U) so triangles that don't straddle the seam, and the
        un-shifted corner of ones that do, keep sharing their original
        vertices untouched.
        """
        u = uv[:, 0]
        u_corner = u[faces]  # (M, 3)
        shift1 = -np.round(u_corner[:, 1] - u_corner[:, 0])
        shift2 = -np.round(u_corner[:, 2] - u_corner[:, 0])

        face_idx1 = np.nonzero(shift1 != 0)[0]
        face_idx2 = np.nonzero(shift2 != 0)[0]
        if len(face_idx1) == 0 and len(face_idx2) == 0:
            return vertices, faces, uv

        faces = faces.copy()
        vertex_chunks = [vertices]
        uv_chunks = [uv]
        next_index = len(vertices)

        for face_idx, corner, shift in ((face_idx1, 1, shift1), (face_idx2, 2, shift2)):
            if len(face_idx) == 0:
                continue
            orig_vertex_idx = faces[face_idx, corner]
            dup_uv = uv[orig_vertex_idx].copy()
            dup_uv[:, 0] += shift[face_idx]
            vertex_chunks.append(vertices[orig_vertex_idx])
            uv_chunks.append(dup_uv)
            n = len(face_idx)
            faces[face_idx, corner] = np.arange(next_index, next_index + n)
            next_index += n

        return np.concatenate(vertex_chunks, axis=0), faces, np.concatenate(uv_chunks, axis=0)

    @staticmethod
    def _level_water(
        water_grid: np.ndarray,
        height_grid: np.ndarray,
        is_water: np.ndarray,
        Y_pos: np.ndarray,
        row_rm: np.ndarray,
        col_rm: np.ndarray,
        percentile: float,
    ) -> np.ndarray:
        """Flatten each connected water body to a single elevation.

        Water used to take `Y_pos` unchanged -- whatever the reconstructed terrain
        happened to be under each water-typed cell -- which is only correct if the
        terrain under a lake is already flat. It is not. Measured on the Shark Fin
        capture, one 11,872 m2 body of sea: its bulk sits near -1 m (p10 -1.56,
        median -0.85) with a tail climbing to +67.75 m, and 21.7% of its vertices
        stood more than a metre above the median. The ocean surface was draped up
        the cliffs, which no amount of shader work can make read as water.

        Levelled per CONNECTED BODY, not globally: two lakes at different altitudes
        are both real, and a single global level would sink one and float the other.

        The level is a LOW percentile rather than the median because the
        contamination is one-sided. Nothing pushes a water cell's terrain height
        down, but a cliff face misread as water -- spray, foam, wet rock -- pushes it
        up hard, and the Shark Fin tail is exactly that. A level slightly under the
        true surface merely lets the shoreline poke through, while one slightly over
        it floats the whole sheet; the errors are not symmetric and neither is the
        estimator. On that capture p25 gives -1.13 m against a median of -0.85 m, so
        on clean data the choice barely matters and on dirty data it matters a lot.

        Nothing ends up buried by this. The lakebed is carved from the surface, not
        independently of it -- see the water_depression_m line further down generate()
        -- so lowering a body's surface lowers its bed with it.

        The level is measured over the height GRID, not over the mesh vertices, and
        that distinction is load-bearing. Vertices are Poisson-disc sampled with a
        deliberate near-camera density bias (inner_min_dist vs outer_min_dist), so on
        Shark Fin 4,309 of 11,366 water vertices sit in the nearest 20 m and a
        vertex-weighted p25 returns -4.54 m -- an estimate of the cove at the
        viewer's feet, not of the sea. The same percentile over the grid, where every
        cell is the same area, returns -1.13 m against a shoreline whose terrain sits
        at -0.99 m: a 0.14 m step instead of a 3.5 m one.
        """
        from scipy.ndimage import label as _label

        labels, count = _label(water_grid)
        if count == 0:
            return Y_pos.copy()

        # Same nearest-neighbour sampling the mask itself used, so a vertex is
        # assigned the body it was already judged to be inside.
        vertex_component = map_coordinates(
            labels.astype(np.int32), [row_rm, col_rm], order=0, mode="nearest"
        )

        water_Y = Y_pos.copy()
        for component in range(1, count + 1):
            member = is_water & (vertex_component == component)
            if not member.any():
                continue
            if height_grid is not None and height_grid.shape == water_grid.shape:
                heights = height_grid[labels == component]
                heights = heights[np.isfinite(heights)]
            else:
                # Region and height grids are the same resolution on every capture
                # measured, but they are produced by different stages and nothing
                # enforces it. Fall back to the vertex-weighted estimate rather than
                # index one grid with the other's mask -- less accurate, still level.
                heights = Y_pos[member]
            if heights.size == 0:
                continue
            water_Y[member] = float(np.percentile(heights, percentile))
        return water_Y

    @staticmethod
    def _fix_uv_discontinuities(
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: np.ndarray,
        panorama_height: int,
        max_span_ratio: float = 4.0,
        max_passes: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Collapse the UV of triangles that bridge an OCCLUSION boundary.

        _fix_uv_seam above handles the other UV discontinuity this mesh has, the
        longitude wrap, and handles it by correcting the UV -- a wrapped corner has
        a right answer (shift by a whole turn) and shifting it recovers the texture
        exactly. This one has no right answer.

        A ground-level panorama sees a ridge and the terrain behind it at wildly
        different elevation angles, but the height grid puts them in ADJACENT cells,
        so the triangle spanning that step interpolates its UV straight across every
        panorama row between the two -- smearing a hundred-plus pixel band of image
        over two metres of ground. That is the "oil slick" ringing and the chevron
        patterning on the far slopes: not stretch (too FEW texels), but its opposite,
        a whole band of texture compressed onto one small triangle. There is no UV
        that makes it correct, because the panorama simply never photographed the
        ground that triangle covers -- it is behind the ridge.

        So the goal is only to make it inoffensive. Each flagged triangle keeps the
        UV of the two corners that agree with each other (its two corners on the same
        side of the step) and its outlier corner is pulled to the nearer of those,
        which renders it as a roughly correct flat patch instead of a rainbow.

        DETECTED GEOMETRICALLY, not by a pixel threshold. How much panorama a
        legitimate triangle spans depends entirely on how close it is -- a 2 m
        triangle at the viewer's feet really does cover tens of degrees, while the
        same triangle at 70 m covers under two -- so a fixed pixel budget would
        clobber the near field and miss the far field, which is precisely where the
        artefact lives. Compared instead against the span the triangle's own size and
        distance predict (edge / distance, in radians, converted to rows). Measured
        on the Rainier capture that ratio is 1.00 at p75, i.e. well-behaved triangles
        match their prediction almost exactly, with a tail out to 21; 4.0 sits above
        p90 (2.51) and flags 7.1% of faces covering 12% of the surface, median radius
        76 m -- all far field, as expected.

        Iterated to a fixed point. One pass moves the outlier corner onto one of the
        agreeing pair, which resolves a triangle with a clean 2-vs-1 split but leaves
        one whose three corners are all spread -- there the "agreeing pair" is itself
        far apart, and collapsing onto it still spans. A second pass then treats the
        (now 2-vs-1) result and finishes it. Measured on the Rainier capture: pass 1
        fixes 5,516 faces and leaves 108, pass 2 fixes those 108 and leaves 0, pass 3
        finds nothing. max_passes is a guard against a pathological mesh, not a
        tuning knob.

        Returns (vertices, faces, uv, faces_fixed) with faces_fixed summed over passes.
        """
        if len(faces) == 0 or panorama_height <= 0:
            return vertices, faces, uv, 0

        total_fixed = 0
        for _ in range(max(1, max_passes)):
            vertices, faces, uv, fixed = TerrainMeshGenerator._fix_uv_discontinuities_pass(
                vertices, faces, uv, panorama_height, max_span_ratio,
            )
            total_fixed += fixed
            if fixed == 0:
                break
        return vertices, faces, uv, total_fixed

    @staticmethod
    def _fix_uv_discontinuities_pass(
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: np.ndarray,
        panorama_height: int,
        max_span_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """One collapse pass. See _fix_uv_discontinuities, which iterates this."""
        tri_uv = uv[faces]                       # (M, 3, 2)
        tri_xyz = vertices[faces].astype(np.float64)

        # V only. U is longitude and has already been unwrapped by _fix_uv_seam, but
        # a residual whole-turn shift there would still read as a huge U delta here
        # and this must not re-litigate that; elevation has no wrap and is where an
        # occlusion step actually shows up.
        def uv_gap(a: int, b: int) -> np.ndarray:
            return np.abs(tri_uv[:, a, 1] - tri_uv[:, b, 1]) * panorama_height

        def edge_len(a: int, b: int) -> np.ndarray:
            return np.linalg.norm(tri_xyz[:, a] - tri_xyz[:, b], axis=1)

        gaps = np.stack([uv_gap(0, 1), uv_gap(1, 2), uv_gap(0, 2)], axis=1)
        edges = np.stack([edge_len(0, 1), edge_len(1, 2), edge_len(0, 2)], axis=1)
        distance = np.maximum(np.linalg.norm(tri_xyz.mean(axis=1), axis=1), 0.5)
        expected = (edges.max(axis=1) / distance) * (panorama_height / np.pi)
        flagged = np.nonzero(gaps.max(axis=1) > max_span_ratio * np.maximum(expected, 1e-6))[0]
        if len(flagged) == 0:
            return vertices, faces, uv, 0

        # The two corners joined by the SMALLEST V gap are the pair on one surface;
        # the remaining corner is the one that stepped across.
        pair_corners = ((0, 1), (1, 2), (0, 2))
        closest = gaps[flagged].argmin(axis=1)

        faces = faces.copy()
        vertex_chunks = [vertices]
        uv_chunks = [uv]
        next_index = len(vertices)
        for pair_idx, (ca, cb) in enumerate(pair_corners):
            rows = flagged[closest == pair_idx]
            if len(rows) == 0:
                continue
            outlier = 3 - ca - cb            # the corner in neither position
            # Pull it to whichever of the agreeing pair it is already nearer, so the
            # collapse moves the UV as little as it can.
            to_a = np.abs(tri_uv[rows, outlier, 1] - tri_uv[rows, ca, 1])
            to_b = np.abs(tri_uv[rows, outlier, 1] - tri_uv[rows, cb, 1])
            target = np.where(to_a <= to_b, ca, cb)
            dup_uv = tri_uv[rows, target, :].copy()
            vertex_chunks.append(vertices[faces[rows, outlier]])
            uv_chunks.append(dup_uv.astype(uv.dtype))
            faces[rows, outlier] = np.arange(next_index, next_index + len(rows))
            next_index += len(rows)

        return (
            np.concatenate(vertex_chunks, axis=0),
            faces,
            np.concatenate(uv_chunks, axis=0),
            int(len(flagged)),
        )

    @staticmethod
    def _uvs_pinhole(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        Z_safe = np.where(Z < 1e-3, 1e-3, Z).astype(np.float64)
        cx = X * intrinsics.fx / Z_safe + intrinsics.px
        cy = intrinsics.py - Y * intrinsics.fy / Z_safe
        u  = (cx / intrinsics.width).clip(0.0, 1.0)
        # Pre-flip V to cancel trimesh's export-time flip — see the comment in generate()
        # above the baked_tex UV formula for why this is needed.
        v  = (1.0 - (cy / intrinsics.height).clip(0.0, 1.0))
        return np.stack([u, v], axis=-1).astype(np.float32)

    # ── Noise helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _smooth_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
        """Layered Gaussian-smoothed noise (4 octaves) normalised to ±1."""
        rng   = np.random.default_rng(seed)
        noise = np.zeros(shape, dtype=np.float32)
        for octave in range(4):
            amplitude = 0.5 ** octave
            raw   = rng.standard_normal(shape).astype(np.float32) * amplitude
            sigma = max(1.0, min(shape) / (4.0 * (2 ** octave)))
            noise += gaussian_filter(raw, sigma=sigma)
        peak = np.abs(noise).max()
        if peak > 0:
            noise /= peak
        return noise
