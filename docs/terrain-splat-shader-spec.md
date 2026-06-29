# Terrain Splat Shader — Implementation Spec

## Context

This is a Unity (URP) project that receives generated 3D scenes over WebSocket from a Python pipeline server. The terrain is a procedural mesh delivered as a GLB file. Terrain texturing uses a **splat map** system: up to 8 tileable region textures (grass, rock, water, etc.) are blended per-pixel using RGBA blend maps, similar to Unity's built-in terrain system.

All the C# plumbing is already in place. The remaining work is:
1. Write the URP splat shader
2. Create a Material using it
3. Assign the Material in the Inspector

---

## What already exists

### Data flow

```
Python server
  └─ SCENE_INIT WebSocket message
       ├─ scene: { objects, lighting, ... }
       └─ terrain_material:
            ├─ tile_size: int          (px per tile, e.g. 1024)
            ├─ layers: [{ name, tile }]  (up to 8; base64 PNG each)
            └─ blend_maps: [<base64 PNG>, ...]  (up to 2 RGBA images)
```

### C# scripts (all in `Assets/Simulation Client/`)

**`SceneClient.cs`**
- Routes the `SCENE_INIT` message
- Deserialises `terrain_material` into `SplatMaterialData`
- Calls `terrainMaterialManager.Apply(init.terrain_material)`

**`TerrainMaterialManager.cs`** (new)
- `Apply(SplatMaterialData data)` — decodes every base64 PNG into a `Texture2D` (wrap mode `Repeat`, linear colour space)
- `RegisterMeshLoaded(GameObject go, string name)` — called by `SceneObjectManager` after each GLB loads; if the name contains "terrain" (case-insensitive) it applies all textures to the mesh's renderers via `MaterialPropertyBlock`
- If `splatMaterial` is assigned, swaps the renderer's material to it before setting properties
- Exposes `LayerTiles[]`, `BlendMaps[]`, `LayerNames[]`, `TileSize` as public properties

**`SceneObjectManager.cs`**
- After each GLB mesh loads, calls `terrainMaterialManager.RegisterMeshLoaded(container, container.name)`

**Data models in `SceneClient.cs`**
```csharp
[Serializable] public class SplatLayerData   { public string name; public string tile; }
[Serializable] public class SplatMaterialData { public int tile_size; public SplatLayerData[] layers; public string[] blend_maps; }
[Serializable] public class SceneInitPayload  { public SceneParams scene; public SplatMaterialData terrain_material; }
```

---

## Texture formats

### Blend maps (`blend_maps[0]`, `blend_maps[1]`)
- Format: RGBA PNG, dimensions = `blend_map_size` (default 1024×1024)
- Channel → layer mapping is **positional**:
  - `blend_maps[0]` R → layer 0, G → layer 1, B → layer 2, A → layer 3
  - `blend_maps[1]` R → layer 4, G → layer 5, B → layer 6, A → layer 7
- Values are linear weights in [0, 255]; they are pre-normalised to sum to 1.0 across all active layers at each pixel

### Layer tiles (`layers[i].tile`)
- Format: RGB PNG, dimensions = `tile_size` × `tile_size` (e.g. 1024×1024)
- Seamlessly tileable (circular-shift seam fix applied during generation)
- Wrap mode must be `Repeat`

### Mesh UVs
- The terrain mesh has a single UV channel (TEXCOORD_0)
- UVs run **0 → 1** across the full terrain extent (not pre-tiled)
- For blend map sampling: use UV as-is
- For tile sampling: use `UV * tileRepeat` where `tileRepeat` is a shader parameter controlling how many times the tile repeats across the terrain

---

## Shader to write

**Path:** `Assets/Simulation Client/Shaders/TerrainSplat.shader`  
**Pipeline:** URP (use `Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl`)  
**Shader name:** `"IntoFrame/TerrainSplat"`

### Properties

```hlsl
// Blend maps (RGBA weight maps, UV 0→1)
_BlendMap0   ("Blend Map 0",  2D) = "black" {}
_BlendMap1   ("Blend Map 1",  2D) = "black" {}

// Per-layer tileable textures (up to 8)
_Layer0Tile  ("Layer 0 Tile", 2D) = "white" {}
_Layer1Tile  ("Layer 1 Tile", 2D) = "white" {}
_Layer2Tile  ("Layer 2 Tile", 2D) = "white" {}
_Layer3Tile  ("Layer 3 Tile", 2D) = "white" {}
_Layer4Tile  ("Layer 4 Tile", 2D) = "white" {}
_Layer5Tile  ("Layer 5 Tile", 2D) = "white" {}
_Layer6Tile  ("Layer 6 Tile", 2D) = "white" {}
_Layer7Tile  ("Layer 7 Tile", 2D) = "white" {}

// How many times each tile repeats across the terrain (set to match
// the tile density used during generation, e.g. 8.0 for an 8× repeat)
_TileRepeat  ("Tile Repeat",  Float) = 8.0
```

> `TerrainMaterialManager` sets `_BlendMap0`, `_BlendMap1`, `_Layer0Tile`–`_Layer7Tile`, and `_TileSize` (raw px value) via `MaterialPropertyBlock`. The shader uses `_TileRepeat` for the repeat count, which you can expose separately or derive from `_TileSize` and the terrain grid size. The simplest approach is just to expose `_TileRepeat` as a material parameter and set it in the Inspector.

### Fragment shader logic

```hlsl
// 1. Sample blend maps at terrain UV (0→1)
float4 blend0 = SAMPLE_TEXTURE2D(_BlendMap0, sampler_BlendMap0, i.uv);
float4 blend1 = SAMPLE_TEXTURE2D(_BlendMap1, sampler_BlendMap1, i.uv);

// 2. Sample each layer tile at tiled UV
float2 tiledUV = i.uv * _TileRepeat;
float3 col0 = SAMPLE_TEXTURE2D(_Layer0Tile, sampler_Layer0Tile, tiledUV).rgb;
float3 col1 = SAMPLE_TEXTURE2D(_Layer1Tile, sampler_Layer1Tile, tiledUV).rgb;
// ... col2 through col7

// 3. Weighted blend
float3 color =
    col0 * blend0.r +
    col1 * blend0.g +
    col2 * blend0.b +
    col3 * blend0.a +
    col4 * blend1.r +
    col5 * blend1.g +
    col6 * blend1.b +
    col7 * blend1.a;
```

The weights are pre-normalised by the server, so no additional normalisation is needed in the shader.

### Lighting

Use standard URP PBR lighting (`UniversalFragmentPBR`) with:
- `albedo = color` (from the blend above)
- `metallic = 0`
- `smoothness = 0.1` (terrain is matte)
- Normal from mesh vertex normals (no normal maps needed initially)

---

## Material to create

**Path:** `Assets/Simulation Client/Materials/TerrainSplat.mat`

- Shader: `IntoFrame/TerrainSplat`
- Leave all texture slots empty — `TerrainMaterialManager` fills them at runtime via `MaterialPropertyBlock`
- Set `_TileRepeat = 8` as the default

---

## Inspector wiring

On the same GameObject that holds `SceneClient` and `TerrainMaterialManager`:

1. Add `TerrainMaterialManager` component (if not already in scene)
2. Assign `TerrainSplat.mat` to the `Splat Material` field on `TerrainMaterialManager`
3. `SceneClient` already auto-finds `TerrainMaterialManager` via `FindObjectOfType` in `Start()` — no manual wiring needed there
4. `SceneObjectManager` needs `terrainMaterialManager` assigned (drag the same component in)

---

## What the shader does NOT need to handle

- Alpha/transparency (terrain is fully opaque)
- Shadows other than receiving them (standard URP shadow receiver is fine)
- Texture tiling offsets per layer (all layers use the same `_TileRepeat`)
- Normal maps per layer (out of scope for now)
- More than 8 layers (server caps at 8, 2 blend maps × 4 channels)

---

## Files changed by the previous implementation (for reference)

| File | What changed |
|------|-------------|
| `server/scene/splat_material.py` | New — `SplatMaterial` / `SplatLayer` types with `from_region_map`, `from_weight_maps`, `from_single_layer`, `encode`, `decode`, `save`, `load` |
| `server/pipeline/terrain/terrain_texture_generation.py` | New stage — generates tileable tiles per region via FLUX, builds `SplatMaterial`, writes debug images |
| `server/pipeline/terrain/terrain.py` | Reads `TERRAIN_MATERIAL` from context; sets mesh UVs 0→1; embeds first tile as GLB preview |
| `server/server/server.py` | `get_snapshot()` includes `terrain_material: splat.encode()` |
| `server/pipeline/pipeline_context.py` | Added `TERRAIN_MATERIAL` key |
| `server/config.yaml` | Replaced `TerrainTextureBakeStage` + `TerrainTextureRefinementStage` with `TerrainTextureGenerationStage` |
| `Assets/Simulation Client/SceneClient.cs` | Added `SplatLayerData`, `SplatMaterialData`, updated `SceneInitPayload`, wired `terrainMaterialManager` |
| `Assets/Simulation Client/SceneObjectManager.cs` | Calls `terrainMaterialManager.RegisterMeshLoaded()` after GLB load |
| `Assets/Simulation Client/TerrainMaterialManager.cs` | New — decodes textures, applies to terrain renderer |
