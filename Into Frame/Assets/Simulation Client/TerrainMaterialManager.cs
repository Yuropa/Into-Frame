using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Receives SplatMaterial data from the server, decodes base64 tiles and blend maps
/// into Texture2Ds, and applies them to the terrain mesh once it is loaded.
///
/// Attach to the same persistent GameObject as SceneClient.
/// SceneObjectManager calls RegisterMeshLoaded() after every GLB load — this manager
/// checks whether the name matches the terrain and applies textures if so.
///
/// Shader wiring (IntoFrame/TerrainSplat):
///   _BlendMap0/_BlendMap1   RGBA — R=layer0  G=layer1  B=layer2  A=layer3 (per map)
///   _LayerNTile             — per-region tileable texture
///   _LayerNTileRepeat       — UV repeat count for each layer
///   _EquirectLayers         — int bitmask: bit i set = layer i uses equirect world-UV
/// </summary>
public class TerrainMaterialManager : MonoBehaviour
{
    [Header("Terrain identification")]
    [Tooltip("Any loaded mesh whose name contains this string (case-insensitive) is treated as terrain.")]
    public string terrainNamePattern = "terrain";

    [Header("Shader (assign once the splat shader exists)")]
    public Material splatMaterial;

    // ── Decoded data ───────────────────────────────────────────────────────

    public string[]    LayerNames      { get; private set; } = Array.Empty<string>();
    public Texture2D[] LayerTiles      { get; private set; } = Array.Empty<Texture2D>();
    public float[]     LayerFactors    { get; private set; } = Array.Empty<float>();
    public float[]     LayerSmoothness { get; private set; } = Array.Empty<float>();
    public Texture2D[] BlendMaps       { get; private set; } = Array.Empty<Texture2D>();
    public int         EquirectMask    { get; private set; }
    public bool        IsReady         { get; private set; }

    // ── Shader property IDs ────────────────────────────────────────────────

    static readonly int[] _blendMapIds =
    {
        Shader.PropertyToID("_BlendMap0"),
        Shader.PropertyToID("_BlendMap1"),
    };

    static readonly int[] _layerTileIds =
    {
        Shader.PropertyToID("_Layer0Tile"),
        Shader.PropertyToID("_Layer1Tile"),
        Shader.PropertyToID("_Layer2Tile"),
        Shader.PropertyToID("_Layer3Tile"),
        Shader.PropertyToID("_Layer4Tile"),
        Shader.PropertyToID("_Layer5Tile"),
        Shader.PropertyToID("_Layer6Tile"),
        Shader.PropertyToID("_Layer7Tile"),
    };

    static readonly int[] _layerTileRepeatIds =
    {
        Shader.PropertyToID("_Layer0TileRepeat"),
        Shader.PropertyToID("_Layer1TileRepeat"),
        Shader.PropertyToID("_Layer2TileRepeat"),
        Shader.PropertyToID("_Layer3TileRepeat"),
        Shader.PropertyToID("_Layer4TileRepeat"),
        Shader.PropertyToID("_Layer5TileRepeat"),
        Shader.PropertyToID("_Layer6TileRepeat"),
        Shader.PropertyToID("_Layer7TileRepeat"),
    };

    static readonly int[] _layerSmoothnessIds =
    {
        Shader.PropertyToID("_Layer0Smoothness"),
        Shader.PropertyToID("_Layer1Smoothness"),
        Shader.PropertyToID("_Layer2Smoothness"),
        Shader.PropertyToID("_Layer3Smoothness"),
        Shader.PropertyToID("_Layer4Smoothness"),
        Shader.PropertyToID("_Layer5Smoothness"),
        Shader.PropertyToID("_Layer6Smoothness"),
        Shader.PropertyToID("_Layer7Smoothness"),
    };

    static readonly int _equirectLayersId = Shader.PropertyToID("_EquirectLayers");

    // Deferred: terrain may arrive before or after the splat data
    private readonly List<Renderer> _pendingRenderers = new();
    private Material _runtimeSplatMaterial;

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Called by SceneClient when SCENE_INIT contains terrain_material.</summary>
    public void Apply(SplatMaterialData data)
    {
        if (data == null || data.layers == null)
        {
            Debug.LogWarning("[TerrainMaterialManager] Received null or empty terrain_material — skipping.");
            return;
        }

        IsReady = false;

        LayerNames      = new string[data.layers.Length];
        LayerTiles      = new Texture2D[data.layers.Length];
        LayerFactors    = new float[data.layers.Length];
        LayerSmoothness = new float[data.layers.Length];
        EquirectMask    = 0;

        for (int i = 0; i < data.layers.Length; i++)
        {
            LayerNames[i]      = data.layers[i].name;
            LayerTiles[i]      = DecodeTexture(data.layers[i].tile, $"tile_{data.layers[i].name}", linear: true);
            LayerFactors[i]    = data.layers[i].tile_factor > 0f ? data.layers[i].tile_factor : 1.0f;
            LayerSmoothness[i] = data.layers[i].smoothness;
            if (data.layers[i].equirect)
                EquirectMask |= (1 << i);
        }

        int blendCount = data.blend_maps?.Length ?? 0;
        BlendMaps = new Texture2D[blendCount];
        for (int i = 0; i < blendCount; i++)
            BlendMaps[i] = DecodeTexture(data.blend_maps[i], $"blend_map_{i}", linear: true);

        IsReady = true;

        Debug.Log($"[TerrainMaterialManager] Ready — {LayerTiles.Length} layer(s), " +
                  $"{BlendMaps.Length} blend map(s), equirect mask 0x{EquirectMask:X}. " +
                  $"Layers: {string.Join(", ", LayerNames)}");

        foreach (var r in _pendingRenderers)
            if (r != null) ApplyToRenderer(r);
        _pendingRenderers.Clear();
    }

    /// <summary>
    /// Called by SceneObjectManager after every GLB mesh is instantiated.
    /// If the name matches terrainNamePattern the splat textures are applied.
    /// </summary>
    public void RegisterMeshLoaded(GameObject go, string meshName)
    {
        if (string.IsNullOrEmpty(meshName)) return;
        if (meshName.IndexOf(terrainNamePattern, StringComparison.OrdinalIgnoreCase) < 0) return;

        foreach (var r in go.GetComponentsInChildren<Renderer>())
        {
            if (IsReady)
                ApplyToRenderer(r);
            else
                _pendingRenderers.Add(r);
        }
    }

    // ── Apply to renderer ──────────────────────────────────────────────────

    private Material GetSplatMaterial()
    {
        if (splatMaterial != null) return splatMaterial;
        if (_runtimeSplatMaterial != null) return _runtimeSplatMaterial;
        var shader = Shader.Find("IntoFrame/TerrainSplat");
        if (shader == null)
        {
            Debug.LogWarning("[TerrainMaterialManager] Shader 'IntoFrame/TerrainSplat' not found. Assign splatMaterial in the Inspector.");
            return null;
        }
        _runtimeSplatMaterial = new Material(shader) { name = "TerrainSplat (runtime)" };
        return _runtimeSplatMaterial;
    }

    private void ApplyToRenderer(Renderer r)
    {
        var mat = GetSplatMaterial();
        if (mat != null)
            r.sharedMaterial = mat;

        var mpb = new MaterialPropertyBlock();
        r.GetPropertyBlock(mpb);

        for (int i = 0; i < BlendMaps.Length && i < _blendMapIds.Length; i++)
            if (BlendMaps[i] != null)
                mpb.SetTexture(_blendMapIds[i], BlendMaps[i]);

        for (int i = 0; i < LayerTiles.Length && i < _layerTileIds.Length; i++)
        {
            if (LayerTiles[i] != null)
                mpb.SetTexture(_layerTileIds[i], LayerTiles[i]);
            if (i < LayerFactors.Length)
                mpb.SetFloat(_layerTileRepeatIds[i], LayerFactors[i]);
            if (i < LayerSmoothness.Length)
                mpb.SetFloat(_layerSmoothnessIds[i], LayerSmoothness[i]);
        }

        mpb.SetInt(_equirectLayersId, EquirectMask);

        r.SetPropertyBlock(mpb);

        Debug.Log($"[TerrainMaterialManager] Applied to '{r.gameObject.name}' — " +
                  $"{LayerTiles.Length} tile(s), {BlendMaps.Length} blend map(s), " +
                  $"equirect mask 0x{EquirectMask:X}.");
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static Texture2D DecodeTexture(string base64, string label, bool linear)
    {
        byte[] bytes;
        try { bytes = Convert.FromBase64String(base64); }
        catch (Exception e)
        {
            Debug.LogError($"[TerrainMaterialManager] Base64 decode failed for '{label}': {e.Message}");
            return Texture2D.whiteTexture;
        }

        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, mipChain: false, linear: linear);
        if (!tex.LoadImage(bytes))
        {
            Debug.LogError($"[TerrainMaterialManager] PNG decode failed for '{label}'.");
            return Texture2D.whiteTexture;
        }

        tex.name        = label;
        tex.wrapMode    = TextureWrapMode.Repeat;
        tex.filterMode  = FilterMode.Bilinear;
        tex.Apply(updateMipmaps: true, makeNoLongerReadable: false);
        return tex;
    }
}
