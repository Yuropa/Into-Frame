using System;
using UnityEngine;

/// <summary>
/// Swaps the static, GLTFast-baked material on the server's flat water mesh for
/// IntoFrame/WaterSurface, which animates gentle sum-of-sines ripples in the vertex
/// shader (see WaterSurface.shader). Everything else about the mesh -- position,
/// scale, the lakebed depression it sits above -- is untouched; only its material
/// changes.
///
/// Attach to the same persistent GameObject as SceneClient/SceneObjectManager, or let
/// SceneObjectManager auto-add one (see its Start()).
/// </summary>
public class WaterMaterialManager : MonoBehaviour
{
    [Tooltip("Any loaded mesh whose name contains this string (case-insensitive) is treated as water.")]
    public string waterNamePattern = "water";

    [Tooltip("Shader used for animated water. Falls back to Shader.Find if unassigned.")]
    public Shader waterShader;

    private Material _runtimeWaterMaterial;

    /// <summary>
    /// Called by SceneObjectManager after every GLB mesh is instantiated. If the name
    /// matches waterNamePattern, every renderer under it gets the animated water
    /// material instead of whatever GLTFast baked from the server's static PBR data.
    /// </summary>
    public void RegisterMeshLoaded(GameObject go, string meshName)
    {
        if (string.IsNullOrEmpty(meshName)) return;
        if (meshName.IndexOf(waterNamePattern, StringComparison.OrdinalIgnoreCase) < 0) return;

        var mat = GetWaterMaterial();
        if (mat == null) return;

        // The server colours the sheet from the photographed water (TerrainMeshStage.
        // _colour_water_from_photo); carry that base colour over, keeping the
        // shader's own alpha. glTFast exposes glTF's baseColorFactor as "baseColorFactor"
        // (its own shaders) or "_BaseColor" (URP Lit).
        foreach (var r in go.GetComponentsInChildren<Renderer>())
        {
            var baked = r.sharedMaterial;
            if (baked == null) continue;
            string prop = baked.HasProperty("baseColorFactor") ? "baseColorFactor"
                        : baked.HasProperty("_BaseColor") ? "_BaseColor" : null;
            if (prop == null) continue;
            var c = baked.GetColor(prop);
            var current = mat.GetColor("_BaseColor");
            mat.SetColor("_BaseColor", new Color(c.r, c.g, c.b, current.a));
            break;
        }

        foreach (var r in go.GetComponentsInChildren<Renderer>())
            r.sharedMaterial = mat;
    }

    private Material GetWaterMaterial()
    {
        if (_runtimeWaterMaterial != null) return _runtimeWaterMaterial;

        var shader = waterShader != null ? waterShader : Shader.Find("IntoFrame/WaterSurface");
        if (shader == null)
        {
            Debug.LogWarning("[WaterMaterialManager] Shader 'IntoFrame/WaterSurface' not found. Assign waterShader in the Inspector.");
            return null;
        }

        _runtimeWaterMaterial = new Material(shader) { name = "WaterSurface (runtime)" };
        return _runtimeWaterMaterial;
    }
}
