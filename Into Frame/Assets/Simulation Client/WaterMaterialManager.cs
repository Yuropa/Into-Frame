using System;
using UnityEngine;

/// <summary>
/// Swaps the static, GLTFast-baked material on the server's flat water mesh (and
/// the far field's water apron, also named "water") for IntoFrame/WaterSurface:
/// vertex swell, per-pixel ripple, depth-based absorption and shoreline fade from
/// the camera depth texture, refraction from the opaque texture, the scene probe's
/// reflection by Fresnel, and the sun's glitter (see WaterSurface.shader). Everything else about the mesh -- position,
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
            // The shallow tint: the same water a metre deep, lighter and a little
            // greener than the photographed deep colour. Derived rather than
            // authored, so a sunset sea and a city river each get their own.
            Color.RGBToHSV(c, out float h, out float sat, out float val);
            var shallow = Color.HSVToRGB(h + 0.02f, sat * 0.6f, Mathf.Clamp01(val * 1.6f + 0.1f));
            mat.SetColor("_ShallowColor", new Color(shallow.r, shallow.g, shallow.b, 1f));
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
