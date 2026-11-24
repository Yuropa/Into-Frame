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

    [Tooltip("How much of the photographed water (the texture the water GLB carries) shows in place of the shader's computed water colour. 0 turns it off.")]
    [Range(0f, 1f)] public float photoWeight = 1f;

    private Material _runtimeWaterMaterial;
    // Keeps the photographed-water textures referenced once the baked materials are replaced.
    private readonly System.Collections.Generic.List<Texture> _photoTextures = new System.Collections.Generic.List<Texture>();

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

        // The photographed water (server: TerrainMeshStage._photograph_water): the GLB's
        // material carries it as its base colour texture, sampled through the mesh's
        // second UV set (glTF TEXCOORD_1), alpha = weight. Bound per renderer, so a
        // water mesh without that UV set or texture keeps the shader's own look
        // (_PhotoWeight 0) while sharing the one runtime material.
        foreach (var r in go.GetComponentsInChildren<Renderer>())
        {
            Texture photo = PhotographedWater(r);
            r.sharedMaterial = mat;
            var block = new MaterialPropertyBlock();
            r.GetPropertyBlock(block);
            if (photo != null)
            {
                // The seam triangles' U runs past 1 (TerrainMeshGenerator._fix_uv_seam).
                photo.wrapModeU = TextureWrapMode.Repeat;
                photo.wrapModeV = TextureWrapMode.Clamp;
                block.SetTexture("_PhotoTex", photo);
                _photoTextures.Add(photo);
            }
            block.SetFloat("_PhotoWeight", photo != null ? photoWeight : 0f);
            r.SetPropertyBlock(block);
        }
    }

    /// <summary>
    /// The base colour texture glTFast baked on this renderer's material, when its
    /// mesh has the second UV set that texture is sampled through; null otherwise.
    /// </summary>
    private static Texture PhotographedWater(Renderer r)
    {
        var baked = r.sharedMaterial;
        var filter = r.GetComponent<MeshFilter>();
        var mesh = filter != null ? filter.sharedMesh : null;
        if (baked == null || mesh == null) return null;
        if (!mesh.HasVertexAttribute(UnityEngine.Rendering.VertexAttribute.TexCoord1)) return null;
        foreach (var prop in new[] { "baseColorTexture", "_BaseMap", "_MainTex" })
        {
            if (!baked.HasProperty(prop)) continue;
            var tex = baked.GetTexture(prop);
            if (tex != null) return tex;
        }
        return null;
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
