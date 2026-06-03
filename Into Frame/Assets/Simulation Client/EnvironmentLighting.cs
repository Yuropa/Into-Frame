using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Decodes a base64 equirectangular LDR environment map (from SceneLightingData)
/// and re-renders a ReflectionProbe from it.
///
/// Attach to the same persistent GameObject as SceneClient.
/// In the Inspector, set the ReflectionProbe to Realtime / Via Scripting.
/// </summary>
public class EnvironmentLighting : MonoBehaviour
{
    [Header("Reflection")]
    public ReflectionProbe reflectionProbe;

    private Material  _envMaterial;
    private Texture2D _envTexture;

    public void Apply(SceneLightingData data)
    {
        if (data == null || string.IsNullOrEmpty(data.ldr)) return;
        if (gameObject.activeInHierarchy)
            StartCoroutine(ApplyCoroutine(data));
    }

    private IEnumerator ApplyCoroutine(SceneLightingData data)
    {
        byte[]    bytes = Convert.FromBase64String(data.ldr);
        Texture2D tex   = new Texture2D(2, 2, TextureFormat.RGBA32, false);

        if (!ImageConversion.LoadImage(tex, bytes))
        {
            Debug.LogError("[EnvironmentLighting] Failed to decode LDR texture");
            Destroy(tex);
            yield break;
        }

        tex.wrapModeU  = TextureWrapMode.Repeat;
        tex.wrapModeV  = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Trilinear;
        tex.Apply();

        if (_envMaterial != null) Destroy(_envMaterial);
        if (_envTexture  != null) Destroy(_envTexture);

        _envTexture  = tex;
        _envMaterial = new Material(Shader.Find("Skybox/Panoramic"));
        _envMaterial.SetTexture("_MainTex", tex);
        _envMaterial.SetFloat("_Mapping",   0);   // 0 = 360 degrees equirect
        _envMaterial.SetFloat("_Exposure",  1f);

        // Temporarily swap skybox so RenderProbe captures the estimated env map
        Material savedSkybox = RenderSettings.skybox;
        RenderSettings.skybox = _envMaterial;
        DynamicGI.UpdateEnvironment();

        if (reflectionProbe != null)
        {
            int id = reflectionProbe.RenderProbe();
            yield return new WaitUntil(() => reflectionProbe.IsFinishedRendering(id));
        }

        RenderSettings.skybox = savedSkybox;
        DynamicGI.UpdateEnvironment();

        Debug.Log("[EnvironmentLighting] Reflection probe updated from estimated environment map");
    }

    private void OnDestroy()
    {
        if (_envMaterial != null) Destroy(_envMaterial);
        if (_envTexture  != null) Destroy(_envTexture);
    }
}
