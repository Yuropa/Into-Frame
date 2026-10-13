using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Builds the scene's ambient lighting (ambient SH + reflection probe) from the
/// skybox the viewer actually sees.
///
/// One sky. The panorama skybox (PanoramaSkybox), the solar disc the server measured
/// the key light from, and the environment the probe is rendered from are all the
/// same image. Before this the probe was rendered from LuxDiT's own LDR estimate --
/// a different, often dusk-looking map whose sun sat on the opposite side from the
/// directional light -- so the ground bounce, the sky overhead and the key light
/// never agreed with each other. That estimate is still delivered in
/// SceneLightingData (the server derives the sun intensity from its HDR merge) and
/// can be switched back on with useEstimatedEnvironmentMap for comparison.
///
/// Attach to the same persistent GameObject as SceneClient.
/// In the Inspector, set the ReflectionProbe to Realtime / Via Scripting.
/// </summary>
public class EnvironmentLighting : MonoBehaviour
{
    [Header("Reflection")]
    public ReflectionProbe reflectionProbe;

    [Header("Source")]
    [Tooltip("Render the probe from LuxDiT's estimated environment map instead of the " +
             "visible panorama skybox. Off: the skybox the viewer sees is also what " +
             "lights the scene.")]
    public bool useEstimatedEnvironmentMap = false;

    [Tooltip("Seconds to wait for the panorama skybox to load before giving up and " +
             "rendering the probe from whatever skybox is set.")]
    public float skyboxLoadTimeoutSeconds = 30f;

    private Material  _envMaterial;
    private Texture2D _envTexture;

    public void Apply(SceneLightingData data, PanoramaSkybox panorama = null)
    {
        if (!gameObject.activeInHierarchy)
        {
            Debug.LogWarning("[EnvironmentLighting] GameObject inactive — skipping ambient probe update");
            return;
        }
        if (useEstimatedEnvironmentMap)
        {
            if (data == null || string.IsNullOrEmpty(data.ldr))
            {
                Debug.LogWarning("[EnvironmentLighting] No estimated environment map from server — falling back to the skybox");
                StartCoroutine(ApplyFromSkybox(panorama));
                return;
            }
            StartCoroutine(ApplyFromEstimate(data));
            return;
        }
        StartCoroutine(ApplyFromSkybox(panorama));
    }

    private IEnumerator ApplyFromSkybox(PanoramaSkybox panorama)
    {
        if (panorama != null)
        {
            float deadline = Time.realtimeSinceStartup + skyboxLoadTimeoutSeconds;
            // PanoramaSkybox bumps LoadGeneration every time a new panorama is applied,
            // so wait for the one that was just requested, not a previous scene's.
            int wanted = panorama.PendingGeneration;
            while (panorama.AppliedGeneration < wanted && Time.realtimeSinceStartup < deadline)
                yield return null;
            if (panorama.AppliedGeneration < wanted)
                Debug.LogWarning("[EnvironmentLighting] Skybox did not load in time — rendering the probe from the current skybox");
        }

        // PanoramaSkybox.ApplySkybox already called DynamicGI.UpdateEnvironment() for
        // the ambient SH; the reflection probe still has to be rendered explicitly.
        DynamicGI.UpdateEnvironment();
        if (reflectionProbe != null)
        {
            int id = reflectionProbe.RenderProbe();
            yield return new WaitUntil(() => reflectionProbe.IsFinishedRendering(id));
        }
        Debug.Log("[EnvironmentLighting] Ambient probe rendered from the panorama skybox");
    }

    private IEnumerator ApplyFromEstimate(SceneLightingData data)
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

        // Swap the skybox to the estimated env map so the probe captures it,
        // then restore afterward. Do NOT call DynamicGI.UpdateEnvironment() on
        // restore — that would undo the ambient update we just applied.
        Material savedSkybox = RenderSettings.skybox;
        RenderSettings.skybox = _envMaterial;
        DynamicGI.UpdateEnvironment();

        if (reflectionProbe != null)
        {
            int id = reflectionProbe.RenderProbe();
            yield return new WaitUntil(() => reflectionProbe.IsFinishedRendering(id));
        }

        RenderSettings.skybox = savedSkybox;

        Debug.Log("[EnvironmentLighting] Reflection probe updated from estimated environment map");
    }

    private void OnDestroy()
    {
        if (_envMaterial != null) Destroy(_envMaterial);
        if (_envTexture  != null) Destroy(_envTexture);
    }
}
