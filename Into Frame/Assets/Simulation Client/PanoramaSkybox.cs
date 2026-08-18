using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class PanoramaSkybox : MonoBehaviour
{
    [Header("Skybox Settings")]
    public string imageName;
    public float exposure = 1.0f;
    public float rotation = 0.0f;

    [Header("Sky Motion")]
    [Tooltip("Seconds for one full revolution of the sky. 0 disables the animation.")]
    // Off by default. A 90 s revolution is far more motion than a still photograph
    // implies: the panorama's clouds, its baked shadows and the terrain texture
    // underneath it are all fixed to the sky it came from, so wheeling the sky past
    // them reads as the world sliding rather than as weather. Rotating about the sun
    // axis keeps the LIGHTING consistent (see SetSpinAxis) but cannot keep the
    // imagery consistent, and on a scene with a strong horizon -- a cliff line, a
    // city skyline -- the mismatch is obvious.
    //
    // Kept as a knob rather than deleted: the machinery is sound and a slow drift
    // suits an overcast or featureless sky. Set a period in the Inspector to bring
    // it back per-scene.
    public float spinPeriodSeconds = 0f;

    private Material skyboxMaterial;
    private Texture2D currentTexture;

    // Axis the sky turns about, in world space. Set to the direction of the sun by
    // SetSpinAxis; rotating about it leaves the sun itself stationary while everything
    // else wheels around it, so the directional light extracted from this same panorama
    // stays consistent with what's drawn. Defaults to up (a plain yaw drift) until the
    // lighting payload arrives, and stays there if the scene has no identifiable sun.
    private Vector3 _spinAxis = Vector3.up;
    private bool _canSpin;      // false when the spin shader isn't available
    private float _spinDegrees; // accumulated, so changing the period never jumps

    [Header("Assets")]
    public GameObject server;

    void Start()
    {
        if (!string.IsNullOrEmpty(imageName))
            LoadFromName(imageName);
    }

    public void LoadFromName(string name, float rotationDegrees = 0f)
    {
        rotation = rotationDegrees;
        LoadFromName(name);
    }

    public void LoadFromName(string name)
    {
        if (gameObject.activeInHierarchy)
            StartCoroutine(LoadAndApplySkybox(name));
        else
            StartCoroutine(WaitThenLoad(name));
    }

    private IEnumerator WaitThenLoad(string name)
    {
        yield return new WaitUntil(() => gameObject.activeInHierarchy);
        StartCoroutine(LoadAndApplySkybox(name));
    }

    private AssetServer _assetServer = null;
    private AssetServer assetServer()
    {
        if (_assetServer != null) {
            return _assetServer;
        }

        _assetServer = server.GetComponent<AssetServer>();
        return _assetServer;
    }

    private IEnumerator LoadAndApplySkybox(string name)
    {
        using (UnityWebRequest request = assetServer().GetTexture(name))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[Skybox] Failed to load: {request.error}");
                yield break;
            }

            Texture2D texture = DownloadHandlerTexture.GetContent(request);
            ApplySkybox(texture);
        }
    }

    /// <summary>
    /// Point the sky's axis of rotation at the sun, in world space. Pass Vector3.zero
    /// (or nothing at all) for a scene with no identifiable sun and the sky keeps its
    /// default yaw drift about up.
    /// </summary>
    public void SetSpinAxis(Vector3 worldTowardSun)
    {
        if (worldTowardSun.sqrMagnitude < 1e-6f) return;
        _spinAxis = worldTowardSun.normalized;
        Debug.Log($"[Skybox] Spin axis set to sun direction {_spinAxis}");
    }

    private void ApplySkybox(Texture2D texture)
    {
        // Panorama textures need to wrap horizontally
        texture.wrapModeU = TextureWrapMode.Repeat;
        texture.wrapModeV = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Trilinear;
        texture.anisoLevel = 4;
        texture.Apply();

        // Prefer the arbitrary-axis variant; fall back to Unity's built-in when it
        // isn't in the build (e.g. stripped for not being referenced by any material in
        // a scene asset -- see the note in SkyboxSpin's shader). The fallback keeps the
        // static sky working rather than leaving a magenta dome, at the cost of the
        // animation. Both take _MainTex/_Exposure/_Rotation with the same meaning.
        Shader spinShader = Shader.Find("Skybox/PanoramaSpin");
        _canSpin = spinShader != null;
        if (!_canSpin)
        {
            Debug.LogWarning("[Skybox] 'Skybox/PanoramaSpin' not found — falling back to the " +
                             "static Skybox/Panoramic. Add it to Always Included Shaders in " +
                             "Graphics settings if the animation is wanted.");
            spinShader = Shader.Find("Skybox/Panoramic");
        }

        skyboxMaterial = new Material(spinShader);
        if (!_canSpin) skyboxMaterial.SetFloat("_Mapping", 0);
        currentTexture = texture;
        skyboxMaterial.SetTexture("_MainTex", texture);
        skyboxMaterial.SetFloat("_Exposure", exposure);
        skyboxMaterial.SetFloat("_Rotation", rotation);
        if (_canSpin) skyboxMaterial.SetMatrix("_SkyRotation", Matrix4x4.identity);

        // Apply to the scene
        RenderSettings.skybox = skyboxMaterial;

        // Force skybox to update
        DynamicGI.UpdateEnvironment();

        Debug.Log($"[Skybox] Panorama applied ({(_canSpin ? "animated" : "static")})");
    }

    private void Update()
    {
        if (!_canSpin || skyboxMaterial == null || spinPeriodSeconds <= 0f) return;

        // Accumulated rather than derived from Time.time, so editing spinPeriodSeconds
        // at runtime changes the RATE without teleporting the sky to wherever the new
        // period says it should already be.
        _spinDegrees = Mathf.Repeat(_spinDegrees + 360f * Time.deltaTime / spinPeriodSeconds, 360f);
        skyboxMaterial.SetMatrix("_SkyRotation",
            Matrix4x4.Rotate(Quaternion.AngleAxis(_spinDegrees, _spinAxis)));

        // Deliberately NOT calling DynamicGI.UpdateEnvironment() here. It re-bakes the
        // ambient probe from the skybox, which is far too expensive per frame, and the
        // result barely changes anyway: rotating about the sun axis leaves the sun --
        // the overwhelming majority of the irradiance -- exactly where it was.
    }

    void OnDestroy()
    {
        if (skyboxMaterial != null) Destroy(skyboxMaterial);
        if (currentTexture != null) Destroy(currentTexture);
    }
}
