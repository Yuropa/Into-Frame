using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Applies server-driven global scene parameters:
/// fog, ambient light, gravity, time scale.
///
/// Attach to the same persistent GameObject as SceneClient.
/// </summary>
public class SceneParamManager : MonoBehaviour
{
    [Header("Lights")]
    public Light directionalLight;

    [Header("Transition Speed")]
    public float colorLerpSpeed = 2f;
    public float floatLerpSpeed = 3f;

    // Target values (lerped towards in Update)
    private Color  _targetAmbient    = Color.gray;
    private Color  _targetFogColor   = Color.gray;
    private float  _targetFogDensity = 0.02f;
    private float  _targetTimeScale  = 1f;
    private float  _targetGravity    = -9.81f;

    // ── Public API ─────────────────────────────────────────────────────────

    public void ApplyParams(SceneParams p)
    {
        if (p == null) return;

        if (!string.IsNullOrEmpty(p.ambientColor) && ColorUtility.TryParseHtmlString(p.ambientColor, out Color ac))
            _targetAmbient = ac;

        if (!string.IsNullOrEmpty(p.fogColor) && ColorUtility.TryParseHtmlString(p.fogColor, out Color fc))
            _targetFogColor = fc;

        _targetFogDensity = p.fogDensity;
        _targetGravity    = p.gravity;
        _targetTimeScale  = p.timeScale;

        RenderSettings.fog = p.fogEnabled;
        RenderSettings.fogMode = FogMode.Exponential;
    }

    public void ApplyParam(string key, string rawValue)
    {
        switch (key)
        {
            case "ambientColor":
                if (ColorUtility.TryParseHtmlString(rawValue, out Color ac)) _targetAmbient = ac;
                break;
            case "fogColor":
                if (ColorUtility.TryParseHtmlString(rawValue, out Color fc)) _targetFogColor = fc;
                break;
            case "fogEnabled":
                RenderSettings.fog = rawValue.ToLower() == "true";
                break;
            case "fogDensity":
                if (float.TryParse(rawValue, out float fd)) _targetFogDensity = fd;
                break;
            case "gravity":
                if (float.TryParse(rawValue, out float g)) _targetGravity = g;
                break;
            case "timeScale":
                if (float.TryParse(rawValue, out float ts)) _targetTimeScale = ts;
                break;
        }
    }

    // ── Smooth Application ─────────────────────────────────────────────────

    private void Update()
    {
        float ct = Time.unscaledDeltaTime * colorLerpSpeed;
        float ft = Time.unscaledDeltaTime * floatLerpSpeed;

        // Ambient light
        RenderSettings.ambientLight = Color.Lerp(RenderSettings.ambientLight, _targetAmbient, ct);

        // Fog
        RenderSettings.fogColor   = Color.Lerp(RenderSettings.fogColor, _targetFogColor, ct);
        RenderSettings.fogDensity = Mathf.Lerp(RenderSettings.fogDensity, _targetFogDensity, ft);

        // Physics gravity
        Vector3 currentG = Physics.gravity;
        Physics.gravity = new Vector3(currentG.x, Mathf.Lerp(currentG.y, _targetGravity, ft), currentG.z);

        // Time scale (applied immediately — lerping time scale gets weird)
        Time.timeScale = _targetTimeScale;
    }
}
