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
    public Light directionalLight;  // drag your directional light here

    private Color _targetColor = Color.white;

    public void ApplyParams(SceneParams p)
    {
        if (p == null) return;
        Debug.Log($"[SceneParamManager] Received ambientColor: '{p.ambientColor}'");

        if (!string.IsNullOrEmpty(p.ambientColor) &&
            ColorUtility.TryParseHtmlString(p.ambientColor, out Color c))
        {
            _targetColor = c;
        }
    }

    private void Update()
    {
        if (directionalLight != null) {
            directionalLight.color = Color.Lerp(directionalLight.color, _targetColor, Time.deltaTime * 2f);
        }
    }
}
