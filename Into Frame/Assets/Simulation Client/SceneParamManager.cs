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

    [Header("Camera")]
    public GameObject camera;

    public void ApplyParams(SceneParams p)
    {
        if (p == null) return;
        Debug.Log($"[SceneParamManager] Received ambientColor: '{p.ambientColor}'");

        if (!string.IsNullOrEmpty(p.ambientColor) &&
            ColorUtility.TryParseHtmlString(p.ambientColor, out Color c))
        {
            _targetColor = c;
        }


        if (p.extrinsics != null)
        {
            float[] r = p.extrinsics.rotation;     // 9 floats, row-major
            float[] t = p.extrinsics.translation;  // 3 floats

            Vector3 position = new Vector3(t[0], t[1], t[2]);

            Matrix4x4 m = new Matrix4x4();
            m.SetRow(0, new Vector4(r[0], r[1], r[2], 0));
            m.SetRow(1, new Vector4(r[3], r[4], r[5], 0));
            m.SetRow(2, new Vector4(r[6], r[7], r[8], 0));
            m.SetRow(3, new Vector4(0,    0,    0,    1));

            Quaternion rotation = m.rotation;

            camera.transform.SetPositionAndRotation(position, rotation);
        }
    }

    private void Update()
    {
        if (directionalLight != null) {
            directionalLight.color = Color.Lerp(directionalLight.color, _targetColor, Time.deltaTime * 2f);
        }
    }
}
