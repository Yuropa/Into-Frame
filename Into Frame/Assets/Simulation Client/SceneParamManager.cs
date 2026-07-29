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
    public new GameObject camera;

    [Header("Scene")]
    [Tooltip("Parent of all server-spawned scene content (terrain, objects). Pushed down so the terrain center lands eyeHeightMeters below the floor (XR Origin stays fixed at y=0).")]
    public GameObject sceneRoot;

    [Header("Skybox")]
    public GameObject skybox;

    [Header("Lighting")]
    public EnvironmentLighting environmentLighting;

    public void ApplyParams(SceneParams p)
    {
        if (p == null) return;
        Debug.Log($"[SceneParamManager] Received ambientColor: '{p.ambientColor}'");

        if (!string.IsNullOrEmpty(p.ambientColor) &&
            ColorUtility.TryParseHtmlString(p.ambientColor, out Color c))
        {
            _targetColor = c;
        }

        // Key light from the estimated environment map (SceneLighting.sun on the
        // server). Without this the directional light kept whatever direction and
        // intensity the prefab shipped with, entirely unrelated to where the light
        // in the panorama actually comes from -- so every billboard and category
        // mesh, whose textures are photo crops with the real sun already baked
        // into their albedo, got lit a second time from the wrong angle and read
        // as a different colour and exposure from the terrain around them.
        ApplySun(p.lighting != null ? p.lighting.sun : null, p.skyboxRotation);


        Camera cam = camera != null ? camera.GetComponent<Camera>() : Camera.main;
        if (cam != null)
        {
            if (p.nearClipPlane > 0f) cam.nearClipPlane = p.nearClipPlane;
            if (p.farClipPlane  > 0f) cam.farClipPlane  = p.farClipPlane;
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

        // Push the whole scene down so the terrain center sits eyeHeightMeters below
        // the floor. The XR Origin always stays at the real-world floor (y=0); we
        // never move the player, only the content, so tracking/teleport/physics
        // keep their usual floor-relative meaning.
        if (sceneRoot != null)
        {
            var pos = sceneRoot.transform.position;
            pos.y = -p.eyeHeightMeters - p.terrainCenterY;
            sceneRoot.transform.position = pos;
        }

        if (p.skybox != null)
        {
            skybox.GetComponent<PanoramaSkybox>().LoadFromName(p.skybox, p.skyboxRotation);
        }

        if (environmentLighting != null)
            environmentLighting.Apply(p.lighting);
    }

    /// <summary>
    /// Point the directional light along the estimated sun direction and take its
    /// colour/intensity, then hand ambientColor to the ambient slot it actually
    /// describes.
    ///
    /// The server sends the direction in the panorama's own frame, so it needs the
    /// same skyboxRotation yaw the skybox itself gets (see PanoramaSkybox) --
    /// otherwise the light and the sky it was extracted from point different ways,
    /// which is more obviously wrong than no sun at all.
    /// </summary>
    private void ApplySun(SunData sun, float skyboxRotation)
    {
        if (directionalLight == null) return;

        if (sun == null || sun.direction == null || sun.direction.Length != 3)
        {
            // Overcast, or a degenerate environment map with no identifiable key
            // light. Leave the prefab's light alone rather than inventing a sun.
            Debug.Log("[SceneParamManager] No sun in lighting data — leaving directional light as authored");
            return;
        }

        Vector3 toSun = new Vector3(sun.direction[0], sun.direction[1], sun.direction[2]);
        if (toSun.sqrMagnitude < 1e-6f) return;
        toSun = Quaternion.Euler(0f, skyboxRotation, 0f) * toSun.normalized;

        // Same world-space vector the light uses, handed to the sky as its axis of
        // rotation: spinning about the sun keeps the sun (and therefore this light)
        // exactly where it is while the rest of the sky turns around it. Set here
        // rather than in PanoramaSkybox so the yaw correction is applied once, in the
        // one place that already reasons about it.
        if (skybox != null && skybox.TryGetComponent(out PanoramaSkybox panoramaSkybox))
            panoramaSkybox.SetSpinAxis(toSun);

        // A directional light shines along its own forward axis, so it has to face
        // the opposite way from "towards the sun".
        directionalLight.transform.rotation = Quaternion.LookRotation(-toSun);
        if (sun.intensity > 0f) directionalLight.intensity = sun.intensity;
        if (!string.IsNullOrEmpty(sun.color) && ColorUtility.TryParseHtmlString(sun.color, out Color sunColor))
            _targetColor = sunColor;

        Debug.Log($"[SceneParamManager] Sun: dir {toSun}, intensity {sun.intensity}, color {sun.color}"
                  + (sun.hdr ? "" : " (no HDR merge — intensity/colour approximate)"));
    }

    private void Update()
    {
        if (directionalLight != null) {
            directionalLight.color = Color.Lerp(directionalLight.color, _targetColor, Time.deltaTime * 2f);
        }
    }
}
