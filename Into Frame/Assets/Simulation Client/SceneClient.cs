using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket; // https://github.com/endel/NativeWebSocket

/// <summary>
/// Attach to a persistent GameObject (e.g. "SceneManager").
/// Connects to the server, receives messages, and routes them
/// to the SceneObjectManager.
/// </summary>
public class SceneClient : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "ws://localhost:8080";
    public float reconnectDelay = 3f;

    [Header("References")]
    public SceneObjectManager objectManager;
    public SceneParamManager paramManager;
    public ProgressController progressController;
    public TerrainMaterialManager terrainMaterialManager;

    private WebSocket _ws;
    private bool _reconnecting = false;
    private string _lastSceneId = null;

    // ── Unity Lifecycle ────────────────────────────────────────────────────

    private void Start()
    {
        if (objectManager         == null) objectManager         = FindObjectOfType<SceneObjectManager>();
        if (paramManager          == null) paramManager          = FindObjectOfType<SceneParamManager>();
        if (progressController    == null) progressController    = FindObjectOfType<ProgressController>();
        if (terrainMaterialManager == null) terrainMaterialManager = FindObjectOfType<TerrainMaterialManager>();
        ConnectAsync();
    }

    private void Update()
    {
        // NativeWebSocket requires this on all platforms except WebGL
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
    }

    private void OnDestroy() => _ws?.Close();

    // ── Connection ─────────────────────────────────────────────────────────

    private async void ConnectAsync()
    {
        Debug.Log($"[SceneClient] Connecting to {serverUrl}…");
        _ws = new WebSocket(serverUrl);

        _ws.OnOpen += OnOpen;
        _ws.OnMessage += OnMessage;
        _ws.OnError += OnError;
        _ws.OnClose += OnClose;

        await _ws.Connect();
    }

    [Serializable]
    private class TypeOnlyMessage
    {
        public string type;
        public TypeOnlyMessage(string type) { this.type = type; }
    }

    private void OnOpen()
    {
        Debug.Log("[SceneClient] Connected!");
        Send(new TypeOnlyMessage("CLIENT_READY"));

        progressController.SetConnected(true);
    }

    private void OnError(string error)
    {
        Debug.LogWarning($"[SceneClient] Error: {error}");
        progressController.SetConnected(false, $"Error ({error})");
    }

    private void OnClose(WebSocketCloseCode code)
    {
        Debug.Log($"[SceneClient] Disconnected ({code})");
        progressController.SetConnected(false);

        if (!_reconnecting) StartCoroutine(Reconnect());
    }

    private IEnumerator Reconnect()
    {
        _reconnecting = true;
        Debug.Log($"[SceneClient] Reconnecting in {reconnectDelay}s…");
        progressController.SetConnected(false, "Reconnecting...");
        yield return new WaitForSeconds(reconnectDelay);
        _reconnecting = false;
        ConnectAsync();
    }

    // ── Message Routing ────────────────────────────────────────────────────

    private void OnMessage(byte[] bytes)
    {
        string json = System.Text.Encoding.UTF8.GetString(bytes);

        ServerMessage msg;
        try { msg = JsonUtility.FromJson<ServerMessage>(json); }
        catch (Exception e)
        {
            Debug.LogWarning($"[SceneClient] Failed to parse message: {e.Message}\n{json}");
            return;
        }

        // Route to the right handler on the main thread
        UnityMainThread.Call(() => Route(msg.type, msg.payload, json));
    }

    private void Route(string type, string payloadJson, string fullJson)
    {
        switch (type)
        {
            case "SCENE_INIT":
                string initPayload = ExtractPayload(fullJson);
                Debug.Log($"[SceneClient] SCENE_INIT payload: {initPayload}");
                var init = JsonUtility.FromJson<SceneInitPayload>(initPayload);

                if (!string.IsNullOrEmpty(init.scene_id) && init.scene_id == _lastSceneId)
                {
                    Debug.Log("[SceneClient] Scene unchanged (same scene_id) — skipping reload");
                    objectManager.RevealIfHidden();
                    break;
                }

                _lastSceneId = init.scene_id;
                objectManager.ApplySceneInit(init);
                paramManager.ApplyParams(init.scene);
                terrainMaterialManager?.Apply(init.terrain_material);
                break;

            case "OBJECT_SPAWN":
                var spawnObj = JsonUtility.FromJson<SceneObject>(ExtractPayload(fullJson));
                objectManager.Spawn(spawnObj);
                break;

            case "OBJECT_UPDATE":
                var update = JsonUtility.FromJson<ObjectUpdatePayload>(ExtractPayload(fullJson));
                objectManager.ApplyUpdate(update);
                break;

            case "OBJECT_DESTROY":
                var destroy = JsonUtility.FromJson<ObjectDestroyPayload>(ExtractPayload(fullJson));
                objectManager.Destroy(destroy.id);
                break;

            case "PROGRESS":
                var progress = JsonUtility.FromJson<SceneProgressPayload>(ExtractPayload(fullJson));
                Debug.Log($"[SceneClient] Progress {progress.step} at {progress.percent * 100.0}%");

                progressController.ReportServerProgress(progress.step, progress.percent);
                break;

            default:
                Debug.LogWarning($"[SceneClient] Unknown message type: {type}");
                break;
        }
    }

    // ── Sending ────────────────────────────────────────────────────────────

    public async void Send(object data)
    {
        if (_ws?.State != WebSocketState.Open) return;
        string json = JsonUtility.ToJson(data); // or use Newtonsoft for complex types
        await _ws.SendText(json);
    }

    /// <summary>Report an in-game event back to the server.</summary>
    public void SendObjectEvent(string objectId, string eventName, object extraData = null)
    {
        Send(new {
            type = "OBJECT_EVENT",
            payload = new { objectId, @event = eventName }
        });
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    // JsonUtility doesn't support nested dynamic payloads well.
    // This extracts the raw "payload" sub-object as a JSON string for
    // a second-pass parse. For production, use Newtonsoft.Json instead.
    private static string ExtractPayload(string fullJson)
    {
        int idx = fullJson.IndexOf("\"payload\":", StringComparison.Ordinal);
        if (idx < 0) return "{}";
        int start = fullJson.IndexOf('{', idx + 10);
        if (start < 0) return "{}";

        int depth = 0, end = start;
        for (int i = start; i < fullJson.Length; i++)
        {
            if (fullJson[i] == '{') depth++;
            else if (fullJson[i] == '}') { depth--; if (depth == 0) { end = i; break; } }
        }
        return fullJson.Substring(start, end - start + 1);
    }
}

// ── Data Models ────────────────────────────────────────────────────────────────

[Serializable] public class ServerMessage   { public string type; public string payload; }
[Serializable] public class SceneParamPayload { public string key; public string value; }
[Serializable] public class ObjectDestroyPayload { public string id; }

[Serializable] public class SceneProgressPayload { public string step; public string detail; public float percent; }

[Serializable]
public class Vec3 { public float x, y, z; }

[Serializable]
public class SwayData
{
    public float amplitude;    // fraction of the object's own footprint, ~[0,1]
    public float frequencyHz;
    public float phase;        // radians, randomized per instance server-side
    public float axisDegrees;  // shared per scene -- the wind's own lean direction
}

[Serializable]
public class PhysicsData
{
    public Vec3   velocity;      // m/s, world space, applied once at spawn
    public Vec3   acceleration;  // m/s^2, applied once at spawn as a single kick
    public string colliderShape; // "box" | "capsule"
    public Vec3   colliderSize;  // box: full extents; capsule: (width, height, depth) -- see PhysicsHandoff
}

[Serializable]
public class SceneObject
{
    public string id;
    public string type;
    public string texture;
    public string mesh;
    // Optional separate, lower-poly mesh asset key loaded as a MeshCollider on
    // this object instead of colliding against the (possibly dense, textured)
    // `mesh` above -- e.g. the terrain's coarse geometry-only collision proxy.
    // Empty -> no collider is attached.
    public string physicsMesh;
    public string name;
    public Vec3   position;
    public Vec3   rotation;
    public Vec3   scale;

    // Animated billboard: an object's extracted clip, split into a color video and
    // a grayscale matte (see server VideoObjectExtractionStage) -- set together or
    // not at all. Empty/null -> render `texture` as a static billboard as before.
    public string videoColor;
    public string videoAlpha;

    // Gentle procedural sway (stationary mesh instances only) or rigid-body physics
    // handoff (moving objects, billboard or mesh) -- at most one of these is set.
    public SwayData    sway;
    public PhysicsData physics;
}

[Serializable]
public class ObjectUpdatePayload
{
    public string    id;
    public SceneObject changes; // partial — only changed fields
}

[Serializable]
public class ExtrinsicsParams
{
    public float[] rotation;     // 9
    public float[]   translation;  // 3
}

[Serializable]
public class SceneLightingData
{
    public string ldr;    // base64 PNG — LDR equirectangular environment map
    public string log;    // base64 PNG — log-domain environment map
    public int    width;
    public int    height;
    public SunData sun;   // dominant directional light, or null when overcast
}

// Key light extracted from the environment map (server SceneLighting.sun).
[Serializable]
public class SunData
{
    // Unit vector pointing FROM the scene TOWARD the sun, in the panorama's own
    // frame — apply skyboxRotation before use, same as the skybox.
    public float[] direction;   // 3
    public string  color;       // "#rrggbb", tint only; brightness is `intensity`
    // Relative, not photometric: the environment maps are LDR/log-compressed with
    // no absolute calibration, so this is a directionality heuristic scaled so a
    // typical clear sky lands near 1. See SceneLighting.sun's docstring.
    public float   intensity;
}

[Serializable]
public class SceneParams
{
    public string ambientColor;
    public float  gravity;
    public float  timeScale;
    public float  nearClipPlane;
    public float  farClipPlane;
    public string skybox;
    public float  skyboxRotation;
    public float  cameraHeight;
    public float  eyeHeightMeters;
    public float  terrainCenterY;

    public SceneObject[]    objects;
    public ExtrinsicsParams extrinsics;
    public SceneLightingData lighting;
}

[Serializable]
public class SplatLayerData
{
    public string name;        // region label, e.g. "vegetation", "terrain"
    public string tile;        // base64-encoded PNG — tileable texture
    public float  tile_factor; // UV repeat count across terrain (0 treated as 1.0)
    public bool   equirect;    // true → sample with equirectangular world-position UV
    public float  smoothness;  // PBR smoothness [0,1]: 0 = matte, ~0.88 = water
}

[Serializable]
public class SplatMaterialData
{
    public int              tile_size;    // px per tile
    public SplatLayerData[] layers;       // ordered; channel assignment is positional
    public string[]         blend_maps;   // base64-encoded RGBA PNGs (up to 2)
                                          // blend_maps[0] RGBA → layers[0..3]
                                          // blend_maps[1] RGBA → layers[4..7]
}

[Serializable]
public class SceneInitPayload
{
    public string           scene_id;            // identifies the generated content; unchanged id = no-op reconnect
    public SceneParams      scene;
    public SplatMaterialData terrain_material;   // null when not generated
}
