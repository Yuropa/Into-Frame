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

    private WebSocket _ws;
    private bool _reconnecting = false;

    // ── Unity Lifecycle ────────────────────────────────────────────────────

    private void Start()
    {
        if (objectManager == null) objectManager = FindObjectOfType<SceneObjectManager>();
        if (paramManager  == null) paramManager  = FindObjectOfType<SceneParamManager>();
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
    }

    private void OnError(string error)
    {
        Debug.LogWarning($"[SceneClient] Error: {error}");
    }

    private void OnClose(WebSocketCloseCode code)
    {
        Debug.Log($"[SceneClient] Disconnected ({code})");
        if (!_reconnecting) StartCoroutine(Reconnect());
    }

    private IEnumerator Reconnect()
    {
        _reconnecting = true;
        Debug.Log($"[SceneClient] Reconnecting in {reconnectDelay}s…");
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
                objectManager.ApplySceneInit(init);
                paramManager.ApplyParams(init.scene);
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
public class SceneObject
{
    public string id;
    public string type;        // "cube", "sphere", "capsule", "cylinder", "prefab"
    public string prefabName;
    public Vec3   position;
    public Vec3   rotation;
    public Vec3   scale;
    public string color;       // hex e.g. "#ff0000"
    // params is free-form JSON — handle manually if needed
}

[Serializable]
public class ObjectUpdatePayload
{
    public string    id;
    public SceneObject changes; // partial — only changed fields
}

[Serializable]
public class SceneParams
{
    public string ambientColor;
    public float  gravity;
    public float  timeScale;
}

[Serializable]
public class SceneInitPayload
{
    public SceneParams   scene;
    public SceneObject[] objects;
}
