using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Manages all server-driven GameObjects in the scene.
/// Attach to a persistent GameObject alongside SceneClient.
/// </summary>
public class SceneObjectManager : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject billboardPrefab;  // drag your "Billboard" prefab here

    [Header("Asset Server")]
    public string assetBaseUrl = "http://localhost:3000/assets/";

    [Header("Interpolation")]
    [Tooltip("Smooth out position/rotation updates from server")]
    public float lerpSpeed = 10f;

    // id → tracked object
    private readonly Dictionary<string, TrackedObject> _tracked = new();

    // textureId → cached texture (so we don't re-download the same texture)
    private readonly Dictionary<string, Texture2D> _textureCache = new();

    // textureId → list of GameObjects waiting on that texture to download
    private readonly Dictionary<string, List<GameObject>> _textureWaiters = new();

    // ── Public API ─────────────────────────────────────────────────────────

    public void ApplySceneInit(SceneInitPayload init)
    {
        // Destroy any previously tracked objects (reconnect scenario)
        foreach (var t in _tracked.Values)
            if (t.go != null) Destroy(t.go);
        _tracked.Clear();

        if (init.scene.objects == null) return;
        foreach (var obj in init.scene.objects)
            Spawn(obj);
    }

    public void Spawn(SceneObject data)
    {
        if (_tracked.ContainsKey(data.id))
        {
            ApplyUpdate(new ObjectUpdatePayload { id = data.id, changes = data });
            return;
        }

        if (data.type != "billboard")
        {
            Debug.LogWarning($"[SceneObjectManager] Unsupported type '{data.type}' — only 'billboard' is handled");
            return;
        }

        if (billboardPrefab == null)
        {
            Debug.LogError("[SceneObjectManager] billboardPrefab is not assigned in the Inspector");
            return;
        }

        GameObject go = Instantiate(billboardPrefab, ToVec3(data.position), ToQuat(data.rotation));
        go.name = $"[billboard] {data.id[..6]}";
        go.transform.localScale = ToVec3(data.scale);

        var tag = go.AddComponent<ServerObjectTag>();
        tag.serverId = data.id;

        _tracked[data.id] = new TrackedObject
        {
            go        = go,
            data      = data,
            targetPos = ToVec3(data.position),
            targetRot = ToQuat(data.rotation),
        };

        // Apply texture — from cache or kick off download
        if (!string.IsNullOrEmpty(data.texture))
            StartCoroutine(ApplyTexture(go, data.texture));
    }

    public void ApplyUpdate(ObjectUpdatePayload update)
    {
        if (!_tracked.TryGetValue(update.id, out var tracked)) return;

        var changes = update.changes;
        if (changes == null) return;

        if (changes.position != null)
            tracked.targetPos = ToVec3(changes.position);

        if (changes.rotation != null)
            tracked.targetRot = ToQuat(changes.rotation);

        if (changes.scale != null && tracked.go != null)
            tracked.go.transform.localScale = ToVec3(changes.scale);

        if (!string.IsNullOrEmpty(changes.texture) && changes.texture != tracked.data.texture)
        {
            tracked.data.texture = changes.texture;
            StartCoroutine(ApplyTexture(tracked.go, changes.texture));
        }

        if (changes.position != null) tracked.data.position = changes.position;
        if (changes.rotation != null) tracked.data.rotation = changes.rotation;
        if (changes.scale    != null) tracked.data.scale    = changes.scale;
    }

    public void Destroy(string id)
    {
        if (!_tracked.TryGetValue(id, out var tracked)) return;
        if (tracked.go != null) Destroy(tracked.go);
        _tracked.Remove(id);
    }

    // ── Texture Loading ────────────────────────────────────────────────────

    private IEnumerator ApplyTexture(GameObject go, string textureId)
    {
        // Already cached — apply immediately
        if (_textureCache.TryGetValue(textureId, out Texture2D cached))
        {
            SetTexture(go, cached);
            yield break;
        }

        // Download already in flight — join the waitlist
        if (_textureWaiters.ContainsKey(textureId))
        {
            _textureWaiters[textureId].Add(go);
            yield break;
        }

        // Start a new download
        _textureWaiters[textureId] = new List<GameObject> { go };

        string url = assetBaseUrl + textureId;
        Debug.Log($"[SceneObjectManager] Downloading texture: {url}");

        using var req = UnityWebRequestTexture.GetTexture(url);
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[SceneObjectManager] Failed to download '{textureId}': {req.error}");
            _textureWaiters.Remove(textureId);
            yield break;
        }

        Texture2D tex = DownloadHandlerTexture.GetContent(req);
        _textureCache[textureId] = tex;

        // Apply to all objects that were waiting on this texture
        foreach (var waiter in _textureWaiters[textureId])
            if (waiter != null) SetTexture(waiter, tex);

        _textureWaiters.Remove(textureId);
    }

    private static void SetTexture(GameObject go, Texture2D tex)
    {
        var renderer = go.GetComponentInChildren<Renderer>();
        if (renderer == null) return;

        var mpb = new MaterialPropertyBlock();
        renderer.GetPropertyBlock(mpb);
        mpb.SetTexture("_BaseMap", tex);  // URP Lit shader — change to "_MainTex" for Built-in
        renderer.SetPropertyBlock(mpb);
    }

    // ── Smooth Interpolation ────────────────────────────────────────────────

    private void Update()
    {
        float t = Time.deltaTime * lerpSpeed;
        foreach (var tracked in _tracked.Values)
        {
            if (tracked.go == null) continue;
            tracked.go.transform.position = Vector3.Lerp(tracked.go.transform.position, tracked.targetPos, t);
            tracked.go.transform.rotation = Quaternion.Slerp(tracked.go.transform.rotation, tracked.targetRot, t);
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static Vector3    ToVec3(Vec3 v) => v != null ? new Vector3(v.x, v.y, v.z) : Vector3.zero;
    private static Quaternion ToQuat(Vec3 e) => e != null ? Quaternion.Euler(e.x, e.y, e.z) : Quaternion.identity;

    // ── Inner Types ────────────────────────────────────────────────────────

    private class TrackedObject
    {
        public GameObject go;
        public SceneObject data;
        public Vector3     targetPos;
        public Quaternion  targetRot;
    }
}

/// <summary>Component added to every server-spawned object for identification.</summary>
public class ServerObjectTag : MonoBehaviour
{
    public string serverId;
}