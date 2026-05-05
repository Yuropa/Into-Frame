using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;
using GLTFast;

/// <summary>
/// Manages all server-driven GameObjects in the scene.
/// Attach to a persistent GameObject alongside SceneClient.
/// </summary>
public class SceneObjectManager : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject billboardPrefab;

    [Header("Scene")]
    public GameObject sceneRoot;

    [Header("Interpolation")]
    [Tooltip("Smooth out position/rotation updates from server")]
    public float lerpSpeed = 10f;

    [Header("Assets")]
    public GameObject server;

    [Header("UI")]
    public ProgressController progress;

    private readonly Dictionary<string, TrackedObject> _tracked = new();
    private readonly Dictionary<string, Texture2D> _textureCache = new();
    private readonly Dictionary<string, List<GameObject>> _textureWaiters = new();

    // ── Public API ─────────────────────────────────────────────────────────

    public void ApplySceneInit(SceneInitPayload init)
    {
        _queueGeneration++;
        _taskQueue.Clear();
        _totalTasks = 0;
        _completedTasks = 0;

        foreach (var t in _tracked.Values)
            if (t.go != null) Destroy(t.go);
        _tracked.Clear();

        if (sceneRoot != null) sceneRoot.SetActive(false); // hide at start

        if (init.scene.objects == null) return;

        foreach (var obj in init.scene.objects)
            SpawnImmediate(obj);

        if (!_isProcessingQueue && _taskQueue.Count > 0)
            StartCoroutine(ProcessQueue());
        else if (_taskQueue.Count == 0)
        {
            // No meshes to load (e.g. billboards only) — reveal immediately
            if (sceneRoot != null) sceneRoot.SetActive(true);
            progress?.ReportSceneComplete();
        }
    }

    public void Spawn(SceneObject data)
    {
        if (_tracked.ContainsKey(data.id))
        {
            ApplyUpdate(new ObjectUpdatePayload { id = data.id, changes = data });
            return;
        }

        switch (data.type)
        {
            case "billboard":
                SpawnBillboard(data);
                break;
            case "mesh":
                SpawnMeshImmediate(data);
                break;
            default:
                Debug.LogWarning($"[SceneObjectManager] Unsupported type '{data.type}'");
                break;
        }
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

        if (!string.IsNullOrEmpty(changes.mesh) && changes.mesh != tracked.data.mesh)
        {
            tracked.data.mesh = changes.mesh;
            var capturedTracked = tracked;
            var capturedMesh = changes.mesh;
            EnqueueTask(() => ReloadMesh(capturedTracked, capturedMesh));
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

    // ── Async Task Queue ───────────────────────────────────────────────────

    private readonly Queue<Func<IEnumerator>> _taskQueue = new();
    private bool _isProcessingQueue = false;
    private int _queueGeneration = 0;
    private int _totalTasks = 0;
    private int _completedTasks = 0;

    private void EnqueueTask(Func<IEnumerator> task)
    {
        _totalTasks++;
        _taskQueue.Enqueue(task);
        if (!_isProcessingQueue)
            StartCoroutine(ProcessQueue());
    }


    private IEnumerator ProcessQueue()
    {
        _isProcessingQueue = true;

        while (_taskQueue.Count > 0)
        {
            var task = _taskQueue.Dequeue();
            yield return StartCoroutine(task());

            _completedTasks++;
            progress?.ReportSceneProgress(_completedTasks, _totalTasks);

            yield return null;
        }

        // Only reveal once the queue is fully drained
        if (sceneRoot != null) sceneRoot.SetActive(true);
        progress?.ReportSceneComplete();

        _isProcessingQueue = false;
        Debug.Log("[SceneObjectManager] Scene ready");
    }

    // ── Spawn Helpers ──────────────────────────────────────────────────────

    private void SpawnImmediate(SceneObject data)
    {
        switch (data.type)
        {
            case "billboard": SpawnBillboard(data); break;
            case "mesh":      SpawnMeshImmediate(data); break;
            default:
                Debug.LogWarning($"[SceneObjectManager] Unsupported type '{data.type}'");
                break;
        }
    }

    private AssetServer _assetServer;
    private AssetServer assetServer()
    {
        return _assetServer ??= server.GetComponent<AssetServer>();
    }

    private void SpawnBillboard(SceneObject data)
    {
        if (billboardPrefab == null)
        {
            Debug.LogError("[SceneObjectManager] billboardPrefab is not assigned in the Inspector");
            return;
        }

        var go = Instantiate(billboardPrefab, ToVec3(data.position), ToQuat(data.rotation));
        go.name = $"[billboard] {data.id[..6]}";
        go.transform.localScale = ToVec3(data.scale);
        if (sceneRoot != null)
            go.transform.SetParent(sceneRoot.transform, worldPositionStays: true);
        go.AddComponent<ServerObjectTag>().serverId = data.id;

        _tracked[data.id] = new TrackedObject
        {
            go        = go,
            data      = data,
            targetPos = ToVec3(data.position),
            targetRot = ToQuat(data.rotation),
        };

        if (!string.IsNullOrEmpty(data.texture))
            StartCoroutine(ApplyTexture(go, data.texture));
    }

    private void SpawnMeshImmediate(SceneObject data)
    {
        var container = new GameObject($"[mesh] {data.id[..6]}");
        container.transform.position   = ToVec3(data.position);
        container.transform.rotation   = ToQuat(data.rotation);
        container.transform.localScale = ToVec3(data.scale);
        if (sceneRoot != null)
            container.transform.SetParent(sceneRoot.transform, worldPositionStays: true);
        container.AddComponent<ServerObjectTag>().serverId = data.id;

        _tracked[data.id] = new TrackedObject
        {
            go        = container,
            data      = data,
            targetPos = ToVec3(data.position),
            targetRot = ToQuat(data.rotation),
        };

        var capturedContainer = container;
        var capturedMesh = data.mesh;
        var capturedGeneration = _queueGeneration;
        EnqueueTask(() => LoadGlbInto(capturedContainer, capturedMesh, capturedGeneration));
    }

    private IEnumerator ReloadMesh(TrackedObject tracked, string meshId)
    {
        foreach (Transform child in tracked.go.transform)
            Destroy(child.gameObject);

        yield return LoadGlbInto(tracked.go, meshId, _queueGeneration);
    }

    // ── GLB Loading ────────────────────────────────────────────────────────

    private IEnumerator LoadGlbInto(GameObject container, string meshId, int generation)
    {
        if (string.IsNullOrEmpty(meshId)) yield break;

        using var req = assetServer().GetResource(meshId);
        yield return req.SendWebRequest();

        if (generation != _queueGeneration)
        {
            Debug.Log($"[SceneObjectManager] Discarding stale load for '{meshId}'");
            yield break;
        }

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[SceneObjectManager] Failed to download '{meshId}': {req.error}");
            yield break;
        }

        byte[] glbBytes = req.downloadHandler.data.ToArray();
        var logger = new GLTFastLogger();
        var gltf = new GltfImport(logger: logger);
        var glbTask = gltf.Load(glbBytes, new System.Uri(req.url));

        while (!glbTask.IsCompleted)
            yield return null;

        if (generation != _queueGeneration) yield break;

        if (glbTask.IsFaulted)
        {
            Debug.LogError($"[SceneObjectManager] GLTFast exception on '{meshId}': {glbTask.Exception}");
            yield break;
        }

        if (!glbTask.Result)
        {
            Debug.LogError($"[SceneObjectManager] GLTFast failed to parse '{meshId}'. " +
                           $"Errors: {string.Join(" | ", logger.Errors)}");
            yield break;
        }

        var instantiateTask = gltf.InstantiateMainSceneAsync(container.transform);
        while (!instantiateTask.IsCompleted)
            yield return null;

        if (instantiateTask.IsFaulted)
        {
            Debug.LogError($"[SceneObjectManager] Instantiation failed for '{meshId}': {instantiateTask.Exception}");
            yield break;
        }

        Debug.Log($"[SceneObjectManager] Mesh '{meshId}' loaded into {container.name}");
    }

    // ── Texture Loading ────────────────────────────────────────────────────

    private IEnumerator ApplyTexture(GameObject go, string textureId)
    {
        if (_textureCache.TryGetValue(textureId, out Texture2D cached))
        {
            SetTexture(go, cached);
            yield break;
        }

        if (_textureWaiters.ContainsKey(textureId))
        {
            _textureWaiters[textureId].Add(go);
            yield break;
        }

        _textureWaiters[textureId] = new List<GameObject> { go };

        using var req = assetServer().GetTexture(textureId);
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[SceneObjectManager] Failed to download texture '{textureId}': {req.error}");
            _textureWaiters.Remove(textureId);
            yield break;
        }

        Texture2D tex = DownloadHandlerTexture.GetContent(req);
        _textureCache[textureId] = tex;

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
        mpb.SetTexture("_BaseMap", tex);
        renderer.SetPropertyBlock(mpb);
    }

    // ── Smooth Interpolation ───────────────────────────────────────────────

    private void Update()
    {
        // float t = Time.deltaTime * lerpSpeed;
        // foreach (var tracked in _tracked.Values)
        // {
        //     if (tracked.go == null) continue;
        //     tracked.go.transform.position = Vector3.Lerp(tracked.go.transform.position, tracked.targetPos, t);
        //     tracked.go.transform.rotation = Quaternion.Slerp(tracked.go.transform.rotation, tracked.targetRot, t);
        // }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static Vector3    ToVec3(Vec3 v) => v != null ? new Vector3(v.x, v.y, v.z) : Vector3.zero;
    private static Quaternion ToQuat(Vec3 e) => e != null ? Quaternion.Euler(e.x, e.y, e.z) : Quaternion.identity;

    private class TrackedObject
    {
        public GameObject go;
        public SceneObject data;
        public Vector3     targetPos;
        public Quaternion  targetRot;
    }
}

public class ServerObjectTag : MonoBehaviour
{
    public string serverId;
}

class GLTFastLogger : GLTFast.Logging.ICodeLogger
{
    public readonly List<string> Errors = new();

    public void Error(GLTFast.Logging.LogCode code, params string[] messages)
    {
        var msg = $"[GLTFast ERROR {code}] {string.Join(", ", messages)}";
        Errors.Add(msg);
        Debug.LogError(msg);
    }
    public void Warning(GLTFast.Logging.LogCode code, params string[] messages) =>
        Debug.LogWarning($"[GLTFast WARN {code}] {string.Join(", ", messages)}");
    public void Info(GLTFast.Logging.LogCode code, params string[] messages) =>
        Debug.Log($"[GLTFast INFO {code}] {string.Join(", ", messages)}");
    public void Error(string message)
    {
        Errors.Add(message);
        Debug.LogError($"[GLTFast ERROR] {message}");
    }
    public void Warning(string message) =>
        Debug.LogWarning($"[GLTFast WARN] {message}");
    public void Info(string message) =>
        Debug.Log($"[GLTFast INFO] {message}");
}
