using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Manages all server-driven GameObjects in the scene.
/// Attach to a persistent GameObject alongside SceneClient.
/// </summary>
public class SceneObjectManager : MonoBehaviour
{
    [Header("Interpolation")]
    [Tooltip("Smooth out position/rotation updates from server")]
    public float lerpSpeed = 10f;

    // id → wrapper holding the GameObject + server target state
    private readonly Dictionary<string, TrackedObject> _tracked = new();

    // ── Public API ─────────────────────────────────────────────────────────

    public void ApplySceneInit(SceneInitPayload init)
    {
        // Destroy any previously tracked objects (reconnect scenario)
        foreach (var t in _tracked.Values)
            if (t.go != null) Destroy(t.go);
        _tracked.Clear();

        if (init.objects == null) return;
        foreach (var obj in init.objects)
            Spawn(obj);
    }

    public void Spawn(SceneObject data)
    {
        // if (_tracked.ContainsKey(data.id))
        // {
        //     // Already exists — treat as update
        //     ApplyUpdate(new ObjectUpdatePayload { id = data.id, changes = data });
        //     return;
        // }

        // GameObject prefab = ResolvePrefab(data);
        // if (prefab == null)
        // {
        //     Debug.LogWarning($"[SceneObjectManager] No prefab for type '{data.type}' / prefabName '{data.prefabName}'");
        //     return;
        // }

        // GameObject go = Instantiate(prefab, ToVec3(data.position), ToQuat(data.rotation));
        // go.name = $"[server] {data.type}_{data.id[..6]}";
        // go.transform.localScale = ToVec3(data.scale);
        // ApplyColor(go, data.color);

        // // Tag with server ID for easy lookup
        // var tag = go.AddComponent<ServerObjectTag>();
        // tag.serverId = data.id;

        // _tracked[data.id] = new TrackedObject
        // {
        //     go      = go,
        //     data    = data,
        //     targetPos = ToVec3(data.position),
        //     targetRot = ToQuat(data.rotation),
        // };
    }

    public void ApplyUpdate(ObjectUpdatePayload update)
    {
        if (!_tracked.TryGetValue(update.id, out var tracked)) return;

        var changes = update.changes;
        if (changes == null) return;

        // Update target state for smooth lerping
        if (changes.position != null)
            tracked.targetPos = ToVec3(changes.position);

        if (changes.rotation != null)
            tracked.targetRot = ToQuat(changes.rotation);

        if (changes.scale != null && tracked.go != null)
            tracked.go.transform.localScale = ToVec3(changes.scale);

        if (!string.IsNullOrEmpty(changes.color))
            ApplyColor(tracked.go, changes.color);

        // Merge changed data fields
        if (changes.position != null) tracked.data.position = changes.position;
        if (changes.rotation != null) tracked.data.rotation = changes.rotation;
        if (changes.scale    != null) tracked.data.scale    = changes.scale;
        if (!string.IsNullOrEmpty(changes.color)) tracked.data.color = changes.color;
    }

    public void Destroy(string id)
    {
        if (!_tracked.TryGetValue(id, out var tracked)) return;
        if (tracked.go != null) Destroy(tracked.go);
        _tracked.Remove(id);
    }

    // ── Smooth Interpolation ────────────────────────────────────────────────

    private void Update()
    {
        float t = Time.deltaTime * lerpSpeed;
        foreach (var tracked in _tracked.Values)
        {
            if (tracked.go == null) continue;
            tracked.go.transform.position = Vector3.Lerp(
                tracked.go.transform.position, tracked.targetPos, t);
            tracked.go.transform.rotation = Quaternion.Slerp(
                tracked.go.transform.rotation, tracked.targetRot, t);
        }
    }

    private static Vector3    ToVec3(Vec3 v) => v != null ? new Vector3(v.x, v.y, v.z) : Vector3.zero;
    private static Quaternion ToQuat(Vec3 e) => e != null ? Quaternion.Euler(e.x, e.y, e.z) : Quaternion.identity;

    private static void ApplyColor(GameObject go, string hex)
    {
        if (go == null || string.IsNullOrEmpty(hex)) return;
        if (!ColorUtility.TryParseHtmlString(hex, out Color color)) return;

        foreach (var renderer in go.GetComponentsInChildren<Renderer>())
        {
            // Use MaterialPropertyBlock to avoid creating material instances
            var mpb = new MaterialPropertyBlock();
            renderer.GetPropertyBlock(mpb);
            mpb.SetColor("_Color", color);
            renderer.SetPropertyBlock(mpb);
        }
    }

    // ── Inner Types ────────────────────────────────────────────────────────

    private class TrackedObject
    {
        public GameObject go;
        public SceneObject data;
        public Vector3     targetPos;
        public Quaternion  targetRot;
    }

    [System.Serializable]
    public class PrefabEntry
    {
        public string     name;
        public GameObject prefab;
    }
}

/// <summary>Component added to every server-spawned object for identification.</summary>
public class ServerObjectTag : MonoBehaviour
{
    public string serverId;
}
