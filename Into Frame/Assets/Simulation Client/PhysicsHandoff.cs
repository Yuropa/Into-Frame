using UnityEngine;

/// <summary>
/// Applies a detected moving object's estimated initial velocity once at spawn,
/// then lets Unity's own Rigidbody/physics own the motion from then on -- no
/// continuous re-sync toward the source clip's recorded trajectory (see
/// server SceneAnimationStage/ObjectMotionClassificationStage).
/// </summary>
public class PhysicsHandoff : MonoBehaviour
{
    // If the video's own measured vertical acceleration is close to Earth gravity,
    // the object was already behaving like something in free-fall, so Unity's own
    // gravity is left on. Otherwise (e.g. a bird in roughly level flight) turning
    // gravity off keeps it from nosediving the instant physics takes over, which
    // the source video never actually showed.
    private const float GravityMatchToleranceMs2 = 3.0f;

    private PhysicsData _pending;

    public void Apply(PhysicsData data)
    {
        // Deferred, not applied here: SceneObjectManager.SpawnMeshImmediate adds this
        // component before it even queues the GLB download, so there is no Renderer to
        // measure yet and no geometry for a collider to bound. Starting a Rigidbody
        // falling against a collider placed on guesswork is exactly how an object ends
        // up resting somewhere its visible mesh isn't. Same wait-until-it-exists pattern
        // WindSway uses for its bones.
        _pending = data;
    }

    private void LateUpdate()
    {
        if (_pending == null) return;

        var renderer = GetComponentInChildren<Renderer>();
        if (renderer == null) return; // GLB still loading

        var data = _pending;
        _pending = null;

        AddCollider(data, renderer);

        var rb = gameObject.AddComponent<Rigidbody>();
        rb.linearVelocity = ToVector3(data.velocity);

        float verticalAccel = data.acceleration != null ? data.acceleration.y : Physics.gravity.y;
        rb.useGravity = Mathf.Abs(verticalAccel - Physics.gravity.y) < GravityMatchToleranceMs2;
    }

    private void AddCollider(PhysicsData data, Renderer renderer)
    {
        Vector3 size = ToVector3(data.colliderSize);
        size = Vector3.Max(size, Vector3.one * 0.05f); // never a degenerate/zero-volume collider

        // colliderSize arrives in WORLD metres (SceneAnimationStage derives it from the
        // object's own world_width/world_height), but collider extents are expressed in
        // the transform's LOCAL space -- and SceneObjectManager.SpawnMeshImmediate puts
        // data.scale on this very GameObject, which for a category mesh is the object's
        // whole world size (2-10 for the trees on the Rainier capture). Assigning the
        // world size directly therefore produced a collider that many times too large,
        // so the body came to rest with its visible mesh nowhere near the ground.
        Vector3 lossy = transform.lossyScale;
        size = new Vector3(
            size.x / Mathf.Max(Mathf.Abs(lossy.x), 1e-4f),
            size.y / Mathf.Max(Mathf.Abs(lossy.y), 1e-4f),
            size.z / Mathf.Max(Mathf.Abs(lossy.z), 1e-4f)
        );

        // The server places a mesh so its BOTTOM sits on the terrain, which puts this
        // transform's origin partway up the object, not at its centre (see
        // SceneGenerationStage's mesh_place_y / mesh_min_y -- for a tree the origin
        // lands ~40% of the way up). A collider centred on the origin therefore
        // straddles the ground instead of resting on it, and the solver's first
        // push-out is what reads as the object dropping and then sticking.
        Vector3 center = transform.InverseTransformPoint(renderer.bounds.center);

        if (data.colliderShape == "capsule")
        {
            var capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.direction = 1; // Y axis
            capsule.height = size.y;
            capsule.radius = Mathf.Max(size.x, size.z) * 0.5f;
            capsule.center = center;
        }
        else
        {
            var box = gameObject.AddComponent<BoxCollider>();
            box.size = size;
            box.center = center;
        }
    }

    private static Vector3 ToVector3(Vec3 v) => v != null ? new Vector3(v.x, v.y, v.z) : Vector3.zero;
}
