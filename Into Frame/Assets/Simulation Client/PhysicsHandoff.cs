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

    public void Apply(PhysicsData data)
    {
        AddCollider(data);

        var rb = gameObject.AddComponent<Rigidbody>();
        rb.linearVelocity = ToVector3(data.velocity);

        float verticalAccel = data.acceleration != null ? data.acceleration.y : Physics.gravity.y;
        rb.useGravity = Mathf.Abs(verticalAccel - Physics.gravity.y) < GravityMatchToleranceMs2;
    }

    private void AddCollider(PhysicsData data)
    {
        Vector3 size = ToVector3(data.colliderSize);
        size = Vector3.Max(size, Vector3.one * 0.05f); // never a degenerate/zero-volume collider

        if (data.colliderShape == "capsule")
        {
            var capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.direction = 1; // Y axis
            capsule.height = size.y;
            capsule.radius = Mathf.Max(size.x, size.z) * 0.5f;
        }
        else
        {
            var box = gameObject.AddComponent<BoxCollider>();
            box.size = size;
        }
    }

    private static Vector3 ToVector3(Vec3 v) => v != null ? new Vector3(v.x, v.y, v.z) : Vector3.zero;
}
