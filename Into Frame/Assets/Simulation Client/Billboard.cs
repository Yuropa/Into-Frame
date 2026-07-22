using UnityEngine;

public class Billboard : MonoBehaviour
{
    // A billboard driven by PhysicsHandoff owns its own rotation (tumbling with
    // the Rigidbody it's parented to) -- always-face-camera would fight that
    // every frame and the quad would never actually appear to tumble.
    private Rigidbody _rigidbody;

    void Awake()
    {
        _rigidbody = GetComponentInParent<Rigidbody>();
    }

    void LateUpdate()
    {
        if (_rigidbody != null) return;
        if (Camera.main == null) return;
        Vector3 dir = Camera.main.transform.position - transform.position;
        dir.y = 0f;
        if (dir.sqrMagnitude > 0f)
            transform.rotation = Quaternion.LookRotation(-dir);
    }
}
