using UnityEngine;

public class Billboard : MonoBehaviour
{
    void LateUpdate()
    {
        if (Camera.main == null) return;
        Vector3 dir = Camera.main.transform.position - transform.position;
        dir.y = 0f;
        if (dir.sqrMagnitude > 0f)
            transform.rotation = Quaternion.LookRotation(-dir);
    }
}
