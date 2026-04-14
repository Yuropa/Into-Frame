using UnityEngine;

public class Billboard : MonoBehaviour
{
    void LateUpdate()
    {
        if (Camera.main != null)
        {
            Vector3 direction = Camera.main.transform.position - transform.position;
            direction.y = 0; // Ignore vertical tilt
            transform.rotation = Quaternion.LookRotation(direction);
        }
    }
}
