using UnityEngine;

/// <summary>
/// Drives the 3-bone sway skeleton CategoryMeshRiggingStage bakes into a shared
/// category mesh, using this instance's own sway parameters (SceneAnimationStage).
/// Procedural, not a baked animation clip -- many differently-phased instances
/// share the same rigged mesh asset, so the timing has to live on the instance,
/// not the asset.
/// </summary>
public class WindSway : MonoBehaviour
{
    private static readonly string[] BoneNames = { "SwayBone_0", "SwayBone_1", "SwayBone_2" };
    // Cantilever falloff: rotating the base bone bends everything above it in the
    // chain, so amplitude has to shrink toward the root or the whole mesh would
    // swing like a rigid pendulum instead of bending mostly near the tip.
    private static readonly float[] AmplitudeFalloff = { 0.2f, 0.6f, 1.0f };

    private Transform[] _bones;
    private SwayData _data;

    public void Apply(SwayData data)
    {
        _data = data;
    }

    private void LateUpdate()
    {
        if (_data == null) return;

        // The bone hierarchy only exists once GLTFast finishes instantiating the
        // mesh (SceneObjectManager adds this component before that completes, so
        // the mesh can start swaying the instant it appears) -- keep looking each
        // frame until found rather than giving up on an early miss.
        if (_bones == null)
        {
            var found = new Transform[BoneNames.Length];
            for (int i = 0; i < BoneNames.Length; i++)
                found[i] = FindRecursive(transform, BoneNames[i]);
            if (found[0] == null) return;
            _bones = found;
        }

        float t = Time.time * _data.frequencyHz * 2f * Mathf.PI + _data.phase;
        // amplitude is a normalized [0,1] fraction of the object's own footprint --
        // 45 degrees at amplitude=1 is a plain heuristic scale, not a measured unit.
        float swing = Mathf.Sin(t) * _data.amplitude * 45f;
        Vector3 axis = Quaternion.Euler(0f, _data.axisDegrees, 0f) * Vector3.right;

        for (int i = 0; i < _bones.Length; i++)
        {
            if (_bones[i] == null) continue;
            Vector3 localAxis = _bones[i].InverseTransformDirection(axis);
            _bones[i].localRotation = Quaternion.AngleAxis(swing * AmplitudeFalloff[i], localAxis);
        }
    }

    private static Transform FindRecursive(Transform root, string name)
    {
        if (root.name == name) return root;
        foreach (Transform child in root)
        {
            var found = FindRecursive(child, name);
            if (found != null) return found;
        }
        return null;
    }
}
