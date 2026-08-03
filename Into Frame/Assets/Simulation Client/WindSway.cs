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

    // ── Diagnostics ────────────────────────────────────────────────────────
    // Read by SwayDiagnostics. This component fails silently by design -- no bones
    // found means LateUpdate returns and the instance simply never moves -- so the
    // only way to tell "not driven" from "driven but imperceptible" is to expose the
    // state and the angle actually being applied.

    /// <summary>Global multiplier on every instance's amplitude. 1 = as authored.
    /// Crank it up to tell a plumbing failure (still nothing moves) apart from a
    /// tuning problem (motion appears, so it was only too subtle to see).</summary>
    public static float DebugAmplitudeScale = 1f;

    /// <summary>True once the SwayBone_* chain has been located in the instantiated GLB.</summary>
    public bool BonesFound => _bones != null;

    /// <summary>True once server sway params have been applied.</summary>
    public bool HasData => _data != null;

    /// <summary>Degrees applied to the tip bone last frame; 0 before the first drive.</summary>
    public float CurrentSwing { get; private set; }

    /// <summary>Peak |degrees| this instance has ever applied -- a value that stays at 0
    /// while BonesFound is true means it is being driven but not perceptibly.</summary>
    public float PeakSwing { get; private set; }

    public void Apply(SwayData data)
    {
        _data = data;
        // Every instance in a scene carries the same server wind_axis_degrees (see
        // SceneAnimationStage) -- cheap and idempotent to re-set the shared field's
        // direction from whichever instance happens to apply first.
        WindField.Instance.SetDirection(data.axisDegrees);
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

        // The per-instance frequency/phase (server-jittered, see SceneAnimationStage)
        // still drives each instance's own flutter -- that's what keeps a whole field
        // of grass from moving as one rigid sheet even while a single gust event is
        // passing over all of it together. WindField layers a shared clock delay on
        // top: this point's gust envelope AND its oscillation both run "late" by
        // however long the wind takes to physically reach it, so a gust visibly
        // starts on one side of the scene and sweeps across to the other instead of
        // the whole scene flaring up in lockstep -- and between gusts, motion drops
        // to a low ambient idle rather than continuing forever at constant strength.
        var wind = WindField.Instance;
        float delay     = wind.SampleDelaySeconds(transform.position);
        float envelope  = wind.SampleStrength(transform.position, Time.time);

        float t = (Time.time - delay) * _data.frequencyHz * 2f * Mathf.PI + _data.phase;
        // amplitude is a normalized [0,1] fraction of the object's own footprint --
        // 45 degrees at amplitude=1 is a plain heuristic scale, not a measured unit.
        float swing = Mathf.Sin(t) * _data.amplitude * 45f * envelope * DebugAmplitudeScale;
        CurrentSwing = swing;
        PeakSwing = Mathf.Max(PeakSwing, Mathf.Abs(swing));
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
