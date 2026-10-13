using System.Collections.Generic;
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
    // Cantilever falloff, per joint, base to tip. Rotations compound down the chain,
    // so the tip's total deflection is the sum of all three.
    //
    // SwayBone_0 MUST stay at 0. It is the root of the chain, pinned at the mesh's
    // own y_min (see CategoryMeshRiggingStage) -- i.e. at the ground -- and its
    // transform applies to every vertex in the mesh, including the ones sitting on
    // the terrain. Rotating it does not bend anything; it swings the whole object
    // rigidly about its base point. At 0.2 that was ~3 degrees of whole-mesh roll at
    // a typical amplitude, lifting the outer base edge of a 0.35 m tuft ~1 cm in and
    // out of the ground every cycle, which is what made grass read as an oscillating
    // slab rather than as blades moving over a planted base. Bending is what the
    // joints ABOVE the root are for.
    private static readonly float[] AmplitudeFalloff = { 0.0f, 0.5f, 1.0f };

    // One bone chain per rigged child. A runtime LOD set (see SceneObjectManager.
    // SpawnLodLevels) parents several rigged meshes under one object, each with its
    // own SwayBone_0..2; every chain has to swing or the level the viewer actually
    // sees stands still while a hidden one animates.
    private List<Transform[]> _boneSets;
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
    public bool BonesFound => _boneSets != null && _boneSets.Count > 0;

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
        if (_boneSets == null)
        {
            var roots = new List<Transform>();
            FindAllRecursive(transform, BoneNames[0], roots);
            if (roots.Count == 0) return;
            _boneSets = new List<Transform[]>();
            foreach (var root in roots)
            {
                var found = new Transform[BoneNames.Length];
                found[0] = root;
                for (int i = 1; i < BoneNames.Length; i++)
                    found[i] = FindRecursive(root, BoneNames[i]);
                _boneSets.Add(found);
            }
            _bones = _boneSets[0];
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

        foreach (var bones in _boneSets)
        {
            for (int i = 0; i < bones.Length; i++)
            {
                if (bones[i] == null) continue;
                Vector3 localAxis = bones[i].InverseTransformDirection(axis);
                bones[i].localRotation = Quaternion.AngleAxis(swing * AmplitudeFalloff[i], localAxis);
            }
        }
    }

    private static void FindAllRecursive(Transform root, string name, List<Transform> into)
    {
        if (root.name == name) { into.Add(root); return; }
        foreach (Transform child in root)
            FindAllRecursive(child, name, into);
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
