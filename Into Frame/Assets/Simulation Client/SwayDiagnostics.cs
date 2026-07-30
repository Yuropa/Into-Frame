using System.Linq;
using System.Text;
using UnityEngine;

/// <summary>
/// Runtime report on why the scene isn't visibly moving.
///
/// The sway path fails silently at four different points and every one of them looks
/// identical from inside the headset -- nothing moves. This walks the chain and says
/// which link is broken:
///
///   1. no WindSway components          -> SceneObjectManager never attached them
///                                          (check HasSway / the server payload)
///   2. components but HasData false    -> Apply() was never called
///   3. HasData but BonesFound false    -> the served GLB has no SwayBone_* chain,
///                                          so LateUpdate returns every frame
///   4. BonesFound and PeakSwing tiny   -> it IS being driven, just imperceptibly;
///                                          raise DebugAmplitudeScale to confirm
///   5. all of the above healthy        -> bones are moving and the mesh is not:
///                                          skinning/culling, not sway
///
/// Point 5 is the one worth the extra work: it measures a sample bone's actual
/// rotation delta between frames, so "the transform is animating but the vertices
/// aren't following" is distinguishable from "nothing is animating at all".
///
/// Attach to any GameObject in the scene. Costs one FindObjectsOfType per interval,
/// which is why the default interval is slow and it is trivially disabled.
/// </summary>
public class SwayDiagnostics : MonoBehaviour
{
    [Tooltip("Seconds between reports. FindObjectsOfType is not cheap at 6k+ objects.")]
    public float reportIntervalSeconds = 3f;

    [Range(0f, 20f)]
    [Tooltip("Multiplier applied to every WindSway instance. 1 = as authored. Try 10 " +
             "to see whether the motion exists but is too small to notice. Pushed to " +
             "WindSway every frame, so dragging this while in Play mode takes effect live.")]
    public float amplitudeScale = 1f;

    [Tooltip("Log a per-asset breakdown as well as the scene total.")]
    public bool breakdownByAsset = true;

    private float _nextReport;
    private Transform _sampleBone;
    private Quaternion _lastSampleRotation;
    private float _maxObservedBoneDelta;

    private void Update()
    {
        // Pushed every frame so the Inspector slider takes effect live while paused-
        // and-resumed, which is the whole point of having it.
        WindSway.DebugAmplitudeScale = amplitudeScale;

        if (_sampleBone != null)
        {
            float delta = Quaternion.Angle(_lastSampleRotation, _sampleBone.localRotation);
            _maxObservedBoneDelta = Mathf.Max(_maxObservedBoneDelta, delta);
            _lastSampleRotation = _sampleBone.localRotation;
        }

        if (Time.unscaledTime < _nextReport) return;
        _nextReport = Time.unscaledTime + Mathf.Max(0.5f, reportIntervalSeconds);
        Report();
    }

    private void Report()
    {
        // Include inactive: SceneObjectManager keeps sceneRoot deactivated until its
        // load queue drains, so every spawned object is inactive for the whole load.
        // Excluding them (the default) would report "no WindSway components" for
        // minutes and then abruptly start working, which reads as a real failure.
        var sways = FindObjectsByType<WindSway>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (sways.Length == 0)
        {
            Debug.LogWarning("[SwayDiagnostics] No WindSway components in the scene at all. " +
                             "SceneObjectManager never attached them — check that scene.json " +
                             "objects carry sway, and that HasSway() accepts it.");
            return;
        }

        // An inactive component's LateUpdate never runs, so it cannot have been driven
        // yet. Counting those as failures would blame the rig for what is just load
        // ordering -- report them separately instead.
        int inactive = sways.Count(s => !s.isActiveAndEnabled);

        int withData = sways.Count(s => s.HasData);
        int withBones = sways.Count(s => s.BonesFound);
        var driven = sways.Where(s => s.BonesFound).ToArray();
        float peak = driven.Length > 0 ? driven.Max(s => s.PeakSwing) : 0f;
        float median = 0f;
        if (driven.Length > 0)
        {
            var peaks = driven.Select(s => s.PeakSwing).OrderBy(v => v).ToArray();
            median = peaks[peaks.Length / 2];
        }

        if (_sampleBone == null && driven.Length > 0)
        {
            // Latch onto one real bone so the next interval can report whether the
            // transform is genuinely changing, independent of what WindSway claims.
            _sampleBone = FindBone(driven[0].transform, "SwayBone_2");
            if (_sampleBone != null) _lastSampleRotation = _sampleBone.localRotation;
        }

        var sb = new StringBuilder();
        sb.AppendLine($"[SwayDiagnostics] {sways.Length} WindSway | data {withData} | bones {withBones} " +
                      $"| inactive {inactive} | amplitudeScale {amplitudeScale:0.##} " +
                      $"| Time.timeScale {Time.timeScale:0.##}");
        if (inactive == sways.Length)
            sb.AppendLine("  (all inactive — scene still loading, sceneRoot hidden. Not a fault.)");
        sb.AppendLine($"  swing degrees: median peak {median:0.00}, max peak {peak:0.00}");
        sb.AppendLine($"  sample bone rotation delta between frames: max {_maxObservedBoneDelta:0.000} deg" +
                      (_sampleBone == null ? "  (no sample bone latched)" : ""));

        if (withData < sways.Length)
            sb.AppendLine($"  !! {sways.Length - withData} component(s) never got Apply() — no sway data.");
        // Only meaningful for components that have actually had a frame to look: bone
        // discovery happens in LateUpdate, which never runs while sceneRoot is hidden.
        int activeWithoutBones = sways.Count(s => s.isActiveAndEnabled && s.HasData && !s.BonesFound);
        if (activeWithoutBones > 0)
            sb.AppendLine($"  !! {activeWithoutBones} ACTIVE component(s) cannot find SwayBone_0 — the served " +
                          $"GLB is unrigged, so these silently never move. Run scripts/check-rigged.py.");
        if (withBones > 0 && peak < 0.05f)
            sb.AppendLine("  !! bones found but swing is ~0 — amplitude or frequencyHz is zero in the payload.");
        if (withBones > 0 && peak > 1f && _maxObservedBoneDelta < 0.001f)
            sb.AppendLine("  !! WindSway reports a swing but the bone transform never changes — something " +
                          "else is overwriting localRotation, or these are not the bones the " +
                          "SkinnedMeshRenderer is bound to.");
        if (withBones > 0 && _maxObservedBoneDelta > 0.01f && peak > 1f)
            sb.AppendLine("  -> bones ARE animating. If nothing looks like it is moving, the problem is " +
                          "downstream: skinning, renderer bounds/culling, or the motion is genuinely " +
                          "too subtle at this amplitude (raise amplitudeScale to check).");

        if (breakdownByAsset)
        {
            var groups = sways.GroupBy(s => AssetName(s.transform))
                              .OrderByDescending(g => g.Count())
                              .Take(8);
            foreach (var g in groups)
                sb.AppendLine($"    {g.Count(),6}  {g.Key,-38} bones {g.Count(s => s.BonesFound)}  " +
                              $"peak {(g.Any(s => s.BonesFound) ? g.Where(s => s.BonesFound).Max(s => s.PeakSwing) : 0f):0.00}");
        }

        Debug.Log(sb.ToString());
    }

    // SceneObjectManager names containers "[mesh] <id> (<assetName>)".
    private static string AssetName(Transform t)
    {
        string n = t.name;
        int open = n.LastIndexOf('(');
        int close = n.LastIndexOf(')');
        return (open >= 0 && close > open) ? n.Substring(open + 1, close - open - 1) : n;
    }

    private static Transform FindBone(Transform root, string name)
    {
        if (root.name == name) return root;
        foreach (Transform child in root)
        {
            var found = FindBone(child, name);
            if (found != null) return found;
        }
        return null;
    }
}
