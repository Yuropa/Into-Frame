using UnityEngine;

/// <summary>
/// Shared procedural wind: one direction (matching the server's per-scene
/// wind_axis_degrees -- see SceneAnimationStage) plus discrete gust events that
/// sweep across the scene every gustInterval seconds, rather than every object
/// running its own private, always-on oscillator. Without this, every WindSway
/// instance answers only to its own server-randomized phase/frequency and the
/// scene reads as continuously, independently shivering objects instead of a
/// calm scene occasionally moved by a real gust of wind.
///
/// A gust is a single pulse in time, delayed at each point by how long the wind
/// takes to physically reach it (worldPos projected onto the wind direction,
/// divided by gustSpeed). That delay is the whole mechanism: it's what makes a
/// gust visibly start on one side of the scene and sweep across to the other,
/// rather than every point flaring up in lockstep.
///
/// Purely procedural for now -- the intent is to later derive direction/timing
/// from the source image instead, but every consumer (WindSway,
/// WaterSurface.shader) samples through this one API, so that swap won't touch
/// them.
///
/// Lazily self-instantiates (see Instance) so nothing needs to add or wire this
/// into a scene -- the first consumer that asks for it gets one.
/// </summary>
public class WindField : MonoBehaviour
{
    private static WindField _instance;

    public static WindField Instance
    {
        get
        {
            if (_instance == null)
            {
                var go = new GameObject("[wind field]");
                _instance = go.AddComponent<WindField>();
            }
            return _instance;
        }
    }

    [Header("Direction")]
    [Tooltip("Degrees around Y, same convention as SwayData.axisDegrees. Set once " +
             "from the first sway-carrying object's axisDegrees (see WindSway.Apply) " +
             "so gust travel direction matches the direction objects actually lean.")]
    public float windDirectionDegrees = 0f;

    [Header("Ambient")]
    [Tooltip("Sway strength multiplier between gusts. Deliberately low (not zero) " +
             "so the scene has a faint idle motion rather than freezing solid, but " +
             "reads as calm compared to a gust passing through.")]
    [Range(0f, 1f)] public float baseStrength = 0.12f;

    [Header("Gust events")]
    [Tooltip("Extra strength multiplier at a gust's peak, added on top of baseStrength.")]
    public float gustStrength = 1.2f;

    [Tooltip("Seconds between the start of one gust and the next, at a fixed point.")]
    public float gustInterval = 10f;

    [Tooltip("How long, in seconds, the gust takes to pass over any single point -- " +
             "i.e. how long that point's wind stays elevated once the gust reaches it.")]
    public float gustDuration = 3f;

    [Tooltip("Speed the gust front travels across the scene, in meters/second -- also " +
             "sets how many seconds apart two points are: distance / gustSpeed.")]
    public float gustSpeed = 8f;

    [Tooltip("Higher = a punchier, more sudden rise/fall within the gust's duration. " +
             "Lower = a broad, gentle swell.")]
    [Range(0.5f, 8f)] public float gustSharpness = 2f;

    private static readonly int _windDirId       = Shader.PropertyToID("_WindDirXZ");
    private static readonly int _windBaseId      = Shader.PropertyToID("_WindBaseStrength");
    private static readonly int _windGustStrId   = Shader.PropertyToID("_WindGustStrength");
    private static readonly int _windIntervalId  = Shader.PropertyToID("_WindGustInterval");
    private static readonly int _windDurationId  = Shader.PropertyToID("_WindGustDuration");
    private static readonly int _windSpeedId     = Shader.PropertyToID("_WindGustSpeed");
    private static readonly int _windSharpId     = Shader.PropertyToID("_WindGustSharpness");

    private Vector2 _windDirXZ = Vector2.up;

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
        DontDestroyOnLoad(gameObject);
        SetDirection(windDirectionDegrees);
    }

    private void Update()
    {
        // Pushed as globals (not a per-material property) so any shader in the scene
        // -- water, foliage, anything added later -- can opt in just by declaring
        // these uniforms, without SceneObjectManager having to know about it.
        Shader.SetGlobalVector(_windDirId, new Vector4(_windDirXZ.x, _windDirXZ.y, 0f, 0f));
        Shader.SetGlobalFloat(_windBaseId, baseStrength);
        Shader.SetGlobalFloat(_windGustStrId, gustStrength);
        Shader.SetGlobalFloat(_windIntervalId, gustInterval);
        Shader.SetGlobalFloat(_windDurationId, gustDuration);
        Shader.SetGlobalFloat(_windSpeedId, gustSpeed);
        Shader.SetGlobalFloat(_windSharpId, gustSharpness);
    }

    public Vector2 DirectionXZ => _windDirXZ;

    public void SetDirection(float degrees)
    {
        windDirectionDegrees = degrees;
        float rad = degrees * Mathf.Deg2Rad;
        _windDirXZ = new Vector2(Mathf.Sin(rad), Mathf.Cos(rad));
    }

    /// <summary>
    /// How many seconds the wind takes to physically reach a world position, relative
    /// to the plane where travel == 0 along the wind direction. WindSway uses this to
    /// delay BOTH a point's gust envelope and its own oscillation clock, so motion
    /// itself -- not just the amount of it -- visibly propagates across the scene.
    /// </summary>
    public float SampleDelaySeconds(Vector3 worldPos)
    {
        float travel = Vector2.Dot(new Vector2(worldPos.x, worldPos.z), _windDirXZ);
        return travel / Mathf.Max(gustSpeed, 0.01f);
    }

    /// <summary>
    /// Wind strength multiplier at a world position and time: baseStrength between
    /// gusts, rising to baseStrength + gustStrength as a gust passes over. Gusts recur
    /// every gustInterval seconds at a fixed point, each lasting gustDuration seconds,
    /// delayed per-point by SampleDelaySeconds -- so at any instant only a band of the
    /// scene (wherever the current gust front currently is) is actually gusting, not
    /// the whole scene at once and not continuously.
    /// </summary>
    public float SampleStrength(Vector3 worldPos, float time)
    {
        return baseStrength + gustStrength * GustPulse(SampleDelaySeconds(worldPos), time);
    }

    private float GustPulse(float delaySeconds, float time)
    {
        float period = Mathf.Max(gustInterval, 0.01f);
        float u = time - delaySeconds;
        float phase = u - Mathf.Floor(u / period) * period; // wrapped into [0, period)
        float duration = Mathf.Clamp(gustDuration, 0.01f, period);
        if (phase >= duration) return 0f;
        // Single smooth lobe across the gust's duration: 0 -> peak -> 0, not a
        // symmetric sine train, so there is exactly one rise-and-fall per gust event
        // rather than the repeating ripple a raw sin() would give.
        return Mathf.Pow(Mathf.Sin(phase / duration * Mathf.PI), gustSharpness);
    }
}
