using System;
using UnityEngine;
using UnityEngine.Video;

/// <summary>
/// Plays an animated billboard's extracted clip -- a color video and a separate
/// grayscale matte video (see server VideoObjectExtractionStage's object_video_{i}
/// / object_video_alpha_{i}) -- and composites them into one RGBA texture each
/// frame via AlphaVideoComposite.shader, fed into the billboard's existing
/// material through the same MaterialPropertyBlock convention
/// SceneObjectManager.SetTexture already uses for the static-texture path.
/// </summary>
public class AnimatedBillboardVideo : MonoBehaviour
{
    private static Shader _compositeShader;

    private VideoPlayer _colorPlayer;
    private VideoPlayer _alphaPlayer;
    private RenderTexture _colorRT;
    private RenderTexture _alphaRT;
    private RenderTexture _combinedRT;
    private Material _compositeMaterial;
    private Renderer _renderer;
    private MaterialPropertyBlock _mpb;
    private bool _colorReady, _alphaReady;
    private Action _onFailed;
    private bool _failed;

    /// <param name="onFailed">
    /// Invoked at most once if either video fails to prepare/play (e.g. the asset
    /// server's extensionless URL -- same convention as mesh/texture assets -- isn't
    /// reliably sniffable by every platform's video backend, unlike GLTFast/
    /// UnityWebRequestTexture which parse downloaded bytes directly instead of
    /// handing a URL to a native streaming player). Caller should fall back to the
    /// billboard's static texture instead of leaving it blank.
    /// </param>
    public void Play(string assetBaseUrl, string colorKey, string alphaKey, Action onFailed = null)
    {
        _onFailed = onFailed;
        _renderer = GetComponentInChildren<Renderer>();
        if (_renderer == null)
        {
            Fail("no Renderer found in children");
            return;
        }

        if (_compositeShader == null)
            _compositeShader = Shader.Find("Hidden/IntoFrame/AlphaVideoComposite");
        if (_compositeShader == null)
        {
            Fail("composite shader 'Hidden/IntoFrame/AlphaVideoComposite' not found");
            return;
        }

        _mpb = new MaterialPropertyBlock();
        _compositeMaterial = new Material(_compositeShader);

        _colorPlayer = CreatePlayer(assetBaseUrl + colorKey);
        _alphaPlayer = CreatePlayer(assetBaseUrl + alphaKey);
        _colorPlayer.prepareCompleted += _ => TryStart();
        _alphaPlayer.prepareCompleted += _ => TryStart();
        _colorPlayer.errorReceived += (_, msg) => Fail($"color video '{colorKey}': {msg}");
        _alphaPlayer.errorReceived += (_, msg) => Fail($"alpha video '{alphaKey}': {msg}");
        _colorPlayer.Prepare();
        _alphaPlayer.Prepare();
    }

    private void Fail(string reason)
    {
        if (_failed) return; // only report/fall back once, even if both players error
        _failed = true;
        Debug.LogError($"[AnimatedBillboardVideo] Playback failed ({reason}) -- falling back to static texture");
        _onFailed?.Invoke();
        Destroy(this);
    }

    private VideoPlayer CreatePlayer(string url)
    {
        var player = gameObject.AddComponent<VideoPlayer>();
        player.playOnAwake = false;
        player.isLooping = true;
        player.renderMode = VideoRenderMode.RenderTexture;
        player.source = VideoSource.Url;
        player.url = url;
        return player;
    }

    private void TryStart()
    {
        // Both players fire prepareCompleted independently -- wait for both
        // before allocating RenderTextures sized from either one's frame size.
        if (_colorPlayer.isPrepared) _colorReady = true;
        if (_alphaPlayer.isPrepared) _alphaReady = true;
        if (!_colorReady || !_alphaReady || _combinedRT != null) return;

        int width = (int)_colorPlayer.width;
        int height = (int)_colorPlayer.height;
        if (width <= 0 || height <= 0) return;

        _colorRT = new RenderTexture(width, height, 0);
        _alphaRT = new RenderTexture(width, height, 0);
        _combinedRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);

        _colorPlayer.targetTexture = _colorRT;
        _alphaPlayer.targetTexture = _alphaRT;
        _colorPlayer.Play();
        _alphaPlayer.Play();
    }

    private void Update()
    {
        if (_combinedRT == null || _renderer == null) return;

        _compositeMaterial.SetTexture("_ColorTex", _colorRT);
        _compositeMaterial.SetTexture("_AlphaTex", _alphaRT);
        Graphics.Blit(null, _combinedRT, _compositeMaterial);

        _renderer.GetPropertyBlock(_mpb);
        _mpb.SetTexture("_BaseMap", _combinedRT);
        _renderer.SetPropertyBlock(_mpb);
    }

    private void OnDestroy()
    {
        // VideoPlayers were added to this same GameObject (not a child), so they
        // outlive this component unless torn down explicitly here.
        if (_colorPlayer != null) Destroy(_colorPlayer);
        if (_alphaPlayer != null) Destroy(_alphaPlayer);
        if (_colorRT != null) _colorRT.Release();
        if (_alphaRT != null) _alphaRT.Release();
        if (_combinedRT != null) _combinedRT.Release();
        if (_compositeMaterial != null) Destroy(_compositeMaterial);
    }
}
