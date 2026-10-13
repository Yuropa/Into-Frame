using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ProgressController : MonoBehaviour
{
    [Header("UI")]
    public TMPro.TextMeshProUGUI statusLabel;
    public Toggle statusToggle;
    public TMPro.TextMeshProUGUI progressLabel;
    public Slider progressSlider;
    public TMPro.TextMeshProUGUI spinnerLabel;

    private Coroutine _dotsCoroutine;

    // ── Public API ─────────────────────────────────────────────────────────

    public void SetConnected(bool connected, string label = null)
    {
        statusToggle.isOn = connected;
        statusLabel.text = label ?? (connected ? "Connected" : "Disconnected");
    }

    public void SetStatus(string label) => statusLabel.text = label;

    /// <summary>Server-side progress, mapped to 0–50%.</summary>
    public void ReportServerProgress(string step, float percent)
    {
        progressSlider.value = percent * 0.5f;
        progressLabel.text = step;
        SetProgressLabel();
    }

    /// <summary>Client-side loading progress, mapped to 50–100%.</summary>
    public void ReportSceneProgress(int completed, int total)
    {
        progressSlider.value = 0.5f + ((float)completed / total) * 0.5f;
        progressLabel.text = "Downloading Scene";
        SetProgressLabel();
    }

    public void ReportSceneComplete()
    {
        progressSlider.value = 1f;
        StopDots();
        progressLabel.text = _missingAssets > 0
            ? $"Completed ({_missingAssets} asset(s) failed to load)"
            : "Completed";
    }

    // Assets that failed to download or parse during the client-side load. Before
    // this the queue reported "Completed" whether or not anything was actually
    // missing, and the only trace of a failed GLB was a line in the Editor log.
    private int _missingAssets = 0;
    public void ReportAssetFailed(string assetId, string reason)
    {
        _missingAssets++;
        Debug.LogError($"[ProgressController] Asset '{assetId}' failed: {reason}");
    }
    public void ResetAssetFailures() => _missingAssets = 0;

    /// <summary>
    /// The server reported that the pipeline failed (or was cancelled). Freeze the
    /// bar, show the message, and offer a retry -- the server will not restart a
    /// failed run on its own, so without this the client would sit at the last
    /// reported stage indefinitely.
    /// </summary>
    public void ReportPipelineError(string message, bool retryable, Action onRetry)
    {
        StopDots();
        _onRetry = retryable ? onRetry : null;
        progressLabel.text = string.IsNullOrEmpty(message) ? "Pipeline failed" : $"Pipeline failed: {message}";
        if (retryButton != null)
        {
            retryButton.gameObject.SetActive(retryable);
            retryButton.onClick.RemoveAllListeners();
            if (retryable) retryButton.onClick.AddListener(() => Retry());
        }
        else if (retryable)
        {
            Debug.LogWarning("[ProgressController] No retryButton assigned; call Retry() (e.g. from the Inspector context menu) or restart the server.");
        }
    }

    [Header("Retry")]
    [Tooltip("Optional. Shown after a PIPELINE_ERROR; sends RETRY to the server.")]
    public Button retryButton;
    private Action _onRetry;

    [ContextMenu("Retry pipeline")]
    public void Retry()
    {
        if (_onRetry == null) return;
        var cb = _onRetry;
        _onRetry = null;
        if (retryButton != null) retryButton.gameObject.SetActive(false);
        cb();
    }

    // ── Dots Animation ─────────────────────────────────────────────────────

    private void SetProgressLabel()
    {
        if (_dotsCoroutine == null)
        {
            _dotsCoroutine = StartCoroutine(AnimateDots());
        }
    }

    private void StopDots()
    {
        if (_dotsCoroutine != null)
        {
            StopCoroutine(_dotsCoroutine);
            _dotsCoroutine = null;
            spinnerLabel.text = "";
        }
    }

    private IEnumerator AnimateDots()
    {
        string[] frames = { ".", "..", "..." };
        int i = 0;
        while (true)
        {
            spinnerLabel.text = frames[i % frames.Length];
            i++;
            yield return new WaitForSeconds(1.0f);
        }
    }

    private void OnDestroy() => StopDots();
}