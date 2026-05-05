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
        progressLabel.text = "Completed";
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