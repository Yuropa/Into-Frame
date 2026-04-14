using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Allows code running on background threads (e.g. WebSocket receive callbacks)
/// to safely enqueue work onto Unity's main thread.
///
/// Usage:
///   UnityMainThread.Call(() => { /* Unity API calls here */ });
///
/// Attach this script to a persistent GameObject (e.g. your SceneManager).
/// </summary>
public class UnityMainThread : MonoBehaviour
{
    private static UnityMainThread _instance;
    private readonly Queue<Action> _queue = new();
    private readonly object _lock = new();

    private void Awake()
    {
        if (_instance != null) { Destroy(gameObject); return; }
        _instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Update()
    {
        lock (_lock)
        {
            while (_queue.Count > 0)
                _queue.Dequeue()?.Invoke();
        }
    }

    public static void Call(Action action)
    {
        if (_instance == null)
        {
            Debug.LogError("[UnityMainThread] No instance found. Add UnityMainThread to a scene GameObject.");
            return;
        }
        lock (_instance._lock)
            _instance._queue.Enqueue(action);
    }
}
