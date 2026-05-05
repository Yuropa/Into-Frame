using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class AssetServer : MonoBehaviour
{
    [Header("Asset Server")]
    public string assetBaseUrl = "http://localhost:3000/assets/";

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public UnityWebRequest GetResource(string name)
    {
        string url = assetBaseUrl + name;
        Debug.Log($"[AssetServer] Downloading resource: {url}");
        return UnityWebRequest.Get(url);
    }

    public UnityWebRequest GetTexture(string textureId)
    {
        string url = assetBaseUrl + textureId;
        Debug.Log($"[AssetServer] Downloading texture: {url}");

        return UnityWebRequestTexture.GetTexture(url);
    }
}
