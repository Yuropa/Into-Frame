using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class PanoramaSkybox : MonoBehaviour
{
    [Header("Skybox Settings")]
    public string imageName;
    public float exposure = 1.0f;

    private Material skyboxMaterial;

    [Header("Assets")]
    public GameObject server;

    void Start()
    {
        if (!string.IsNullOrEmpty(imageName))
            LoadFromName(imageName);
    }

    public void LoadFromName(string name)
    {
        if (gameObject.activeInHierarchy)
            StartCoroutine(LoadAndApplySkybox(name));
        else
            StartCoroutine(WaitThenLoad(name));
    }

    private IEnumerator WaitThenLoad(string name)
    {
        yield return new WaitUntil(() => gameObject.activeInHierarchy);
        StartCoroutine(LoadAndApplySkybox(name));
    }

    private AssetServer _assetServer = null;
    private AssetServer assetServer()
    {
        if (_assetServer != null) {
            return _assetServer;
        }

        _assetServer = server.GetComponent<AssetServer>();
        return _assetServer;
    }

    private IEnumerator LoadAndApplySkybox(string name)
    {
        using (UnityWebRequest request = assetServer().GetTexture(name))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[Skybox] Failed to load: {request.error}");
                yield break;
            }

            Texture2D texture = DownloadHandlerTexture.GetContent(request);
            ApplySkybox(texture);
        }
    }

    private void ApplySkybox(Texture2D texture)
    {
        // Panorama textures need to wrap horizontally
        texture.wrapModeU = TextureWrapMode.Repeat;
        texture.wrapModeV = TextureWrapMode.Clamp;
        texture.Apply();

        // Create a Panoramic skybox material
        skyboxMaterial = new Material(Shader.Find("Skybox/Panoramic"));
        skyboxMaterial.SetTexture("_MainTex", texture);
        skyboxMaterial.SetFloat("_Exposure", exposure);

        // Apply to the scene
        RenderSettings.skybox = skyboxMaterial;

        // Force skybox to update
        DynamicGI.UpdateEnvironment();

        Debug.Log("[Skybox] Panorama applied successfully");
    }

    void OnDestroy()
    {
        if (skyboxMaterial != null)
            Destroy(skyboxMaterial);
    }
}
