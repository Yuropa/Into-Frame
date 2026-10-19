using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Grows the meadow from the server's grass field (server/pipeline/grass_cover/
/// grass_field.py) as GPU-instanced blade tufts, replacing the thousands of
/// crossed-card tuft GameObjects Grass Cover used to place.
///
/// Lives on the terrain's container: the field is authored in the terrain grid's
/// own frame (+X grid columns, +Z grid rows, centred on the capture point), which is
/// this container's local frame. Each tuft is dropped onto the terrain's own
/// MeshCollider, so the grass stands on the ground it is drawn on; that has to wait
/// until the scene is revealed, because colliders under an inactive sceneRoot are
/// disabled.
///
/// Density image: R = density 0..1, G = height as a fraction of maxHeightMeters.
/// Colour image: the photograph's delit ground colour, used as the blade colour.
/// </summary>
public class GrassField : MonoBehaviour
{
    const int BatchSize = 1023;
    const float TileMeters = 6f;

    // Beyond this the 3-blade tuft is drawn instead of the full one, and tufts are
    // thinned as (nearRadius / distance), so the budget stays near the viewer.
    public float nearRadius = 10f;
    // Beyond this each tuft is one camera-facing quad cut into blades by the
    // shader (GrassBlades _Card): two triangles instead of a tuft's dozens. A card
    // is wider than a tuft, so cards are thinned by cardKeep on top of the falloff.
    public float cardRadius = 25f;
    public float cardWidth = 1.5f;   // x the tuft height: a 0.25 m tuft card is 0.38 m wide
    [Range(0f, 1f)] public float cardKeep = 0.35f;
    public int maxTufts = 60000;

    GrassFieldData _data;
    Texture2D _density, _colour;
    Material _material, _cardMaterial;
    Mesh _nearMesh, _farMesh, _cardMesh;
    bool _built;

    class Batch
    {
        public Matrix4x4[] local;      // relative to this (terrain) transform
        public Matrix4x4[] matrices;   // world, refreshed when the transform moves
        public int count;
        public MaterialPropertyBlock props;
        public Bounds bounds;
        public int tier;               // 0 full tuft, 1 3-blade tuft, 2 card
    }
    readonly List<Batch> _batches = new List<Batch>();

    public void Init(GrassFieldData data, Texture2D density, Texture2D colour, Material material)
    {
        _data = data;
        _density = density;
        _colour = colour;
        _material = material;
        // The tuft is authored at unit height and scaled uniformly by each tuft's
        // height, so blade width is given relative to a nominal 0.4 m tuft.
        float unitWidth = data.bladeWidthMeters / 0.4f;
        _nearMesh = BuildTuftMesh(Mathf.Max(1, data.bladesPerTuft), unitWidth, 11);
        _farMesh = BuildTuftMesh(Mathf.Max(1, Mathf.Min(3, data.bladesPerTuft)), unitWidth * 1.6f, 23);
        _cardMesh = BuildCardMesh(cardWidth);
        _cardMaterial = new Material(material) { name = material.name + " (card)" };
        _cardMaterial.SetFloat("_Card", 1f);
    }

    void OnDestroy()
    {
        if (_cardMaterial != null) Destroy(_cardMaterial);
    }

    void Update()
    {
        if (_data == null || _material == null) return;
        if (!_built)
        {
            var terrainCollider = GetComponentInChildren<MeshCollider>();
            if (terrainCollider == null || !terrainCollider.enabled || !gameObject.activeInHierarchy) return;
            Build(terrainCollider);
            _built = true;
            transform.hasChanged = true;
        }
        if (transform.hasChanged)
        {
            RefreshWorld();
            transform.hasChanged = false;
        }
        foreach (var b in _batches)
        {
            var rp = new RenderParams(b.tier == 2 ? _cardMaterial : _material)
            {
                worldBounds = b.bounds,
                matProps = b.props,
                shadowCastingMode = ShadowCastingMode.Off,
                receiveShadows = true,
            };
            Graphics.RenderMeshInstanced(rp, b.tier == 0 ? _nearMesh : b.tier == 1 ? _farMesh : _cardMesh, 0, b.matrices, b.count);
        }
    }

    void Build(MeshCollider terrainCollider)
    {
        float size = Mathf.Max(_data.sizeMeters, 1f);
        float half = size / 2f;
        float spacing = 1f / Mathf.Sqrt(Mathf.Max(_data.tuftsPerSquareMeter, 0.01f));
        int n = Mathf.CeilToInt(size / spacing);
        var rng = new System.Random(12345);
        var dens = _density.GetPixels32();
        var cols = _colour.GetPixels32();
        int dw = _density.width, dh = _density.height;
        int cw = _colour.width, ch = _colour.height;

        // tile key -> (matrices, colours) per tier
        var tiles = new Dictionary<long, (List<Matrix4x4> m, List<Vector4> c, int tier)>();
        int placed = 0;
        Vector3 up = transform.up;
        // Candidates first, nearest kept when over budget: filling the lattice row
        // by row stopped at maxTufts wherever the rows had reached, so a dense
        // meadow lost its far side -- the side the photograph looks at.
        var candidates = new List<Vector3>();
        for (int iz = 0; iz < n; iz++)
        for (int ix = 0; ix < n; ix++)
        {
            float x = -half + (ix + (float)rng.NextDouble()) * spacing;
            float z = -half + (iz + (float)rng.NextDouble()) * spacing;
            float r = Mathf.Sqrt(x * x + z * z);
            if (r > half) continue;
            // Image row 0 is z = -half (grass_field.py); Unity textures start at the
            // bottom row, so a PNG's first row is the texture's last.
            float u = (x + half) / size, v = 1f - (z + half) / size;
            float density = Sample(dens, dw, dh, u, v).r / 255f;
            float keep = density;
            if (r > nearRadius) keep *= nearRadius / r;
            if (r > cardRadius) keep *= cardKeep;
            if (rng.NextDouble() >= keep) continue;
            candidates.Add(new Vector3(x, z, r));
        }
        if (candidates.Count > maxTufts)
            candidates.Sort((a, b) => a.z.CompareTo(b.z));
        foreach (var cand in candidates)
        {
            if (placed >= maxTufts) break;
            float x = cand.x, z = cand.y, r = cand.z;
            float u = (x + half) / size, v = 1f - (z + half) / size;
            Color32 d = Sample(dens, dw, dh, u, v);

            Vector3 above = transform.TransformPoint(new Vector3(x, 0f, z)) + up * 200f;
            if (!terrainCollider.Raycast(new Ray(above, -up), out RaycastHit hit, 1000f)) continue;

            float height = Mathf.Max(0.04f, _data.maxHeightMeters * d.g / 255f * (0.8f + 0.4f * (float)rng.NextDouble()));
            float yaw = (float)rng.NextDouble() * 360f;
            // Local to this transform, so the field follows if sceneRoot moves.
            var m = Matrix4x4.TRS(transform.InverseTransformPoint(hit.point), Quaternion.Euler(0f, yaw, 0f), Vector3.one * height);
            Color32 c = Sample(cols, cw, ch, u, v);
            // Small per-tuft variation so neighbouring tufts are not one flat colour.
            float jitter = 0.85f + 0.3f * (float)rng.NextDouble();
            Color lin = new Color(c.r / 255f * jitter, c.g / 255f * jitter, c.b / 255f * jitter).linear;

            int tier = r <= nearRadius ? 0 : r <= cardRadius ? 1 : 2;
            long key = ((long)Mathf.FloorToInt((x + half) / TileMeters) << 34)
                     | ((long)Mathf.FloorToInt((z + half) / TileMeters) << 2) | (long)tier;
            if (!tiles.TryGetValue(key, out var t))
            {
                t = (new List<Matrix4x4>(), new List<Vector4>(), tier);
                tiles[key] = t;
            }
            t.m.Add(m);
            t.c.Add(new Vector4(lin.r, lin.g, lin.b, 1f));
            placed++;
        }

        foreach (var t in tiles.Values)
        {
            for (int start = 0; start < t.m.Count; start += BatchSize)
            {
                int count = Mathf.Min(BatchSize, t.m.Count - start);
                var b = new Batch
                {
                    local = new Matrix4x4[count],
                    matrices = new Matrix4x4[count],
                    count = count,
                    props = new MaterialPropertyBlock(),
                    tier = t.tier,
                };
                var colours = new Vector4[count];
                for (int k = 0; k < count; k++)
                {
                    b.local[k] = t.m[start + k];
                    colours[k] = t.c[start + k];
                }
                b.props.SetVectorArray("_GrassColor", colours);
                _batches.Add(b);
            }
        }
        int cards = 0;
        foreach (var b in _batches) if (b.tier == 2) cards += b.count;
        Debug.Log($"[GrassField] {placed} tufts ({cards} as cards beyond {cardRadius:F0} m) in {_batches.Count} batch(es) over a {size:F0} m field");
    }

    void RefreshWorld()
    {
        var toWorld = transform.localToWorldMatrix;
        foreach (var b in _batches)
        {
            Vector3 min = Vector3.positiveInfinity, max = Vector3.negativeInfinity;
            for (int k = 0; k < b.count; k++)
            {
                b.matrices[k] = toWorld * b.local[k];
                Vector3 p = b.matrices[k].GetColumn(3);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
            b.bounds = new Bounds((min + max) / 2f, (max - min) + Vector3.one * (_data.maxHeightMeters * 2f + 1f));
        }
    }

    static Color32 Sample(Color32[] px, int w, int h, float u, float v)
    {
        int x = Mathf.Clamp(Mathf.FloorToInt(u * w), 0, w - 1);
        int y = Mathf.Clamp(Mathf.FloorToInt(v * h), 0, h - 1);
        return px[y * w + x];
    }

    /// <summary>
    /// The far tier's card: one upright quad, unit height, `width` wide, rooted at
    /// the origin. GrassBlades turns it to the camera and cuts the blades.
    /// </summary>
    static Mesh BuildCardMesh(float width)
    {
        float hw = width / 2f;
        var mesh = new Mesh { name = "GrassCard" };
        mesh.SetVertices(new List<Vector3> { new Vector3(-hw, 0f, 0f), new Vector3(hw, 0f, 0f), new Vector3(-hw, 1f, 0f), new Vector3(hw, 1f, 0f) });
        mesh.SetNormals(new List<Vector3> { Vector3.back, Vector3.back, Vector3.back, Vector3.back });
        mesh.SetUVs(0, new List<Vector2> { new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0f, 1f), new Vector2(1f, 1f) });
        mesh.SetTriangles(new[] { 0, 2, 1, 1, 2, 3 }, 0);
        // Turned to the camera in the vertex shader: bounds must hold any yaw.
        mesh.bounds = new Bounds(new Vector3(0f, 0.5f, 0f), new Vector3(width, 1f, width));
        return mesh;
    }

    /// <summary>
    /// One tuft, unit height: blades fanned around the root, each a tapering,
    /// curving strip with UV.y = 0 at the root and 1 at the tip.
    /// </summary>
    static Mesh BuildTuftMesh(int blades, float width, int seed)
    {
        const int segments = 4;
        var rng = new System.Random(seed);
        var verts = new List<Vector3>();
        var norms = new List<Vector3>();
        var uvs = new List<Vector2>();
        var tris = new List<int>();
        for (int b = 0; b < blades; b++)
        {
            float angle = (float)(rng.NextDouble() * Mathf.PI * 2f);
            Vector3 face = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
            Vector3 side = Vector3.Cross(Vector3.up, face).normalized;
            Vector3 root = face * (float)(rng.NextDouble() * 0.08);
            float lean = 0.1f + 0.35f * (float)rng.NextDouble();
            float bladeH = 0.6f + 0.4f * (float)rng.NextDouble();
            float w = width * (0.7f + 0.6f * (float)rng.NextDouble());
            int baseIndex = verts.Count;
            for (int s = 0; s <= segments; s++)
            {
                float t = s / (float)segments;
                // Curves outward with height: quadratic lean.
                Vector3 centre = root + Vector3.up * (t * bladeH) + face * (lean * t * t * bladeH);
                float halfW = 0.5f * w * (1f - 0.85f * t);
                Vector3 normal = (Vector3.up * 0.3f - face).normalized;
                verts.Add(centre - side * halfW); norms.Add(normal); uvs.Add(new Vector2(0f, t * bladeH));
                verts.Add(centre + side * halfW); norms.Add(normal); uvs.Add(new Vector2(1f, t * bladeH));
            }
            for (int s = 0; s < segments; s++)
            {
                int i0 = baseIndex + s * 2;
                tris.Add(i0); tris.Add(i0 + 2); tris.Add(i0 + 1);
                tris.Add(i0 + 1); tris.Add(i0 + 2); tris.Add(i0 + 3);
            }
        }
        var mesh = new Mesh { name = $"GrassTuft{blades}" };
        mesh.SetVertices(verts);
        mesh.SetNormals(norms);
        mesh.SetUVs(0, uvs);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateBounds();
        return mesh;
    }
}
