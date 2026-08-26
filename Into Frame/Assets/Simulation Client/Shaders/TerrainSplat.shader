Shader "IntoFrame/TerrainSplat"
{
    Properties
    {
        // Blend maps — RGBA weight maps sampled at terrain UV (0→1)
        _BlendMap0  ("Blend Map 0",  2D) = "black" {}
        _BlendMap1  ("Blend Map 1",  2D) = "black" {}

        // Per-layer tileable textures (up to 8)
        _Layer0Tile ("Layer 0 Tile", 2D) = "white" {}
        _Layer1Tile ("Layer 1 Tile", 2D) = "white" {}
        _Layer2Tile ("Layer 2 Tile", 2D) = "white" {}
        _Layer3Tile ("Layer 3 Tile", 2D) = "white" {}
        _Layer4Tile ("Layer 4 Tile", 2D) = "white" {}
        _Layer5Tile ("Layer 5 Tile", 2D) = "white" {}
        _Layer6Tile ("Layer 6 Tile", 2D) = "white" {}
        _Layer7Tile ("Layer 7 Tile", 2D) = "white" {}

        // Per-layer tile repeat factors (how many times the tile repeats across the terrain).
        // Layer 0 is the panorama layer (equirect, always 1.0); layers 1-7 are synthetic
        // region tiles set at runtime via MaterialPropertyBlock from tile_factor on the server.
        // Default 50.0 = 4 m/tile over a 200 m grid (~0.4 cm/texel at 1024 px).
        _Layer0TileRepeat ("Layer 0 Tile Repeat", Float) = 1.0
        _Layer1TileRepeat ("Layer 1 Tile Repeat", Float) = 50.0
        _Layer2TileRepeat ("Layer 2 Tile Repeat", Float) = 50.0
        _Layer3TileRepeat ("Layer 3 Tile Repeat", Float) = 50.0
        _Layer4TileRepeat ("Layer 4 Tile Repeat", Float) = 50.0
        _Layer5TileRepeat ("Layer 5 Tile Repeat", Float) = 50.0
        _Layer6TileRepeat ("Layer 6 Tile Repeat", Float) = 50.0
        _Layer7TileRepeat ("Layer 7 Tile Repeat", Float) = 50.0

        // Per-layer PBR smoothness (0 = perfectly matte, 1 = mirror-smooth).
        // Water should be ~0.88; soil/grass/gravel near 0.
        _Layer0Smoothness ("Layer 0 Smoothness", Range(0,1)) = 0.1
        _Layer1Smoothness ("Layer 1 Smoothness", Range(0,1)) = 0.1
        _Layer2Smoothness ("Layer 2 Smoothness", Range(0,1)) = 0.1
        _Layer3Smoothness ("Layer 3 Smoothness", Range(0,1)) = 0.1
        _Layer4Smoothness ("Layer 4 Smoothness", Range(0,1)) = 0.1
        _Layer5Smoothness ("Layer 5 Smoothness", Range(0,1)) = 0.1
        _Layer6Smoothness ("Layer 6 Smoothness", Range(0,1)) = 0.1
        _Layer7Smoothness ("Layer 7 Smoothness", Range(0,1)) = 0.1

        // Bitmask: bit i set means layer i uses equirectangular UV from world position
        // rather than planar tiled UV. Used for the panorama layer (typically layer 0).
        _EquirectLayers ("Equirect Layers Bitmask", Int) = 0

        // Object-space XZ extent of the terrain grid: (minX, minZ, maxX, maxZ).
        // Set per-renderer from the mesh's own bounds (see TerrainMaterialManager),
        // and used to derive the top-down 0→1 grid UV the blend maps and every
        // planar-tiled layer are authored against.
        //
        // That UV is NOT read from the mesh's TEXCOORD0 any more. TerrainMeshStage
        // runs before TerrainTextureGenerationStage, so no SplatMaterial or baked
        // texture exists yet when TerrainMeshGenerator.generate() picks a UV0 --
        // it falls through to its `elif panorama is not None` branch and writes the
        // equirectangular panorama UV into TEXCOORD0, identical to TEXCOORD1.
        // (Verified on an exported terrain_mesh.glb: the two accessors are
        // byte-identical.) Sampling a top-down blend map with azimuth/elevation
        // coordinates confined to v ∈ [0.29, 0.54] reads layer weights from
        // unrelated world positions and tiles synthetic layers in equirect space.
        //
        // Deriving it here rather than fixing the mesh export keeps the glTF format
        // unchanged: UV0 holding the panorama UV is what lets a plain glTF viewer
        // sample the embedded panorama preview through the mesh's only texture
        // coordinate set. That constraint used to be hard -- the since-removed
        // visionOS renderer bound exactly one UV set -- and is now merely a
        // compatibility preference, so writing a top-down UV0 server-side (and
        // dropping this derivation) is an option if the export is ever revisited.
        _TerrainExtent ("Terrain XZ Extent (minX, minZ, maxX, maxZ)", Vector) = (-100, -100, 100, 100)

        // Height-biased blending: each layer tile's alpha channel carries a local
        // micro-height map (see terrain_texture_generation.py). Blend-map weights are
        // biased by that local height before normalising, so e.g. grass pokes through
        // the cracks between dirt clumps instead of a flat linear cross-fade.
        // 0 = disabled (identical to the old plain-weighted blend); higher = crisper,
        // more height-driven transitions.
        _HeightBlendSharpness ("Height Blend Sharpness", Range(0, 16)) = 4.0
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"          = "Geometry"
        }
        LOD 300

        // ── Forward Lit ────────────────────────────────────────────────────
        Pass
        {
            Name "UniversalForward"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex   Vert
            #pragma fragment Frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            #pragma multi_compile _ SHADOWS_SHADOWMASK
            #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            #pragma multi_compile _ LIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // ── Textures ───────────────────────────────────────────────────
            TEXTURE2D(_BlendMap0);  SAMPLER(sampler_BlendMap0);
            TEXTURE2D(_BlendMap1);  SAMPLER(sampler_BlendMap1);
            TEXTURE2D(_Layer0Tile); SAMPLER(sampler_Layer0Tile);
            TEXTURE2D(_Layer1Tile); SAMPLER(sampler_Layer1Tile);
            TEXTURE2D(_Layer2Tile); SAMPLER(sampler_Layer2Tile);
            TEXTURE2D(_Layer3Tile); SAMPLER(sampler_Layer3Tile);
            TEXTURE2D(_Layer4Tile); SAMPLER(sampler_Layer4Tile);
            TEXTURE2D(_Layer5Tile); SAMPLER(sampler_Layer5Tile);
            TEXTURE2D(_Layer6Tile); SAMPLER(sampler_Layer6Tile);
            TEXTURE2D(_Layer7Tile); SAMPLER(sampler_Layer7Tile);

            // ── Material CBUFFER ───────────────────────────────────────────
            CBUFFER_START(UnityPerMaterial)
                float4 _BlendMap0_ST;
                float4 _BlendMap1_ST;
                float4 _Layer0Tile_ST;
                float4 _Layer1Tile_ST;
                float4 _Layer2Tile_ST;
                float4 _Layer3Tile_ST;
                float4 _Layer4Tile_ST;
                float4 _Layer5Tile_ST;
                float4 _Layer6Tile_ST;
                float4 _Layer7Tile_ST;
                float  _Layer0TileRepeat;
                float  _Layer1TileRepeat;
                float  _Layer2TileRepeat;
                float  _Layer3TileRepeat;
                float  _Layer4TileRepeat;
                float  _Layer5TileRepeat;
                float  _Layer6TileRepeat;
                float  _Layer7TileRepeat;
                float  _Layer0Smoothness;
                float  _Layer1Smoothness;
                float  _Layer2Smoothness;
                float  _Layer3Smoothness;
                float  _Layer4Smoothness;
                float  _Layer5Smoothness;
                float  _Layer6Smoothness;
                float  _Layer7Smoothness;
                int    _EquirectLayers;
                float  _HeightBlendSharpness;
                float4 _TerrainExtent;
            CBUFFER_END

            // ── Structs ────────────────────────────────────────────────────
            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                // Deliberately unused: on the exported terrain mesh TEXCOORD0 is a
                // duplicate of the panorama UV, not the top-down grid UV the blend
                // maps and planar layers need. Vert derives that from positionOS
                // instead -- see _TerrainExtent. Kept in the layout so the vertex
                // stream still matches meshes that carry it.
                float2 uv         : TEXCOORD0;
                // Baked panorama UV (server-side TerrainMeshGenerator.generate's
                // panorama_uv, exported as glTF TEXCOORD_1) -- per-vertex, computed
                // once server-side from each vertex's own true observed panorama
                // pixel where available, falling back to a position-derived
                // projection only where there's no real observation (see
                // HeightMapGenerator._panorama_uv_from_height). Replaces this
                // shader's own live EquirectUV(posOS) computation below, which had
                // no way to prefer a directly-observed pixel over one re-derived
                // from Y_pos after this stage's/upstream stages' own smoothing and
                // noise had already perturbed it.
                float2 panoUV     : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float2 uv         : TEXCOORD2;
                float  fogFactor  : TEXCOORD3;
                float2 panoUV     : TEXCOORD4;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // ── Helpers ────────────────────────────────────────────────────

            // Per-layer UV: the baked panorama UV (see Attributes.panoUV) for panorama
            // layers, tiled planar for synthetic ones. The panorama layer used to
            // recompute an equirect projection live from object-space position here
            // instead -- correct in direction, but with no way to prefer a vertex's
            // own directly-observed panorama pixel over one re-derived from Y_pos
            // after upstream smoothing/noise had already perturbed it (see panoUV's
            // own comment). Baking it server-side, once, fixes that outright and is
            // also strictly cheaper than doing this trig per fragment.
            float2 LayerUV(float2 meshUV, float2 panoUV, float tileRepeat, int layerBit)
            {
                if (layerBit != 0)
                    return panoUV;
                return meshUV * tileRepeat;
            }

            // Top-down 0→1 UV over the terrain grid, derived from object-space XZ --
            // see _TerrainExtent for why this is computed rather than read from
            // TEXCOORD0. Matches the server's own canvas convention exactly:
            // pattern_texture._world_to_px and _panorama_visibility_weight both index
            // row 0 at Z = -half, and a PNG's row 0 lands at v = 1 once Unity uploads
            // it, so V is flipped here.
            float2 TerrainGridUV(float3 positionOS)
            {
                float2 mn = _TerrainExtent.xy;
                float2 mx = _TerrainExtent.zw;
                float2 span = max(mx - mn, 1e-5);
                float u = (positionOS.x - mn.x) / span.x;
                float v = (positionOS.z - mn.y) / span.y;
                return float2(u, 1.0 - v);
            }

            // ── Vertex ─────────────────────────────────────────────────────
            Varyings Vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                VertexPositionInputs posInputs    = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs   normalInputs = GetVertexNormalInputs(IN.normalOS);

                OUT.positionCS = posInputs.positionCS;
                OUT.positionWS = posInputs.positionWS;
                OUT.normalWS   = normalInputs.normalWS;
                OUT.uv         = TerrainGridUV(IN.positionOS.xyz);
                OUT.fogFactor  = ComputeFogFactor(posInputs.positionCS.z);
                OUT.panoUV     = IN.panoUV;
                return OUT;
            }

            // ── Fragment ───────────────────────────────────────────────────
            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 uv = IN.uv;

                // Blend maps sampled at terrain UV (0→1)
                half4 blend0 = SAMPLE_TEXTURE2D(_BlendMap0, sampler_BlendMap0, uv);
                half4 blend1 = SAMPLE_TEXTURE2D(_BlendMap1, sampler_BlendMap1, uv);

                // Per-layer UV — baked panorama UV for panorama layers, tiled planar for synthetic
                float2 uv0 = LayerUV(uv, IN.panoUV, _Layer0TileRepeat, (_EquirectLayers >> 0) & 1);
                float2 uv1 = LayerUV(uv, IN.panoUV, _Layer1TileRepeat, (_EquirectLayers >> 1) & 1);
                float2 uv2 = LayerUV(uv, IN.panoUV, _Layer2TileRepeat, (_EquirectLayers >> 2) & 1);
                float2 uv3 = LayerUV(uv, IN.panoUV, _Layer3TileRepeat, (_EquirectLayers >> 3) & 1);
                float2 uv4 = LayerUV(uv, IN.panoUV, _Layer4TileRepeat, (_EquirectLayers >> 4) & 1);
                float2 uv5 = LayerUV(uv, IN.panoUV, _Layer5TileRepeat, (_EquirectLayers >> 5) & 1);
                float2 uv6 = LayerUV(uv, IN.panoUV, _Layer6TileRepeat, (_EquirectLayers >> 6) & 1);
                float2 uv7 = LayerUV(uv, IN.panoUV, _Layer7TileRepeat, (_EquirectLayers >> 7) & 1);

                // Layer tiles — alpha carries a local micro-height map (server-baked from
                // the tile's own high-frequency detail) used for height-biased blending below.
                half4 tex0 = SAMPLE_TEXTURE2D(_Layer0Tile, sampler_Layer0Tile, uv0);
                half4 tex1 = SAMPLE_TEXTURE2D(_Layer1Tile, sampler_Layer1Tile, uv1);
                half4 tex2 = SAMPLE_TEXTURE2D(_Layer2Tile, sampler_Layer2Tile, uv2);
                half4 tex3 = SAMPLE_TEXTURE2D(_Layer3Tile, sampler_Layer3Tile, uv3);
                half4 tex4 = SAMPLE_TEXTURE2D(_Layer4Tile, sampler_Layer4Tile, uv4);
                half4 tex5 = SAMPLE_TEXTURE2D(_Layer5Tile, sampler_Layer5Tile, uv5);
                half4 tex6 = SAMPLE_TEXTURE2D(_Layer6Tile, sampler_Layer6Tile, uv6);
                half4 tex7 = SAMPLE_TEXTURE2D(_Layer7Tile, sampler_Layer7Tile, uv7);

                // Height-biased blend weights: each layer's server-normalised blend-map
                // weight is biased by its own local micro-height before renormalising, so
                // e.g. gravel pokes through soil where the gravel tile's local height is
                // highest rather than fading in as a flat, uniform cross-blend. Layers with
                // no alpha data (the panorama layer, or unused slots) decode to alpha = 1
                // and are left unbiased. At _HeightBlendSharpness = 0, pow(x, 0) = 1 for all
                // layers and this reduces exactly to the original flat weighted blend.
                half eps = 1e-4h;
                half w0 = blend0.r * pow(tex0.a + eps, _HeightBlendSharpness);
                half w1 = blend0.g * pow(tex1.a + eps, _HeightBlendSharpness);
                half w2 = blend0.b * pow(tex2.a + eps, _HeightBlendSharpness);
                half w3 = blend0.a * pow(tex3.a + eps, _HeightBlendSharpness);
                half w4 = blend1.r * pow(tex4.a + eps, _HeightBlendSharpness);
                half w5 = blend1.g * pow(tex5.a + eps, _HeightBlendSharpness);
                half w6 = blend1.b * pow(tex6.a + eps, _HeightBlendSharpness);
                half w7 = blend1.a * pow(tex7.a + eps, _HeightBlendSharpness);

                half invWSum = rcp(w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + eps);
                w0 *= invWSum; w1 *= invWSum; w2 *= invWSum; w3 *= invWSum;
                w4 *= invWSum; w5 *= invWSum; w6 *= invWSum; w7 *= invWSum;

                half3 albedo =
                    tex0.rgb * w0 + tex1.rgb * w1 + tex2.rgb * w2 + tex3.rgb * w3 +
                    tex4.rgb * w4 + tex5.rgb * w5 + tex6.rgb * w6 + tex7.rgb * w7;

                // Blend per-layer smoothness with the same height-biased weights
                half smoothness =
                    _Layer0Smoothness * w0 + _Layer1Smoothness * w1 +
                    _Layer2Smoothness * w2 + _Layer3Smoothness * w3 +
                    _Layer4Smoothness * w4 + _Layer5Smoothness * w5 +
                    _Layer6Smoothness * w6 + _Layer7Smoothness * w7;

                // URP PBR lighting
                InputData inputData = (InputData)0;
                inputData.positionWS             = IN.positionWS;
                inputData.normalWS               = normalize(IN.normalWS);
                inputData.viewDirectionWS        = GetWorldSpaceNormalizeViewDir(IN.positionWS);
                inputData.fogCoord               = IN.fogFactor;
                inputData.vertexLighting         = half3(0, 0, 0);
                inputData.bakedGI                = SampleSH(inputData.normalWS);
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(IN.positionCS);
                inputData.shadowMask             = half4(1, 1, 1, 1);

#if defined(MAIN_LIGHT_CALCULATE_SHADOWS)
                inputData.shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
#else
                inputData.shadowCoord = float4(0, 0, 0, 0);
#endif

                SurfaceData surfaceData = (SurfaceData)0;
                surfaceData.albedo     = albedo;
                surfaceData.metallic   = 0.0h;
                surfaceData.smoothness = smoothness;
                surfaceData.normalTS   = half3(0.0h, 0.0h, 1.0h);
                surfaceData.occlusion  = 1.0h;
                surfaceData.alpha      = 1.0h;

                half4 color = UniversalFragmentPBR(inputData, surfaceData);
                color.rgb = MixFog(color.rgb, IN.fogFactor);
                return color;
            }
            ENDHLSL
        }

        // ── Shadow Caster ──────────────────────────────────────────────────
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex   ShadowVert
            #pragma fragment ShadowFrag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Layer0TileRepeat;
            CBUFFER_END

            float3 _LightDirection;

            struct ShadowAttribs
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct ShadowVaryings
            {
                float4 positionCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            ShadowVaryings ShadowVert(ShadowAttribs IN)
            {
                ShadowVaryings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                float3 posWS    = TransformObjectToWorld(IN.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(IN.normalOS);
                OUT.positionCS  = TransformWorldToHClip(ApplyShadowBias(posWS, normalWS, _LightDirection));
                return OUT;
            }

            half4 ShadowFrag(ShadowVaryings IN) : SV_Target { return 0; }
            ENDHLSL
        }

        // ── Depth Only ─────────────────────────────────────────────────────
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite On
            ColorMask R
            Cull Back

            HLSLPROGRAM
            #pragma vertex   DepthVert
            #pragma fragment DepthFrag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Layer0TileRepeat;
            CBUFFER_END

            struct DepthAttribs
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct DepthVaryings
            {
                float4 positionCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            DepthVaryings DepthVert(DepthAttribs IN)
            {
                DepthVaryings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            half DepthFrag(DepthVaryings IN) : SV_Target { return IN.positionCS.z; }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
