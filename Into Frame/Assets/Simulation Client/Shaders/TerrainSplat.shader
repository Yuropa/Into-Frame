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

        // How many times each tile repeats across the full terrain
        _TileRepeat ("Tile Repeat",  Float) = 8.0
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
                float  _TileRepeat;
            CBUFFER_END

            // ── Structs ────────────────────────────────────────────────────
            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float2 uv         : TEXCOORD2;
                float  fogFactor  : TEXCOORD3;
                UNITY_VERTEX_OUTPUT_STEREO
            };

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
                OUT.uv         = IN.uv;
                OUT.fogFactor  = ComputeFogFactor(posInputs.positionCS.z);
                return OUT;
            }

            // ── Fragment ───────────────────────────────────────────────────
            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 uv     = IN.uv;
                float2 tileUV = uv * _TileRepeat;

                // Blend maps sampled at terrain UV (0→1)
                half4 blend0 = SAMPLE_TEXTURE2D(_BlendMap0, sampler_BlendMap0, uv);
                half4 blend1 = SAMPLE_TEXTURE2D(_BlendMap1, sampler_BlendMap1, uv);

                // Layer tiles sampled at tiled UV
                half3 col0 = SAMPLE_TEXTURE2D(_Layer0Tile, sampler_Layer0Tile, tileUV).rgb;
                half3 col1 = SAMPLE_TEXTURE2D(_Layer1Tile, sampler_Layer1Tile, tileUV).rgb;
                half3 col2 = SAMPLE_TEXTURE2D(_Layer2Tile, sampler_Layer2Tile, tileUV).rgb;
                half3 col3 = SAMPLE_TEXTURE2D(_Layer3Tile, sampler_Layer3Tile, tileUV).rgb;
                half3 col4 = SAMPLE_TEXTURE2D(_Layer4Tile, sampler_Layer4Tile, tileUV).rgb;
                half3 col5 = SAMPLE_TEXTURE2D(_Layer5Tile, sampler_Layer5Tile, tileUV).rgb;
                half3 col6 = SAMPLE_TEXTURE2D(_Layer6Tile, sampler_Layer6Tile, tileUV).rgb;
                half3 col7 = SAMPLE_TEXTURE2D(_Layer7Tile, sampler_Layer7Tile, tileUV).rgb;

                // Weighted blend — weights pre-normalised by server, no normalisation needed
                half3 albedo =
                    col0 * blend0.r + col1 * blend0.g + col2 * blend0.b + col3 * blend0.a +
                    col4 * blend1.r + col5 * blend1.g + col6 * blend1.b + col7 * blend1.a;

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
                surfaceData.smoothness = 0.1h;
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
                float _TileRepeat;
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
                float _TileRepeat;
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
