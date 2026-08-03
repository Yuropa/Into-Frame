Shader "IntoFrame/WaterSurface"
{
    Properties
    {
        // Mirrors the water_material PBRMaterial baked server-side in
        // TerrainMeshGenerator.generate (terrain_generator.py) -- baseColorFactor
        // [0.10, 0.30, 0.45, 0.75], roughnessFactor 0.15 (smoothness 0.85). Kept as a
        // literal default here rather than read back from the GLTFast-generated
        // material at runtime (see WaterMaterialManager): this shader replaces that
        // material outright, so there is nothing reliable to read the color from
        // beyond re-deriving it from the same two numbers python already picked.
        _BaseColor ("Base Color", Color) = (0.10, 0.30, 0.45, 0.75)
        _Smoothness ("Smoothness", Range(0,1)) = 0.85

        // Sum-of-sines wave parameters. Three fixed, non-harmonic octaves (see
        // ComputeWave) give a gentle, non-repeating ripple without needing a normal
        // map or any authored texture -- the water mesh is flat-shaded PBR only.
        _WaveAmplitude ("Wave Amplitude (m)", Float) = 0.04
        _WaveFrequency ("Wave Frequency (rad/m)", Float) = 0.8
        _WaveSpeed ("Wave Speed", Float) = 0.6
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"          = "Transparent"
        }
        LOD 300

        Pass
        {
            Name "UniversalForward"
            Tags { "LightMode" = "UniversalForward" }

            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Back

            HLSLPROGRAM
            #pragma vertex   Vert
            #pragma fragment Frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            #pragma multi_compile _ SHADOWS_SHADOWMASK
            #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            #pragma multi_compile _ LIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float  _Smoothness;
                float  _WaveAmplitude;
                float  _WaveFrequency;
                float  _WaveSpeed;
            CBUFFER_END

            // Scene-global wind (WindField.cs), pushed via Shader.SetGlobalX every
            // frame -- deliberately declared outside UnityPerMaterial (globals aren't
            // per-material data; putting them in that cbuffer would just make this
            // shader SRP-Batcher-incompatible for no benefit). Same gust formula as
            // WindField.SampleStrength in C#, so the water swells in step with the
            // same gust front that's sweeping amplitude up on WindSway foliage.
            float4 _WindDirXZ;
            float  _WindBaseStrength;
            float  _WindGustStrength;
            float  _WindGustInterval;
            float  _WindGustDuration;
            float  _WindGustSpeed;
            float  _WindGustSharpness;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float  fogFactor  : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // Sum of three non-harmonic sine waves travelling in different XZ
            // directions. Frequency/speed multipliers (1.0, 1.9, 3.7) and (1.0, 1.3,
            // 1.9) are deliberately non-integer ratios of each other so the combined
            // pattern doesn't visibly repeat on a short period. Amplitude weights
            // (0.55, 0.30, 0.15) keep the low-frequency octave dominant so the result
            // reads as gentle swell rather than choppy noise.
            //
            // height(x,z,t) = sum_i A_i * sin(dot(dir_i, xz) * k_i + t * s_i)
            // Analytic slope (dHeight/dx, dHeight/dz) comes along for free from the
            // same terms, so the surface normal can be derived without a second
            // (finite-difference) sample -- exact, and half the ALU cost.
            void ComputeWave(float2 posXZ, float time, out float height, out float2 slope)
            {
                float2 dir1 = normalize(float2( 1.00,  0.60));
                float2 dir2 = normalize(float2(-0.70,  0.90));
                float2 dir3 = normalize(float2( 0.35, -1.00));

                float k1 = _WaveFrequency * 1.0;
                float k2 = _WaveFrequency * 1.9;
                float k3 = _WaveFrequency * 3.7;

                float s1 = _WaveSpeed * 1.0;
                float s2 = _WaveSpeed * 1.3;
                float s3 = _WaveSpeed * 1.9;

                float a1 = _WaveAmplitude * 0.55;
                float a2 = _WaveAmplitude * 0.30;
                float a3 = _WaveAmplitude * 0.15;

                float p1 = dot(dir1, posXZ) * k1 + time * s1;
                float p2 = dot(dir2, posXZ) * k2 + time * s2;
                float p3 = dot(dir3, posXZ) * k3 + time * s3;

                height = a1 * sin(p1) + a2 * sin(p2) + a3 * sin(p3);

                float dH1 = a1 * k1 * cos(p1);
                float dH2 = a2 * k2 * cos(p2);
                float dH3 = a3 * k3 * cos(p3);
                slope = dir1 * dH1 + dir2 * dH2 + dir3 * dH3;
            }

            // Identical shape to WindField.SampleStrength/GustPulse: a single smooth
            // rise-and-fall pulse recurring every _WindGustInterval seconds, delayed
            // per-point by how long the wind takes to physically reach it, on top of
            // an always-on ambient floor. Treated as locally constant across a single
            // vertex's wave amplitude (its own spatial gradient is a second-order
            // effect on top of the wave slope above and not worth the extra derivative).
            float WindEnvelope(float2 posXZ, float time)
            {
                float travel = dot(posXZ, _WindDirXZ.xy);
                float delay  = travel / max(_WindGustSpeed, 0.01);

                float period   = max(_WindGustInterval, 0.01);
                float u        = time - delay;
                float phase    = u - floor(u / period) * period; // wrapped into [0, period)
                float duration = clamp(_WindGustDuration, 0.01, period);
                float pulse    = phase < duration
                    ? pow(sin(phase / duration * 3.14159265), _WindGustSharpness)
                    : 0.0;

                return _WindBaseStrength + _WindGustStrength * pulse;
            }

            Varyings Vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);

                float height;
                float2 slope;
                ComputeWave(positionWS.xz, _Time.y, height, slope);

                float envelope = WindEnvelope(positionWS.xz, _Time.y);
                height *= envelope;
                slope  *= envelope;

                positionWS.y += height;
                float3 normalWS = normalize(float3(-slope.x, 1.0, -slope.y));

                OUT.positionCS = TransformWorldToHClip(positionWS);
                OUT.positionWS = positionWS;
                OUT.normalWS   = normalWS;
                OUT.fogFactor  = ComputeFogFactor(OUT.positionCS.z);
                return OUT;
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                InputData inputData = (InputData)0;
                inputData.positionWS              = IN.positionWS;
                inputData.normalWS                = normalize(IN.normalWS);
                inputData.viewDirectionWS         = GetWorldSpaceNormalizeViewDir(IN.positionWS);
                inputData.fogCoord                = IN.fogFactor;
                inputData.vertexLighting          = half3(0, 0, 0);
                inputData.bakedGI                 = SampleSH(inputData.normalWS);
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(IN.positionCS);
                inputData.shadowMask              = half4(1, 1, 1, 1);

#if defined(MAIN_LIGHT_CALCULATE_SHADOWS)
                inputData.shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
#else
                inputData.shadowCoord = float4(0, 0, 0, 0);
#endif

                SurfaceData surfaceData = (SurfaceData)0;
                surfaceData.albedo     = _BaseColor.rgb;
                surfaceData.metallic   = 0.0h;
                surfaceData.smoothness = _Smoothness;
                surfaceData.normalTS   = half3(0.0h, 0.0h, 1.0h);
                surfaceData.occlusion  = 1.0h;
                surfaceData.alpha      = _BaseColor.a;

                half4 color = UniversalFragmentPBR(inputData, surfaceData);
                color.a   = _BaseColor.a;
                color.rgb = MixFog(color.rgb, IN.fogFactor);
                return color;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
