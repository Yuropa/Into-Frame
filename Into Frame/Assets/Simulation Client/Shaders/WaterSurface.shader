Shader "IntoFrame/WaterSurface"
{
    Properties
    {
        // Mirrors the water_material PBRMaterial baked server-side in
        // TerrainMeshGenerator.generate (terrain_generator.py) -- baseColorFactor
        // [0.10, 0.30, 0.45, 0.90]. WaterMaterialManager copies the server's measured
        // water colour (TerrainMeshStage._colour_water_from_photo) into _BaseColor at
        // load, so this is the DEEP water colour of the photograph. The alpha is no
        // longer a constant transparency: this shader composites its own refraction
        // from the opaque pass, so it writes alpha 1 and fades only at the shoreline.
        _BaseColor ("Deep Water Color", Color) = (0.10, 0.30, 0.45, 0.90)
        // Shallow water lets the bottom through; what you see of the water itself
        // there is a lighter, greener tint than the deep colour. Derived from
        // _BaseColor by WaterMaterialManager unless set.
        _ShallowColor ("Shallow Water Color", Color) = (0.25, 0.50, 0.55, 1.0)
        _Smoothness ("Smoothness", Range(0,1)) = 0.95

        // ── Depth / shore ────────────────────────────────────────────────────
        // Beer-Lambert absorption: how many metres of water turn the refracted
        // bottom into the deep colour (63% at this depth).
        _AbsorptionDepth ("Absorption Depth (m)", Float) = 1.2
        // The sheet goes fully transparent where it meets the bank, over this many
        // metres of water depth, so the shoreline is a wet edge rather than a cut.
        _ShoreFadeDepth ("Shore Fade Depth (m)", Float) = 0.2
        // Foam at the shoreline: a noisy white band over this water depth.
        _FoamDepth ("Foam Depth (m)", Float) = 0.5
        _FoamStrength ("Foam Strength", Range(0,1)) = 0.6
        // How far the refracted bottom is displaced by the surface normal, in
        // screen UV per unit of normal tilt. Small: it only has to break straight lines.
        _RefractionStrength ("Refraction Strength", Float) = 0.04

        // ── Reflection ───────────────────────────────────────────────────────
        // The scene's reflection probe (EnvironmentLighting renders it from the
        // panorama skybox) is what the surface mirrors. Strength is a fudge on top
        // of the physical Fresnel for probes whose exposure differs from the sky's.
        _ReflectionStrength ("Reflection Strength", Range(0,2)) = 1.0
        // 0.02 is water's real normal-incidence reflectance (IOR 1.33).
        _FresnelF0 ("Fresnel F0", Range(0,0.2)) = 0.02

        // ── Sun ──────────────────────────────────────────────────────────────
        // A tight specular track toward the sun on top of the probe reflection: the
        // probe's sun is a few texels wide and blurred by the probe's resolution.
        _GlintPower ("Glint Sharpness", Float) = 900.0
        _GlintStrength ("Glint Strength", Float) = 3.0
        // A broader sun lobe so the glitter has a halo instead of a hard dot.
        _SunLobePower ("Sun Lobe Sharpness", Float) = 60.0
        _SunLobeStrength ("Sun Lobe Strength", Float) = 0.25

        // ── Waves ────────────────────────────────────────────────────────────
        // Sum-of-sines vertex swell (ComputeWave) and per-pixel capillary ripple
        // (ComputeDetailSlope). The swell moves geometry; the ripple only tilts the
        // normal and stays sharp at any distance.
        _WaveAmplitude ("Wave Amplitude (m)", Float) = 0.04
        _WaveFrequency ("Wave Frequency (rad/m)", Float) = 0.8
        _WaveSpeed ("Wave Speed", Float) = 0.6
        _DetailAmplitude ("Detail Amplitude", Range(0,1)) = 0.35
        _DetailFrequency ("Detail Frequency (rad/m)", Float) = 7.0
        _DetailSpeed ("Detail Speed", Float) = 1.4
        // Past roughly this range a ripple is smaller than a pixel and only aliases
        // (and in stereo the two eyes alias differently and the surface stops fusing).
        _DetailFadeDistance ("Detail Fade Distance (m)", Float) = 45.0
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
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile _ _FORWARD_PLUS
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            // Scene depth and the opaque colour buffer: the URP asset has
            // "Depth Texture" and "Opaque Texture" on (Assets/Settings/Project
            // Configuration/*.asset). Without them SampleSceneDepth returns the far
            // plane and SampleSceneColor black, and the water reads as a deep sheet.
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _ShallowColor;
                float  _Smoothness;
                float  _AbsorptionDepth;
                float  _ShoreFadeDepth;
                float  _FoamDepth;
                float  _FoamStrength;
                float  _RefractionStrength;
                float  _ReflectionStrength;
                float  _FresnelF0;
                float  _GlintPower;
                float  _GlintStrength;
                float  _SunLobePower;
                float  _SunLobeStrength;
                float  _WaveAmplitude;
                float  _WaveFrequency;
                float  _WaveSpeed;
                float  _DetailAmplitude;
                float  _DetailFrequency;
                float  _DetailSpeed;
                float  _DetailFadeDistance;
            CBUFFER_END

            // Scene-global wind (WindField.cs), pushed via Shader.SetGlobalX every
            // frame -- declared outside UnityPerMaterial (globals are not
            // per-material data; in that cbuffer the shader would not be SRP-Batcher
            // compatible). Same gust formula as WindField.SampleStrength in C#.
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
                float4 screenPos  : TEXCOORD3;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // Sum of three non-harmonic sine waves travelling in different XZ
            // directions. Frequency/speed multipliers are non-integer ratios so the
            // pattern does not visibly repeat; amplitude weights keep the low octave
            // dominant so it reads as swell rather than chop. The analytic slope
            // comes from the same terms, so the normal costs no second sample.
            void ComputeWave(float2 posXZ, float time, out float height, out float2 slope)
            {
                float2 dir1 = normalize(float2( 1.00,  0.60));
                float2 dir2 = normalize(float2(-0.70,  0.90));
                float2 dir3 = normalize(float2( 0.35, -1.00));

                float k1 = _WaveFrequency * 1.0, k2 = _WaveFrequency * 1.9, k3 = _WaveFrequency * 3.7;
                float s1 = _WaveSpeed * 1.0,     s2 = _WaveSpeed * 1.3,     s3 = _WaveSpeed * 1.9;
                float a1 = _WaveAmplitude * 0.55, a2 = _WaveAmplitude * 0.30, a3 = _WaveAmplitude * 0.15;

                float p1 = dot(dir1, posXZ) * k1 + time * s1;
                float p2 = dot(dir2, posXZ) * k2 + time * s2;
                float p3 = dot(dir3, posXZ) * k3 + time * s3;

                height = a1 * sin(p1) + a2 * sin(p2) + a3 * sin(p3);
                slope  = dir1 * (a1 * k1 * cos(p1)) + dir2 * (a2 * k2 * cos(p2)) + dir3 * (a3 * k3 * cos(p3));
            }

            // Per-pixel capillary ripple: slope only, nothing moves geometry, so it
            // is safe at frequencies the mesh could never resolve. Directions are
            // deliberately not the swell's: fine chop runs across the swell.
            float2 ComputeDetailSlope(float2 posXZ, float time)
            {
                float2 dir1 = normalize(float2( 0.80, -0.55));
                float2 dir2 = normalize(float2(-0.30, -0.95));
                float2 dir3 = normalize(float2( 0.95,  0.25));
                float2 dir4 = normalize(float2(-0.65,  0.75));

                float k1 = _DetailFrequency * 1.00, k2 = _DetailFrequency * 1.70;
                float k3 = _DetailFrequency * 2.90, k4 = _DetailFrequency * 4.60;
                float s1 = _DetailSpeed * 1.00, s2 = _DetailSpeed * 1.40;
                float s3 = _DetailSpeed * 0.80, s4 = _DetailSpeed * 1.90;
                float a1 = _DetailAmplitude * 0.40, a2 = _DetailAmplitude * 0.28;
                float a3 = _DetailAmplitude * 0.20, a4 = _DetailAmplitude * 0.12;

                float p1 = dot(dir1, posXZ) * k1 + time * s1;
                float p2 = dot(dir2, posXZ) * k2 + time * s2;
                float p3 = dot(dir3, posXZ) * k3 + time * s3;
                float p4 = dot(dir4, posXZ) * k4 + time * s4;

                return dir1 * (a1 * cos(p1)) + dir2 * (a2 * cos(p2))
                     + dir3 * (a3 * cos(p3)) + dir4 * (a4 * cos(p4));
            }

            // A cheap value noise for the foam's breakup; animated by time.
            float Hash21(float2 p)
            {
                p = frac(p * float2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return frac(p.x * p.y);
            }
            float ValueNoise(float2 p)
            {
                float2 i = floor(p), f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = Hash21(i), b = Hash21(i + float2(1, 0));
                float c = Hash21(i + float2(0, 1)), d = Hash21(i + float2(1, 1));
                return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
            }

            // Identical shape to WindField.SampleStrength/GustPulse: one smooth
            // rise-and-fall pulse every _WindGustInterval seconds, delayed per point by
            // the wind's travel time, on an always-on ambient floor.
            float WindEnvelope(float2 posXZ, float time)
            {
                float travel = dot(posXZ, _WindDirXZ.xy);
                float delay  = travel / max(_WindGustSpeed, 0.01);
                float period   = max(_WindGustInterval, 0.01);
                float u        = time - delay;
                float phase    = u - floor(u / period) * period;
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

                float height; float2 slope;
                ComputeWave(positionWS.xz, _Time.y, height, slope);
                float envelope = WindEnvelope(positionWS.xz, _Time.y);
                height *= envelope;
                slope  *= envelope;

                positionWS.y += height;
                OUT.positionCS = TransformWorldToHClip(positionWS);
                OUT.positionWS = positionWS;
                OUT.normalWS   = normalize(float3(-slope.x, 1.0, -slope.y));
                OUT.fogFactor  = ComputeFogFactor(OUT.positionCS.z);
                OUT.screenPos  = ComputeScreenPos(OUT.positionCS);
                return OUT;
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float3 cameraPos = GetCameraPositionWS();
                float viewDistance = distance(IN.positionWS, cameraPos);

                // ── normal: swell + faded capillary ripple, combined in slope space ──
                float detailFade = saturate(1.0 - viewDistance / max(_DetailFadeDistance, 0.01));
                detailFade *= detailFade;
                float3 normalWS = normalize(IN.normalWS);
                float2 detailSlope = 0;
                if (detailFade > 0.0)
                {
                    detailSlope = ComputeDetailSlope(IN.positionWS.xz, _Time.y)
                                * detailFade * WindEnvelope(IN.positionWS.xz, _Time.y);
                    float2 baseSlope = -normalWS.xz / max(normalWS.y, 1e-4);
                    float2 totalSlope = baseSlope + detailSlope;
                    normalWS = normalize(float3(-totalSlope.x, 1.0, -totalSlope.y));
                }

                float3 viewDirWS = normalize(cameraPos - IN.positionWS);
                float  NdotV     = saturate(dot(normalWS, viewDirWS));

                // ── water depth under this pixel, from the opaque pass ──────────
                float2 screenUV = IN.screenPos.xy / max(IN.screenPos.w, 1e-5);
                float  surfaceEye = LinearEyeDepth(IN.positionCS.z, _ZBufferParams);
                #if UNITY_REVERSED_Z
                    float rawDepth = SampleSceneDepth(screenUV);
                #else
                    float rawDepth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(screenUV));
                #endif
                float sceneEye   = LinearEyeDepth(rawDepth, _ZBufferParams);
                // Eye-space thickness along the view ray, turned into vertical
                // water depth: the ray crosses the water at the grazing angle it
                // arrives at, so the same bottom is "thicker" the flatter you look.
                float thickness  = max(sceneEye - surfaceEye, 0.0);
                float waterDepth = thickness * max(NdotV, 0.05);

                // ── refraction: the bottom, displaced by the ripple ────────────
                // The displaced sample must still lie under water: a sample that
                // lands on something nearer than the surface (a boat's hull, the
                // bank) would pull that object's colour under the sheet.
                float2 refractUV = screenUV + detailSlope * _RefractionStrength * detailFade
                                 + normalWS.xz * _RefractionStrength * 0.25;
                #if UNITY_REVERSED_Z
                    float rawRefr = SampleSceneDepth(refractUV);
                #else
                    float rawRefr = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(refractUV));
                #endif
                if (LinearEyeDepth(rawRefr, _ZBufferParams) < surfaceEye) refractUV = screenUV;
                float3 bottom = SampleSceneColor(refractUV);

                // ── absorption: the bottom through the water column ───────────
                float absorb = 1.0 - exp(-waterDepth / max(_AbsorptionDepth, 0.01));
                float3 waterTint = lerp(_ShallowColor.rgb, _BaseColor.rgb, absorb);
                // Ambient light scattered back up by the water itself.
                float3 ambient = SampleSH(float3(0, 1, 0));
                float3 scattered = waterTint * ambient;
                float3 underwater = lerp(bottom * lerp(1.0, _ShallowColor.rgb, saturate(absorb * 2.0)), scattered, absorb);

                // ── reflection: the scene's probe (the panorama sky) ──────────
                float3 reflectVector = reflect(-viewDirWS, normalWS);
                // Never reflect below the horizon: the probe's lower half is the
                // ground under the camera, and a ripple tilting the reflection down
                // shows it as brown flecks on the water.
                reflectVector.y = abs(reflectVector.y) * 0.5 + 0.5 * max(reflectVector.y, 0.0);
                reflectVector = normalize(reflectVector);
                float perceptualRoughness = 1.0 - _Smoothness;
                float3 reflection = GlossyEnvironmentReflection(reflectVector, IN.positionWS, perceptualRoughness, 1.0h, screenUV);
                reflection *= _ReflectionStrength;

                // Schlick against water's F0, so the ramp is the physical one.
                float fresnel = _FresnelF0 + (1.0 - _FresnelF0) * pow(1.0 - NdotV, 5.0);

                float3 color = lerp(underwater, reflection, fresnel);

                // ── sun: shadowed glitter track and a soft lobe ───────────────
                #if defined(MAIN_LIGHT_CALCULATE_SHADOWS)
                    float4 shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
                #else
                    float4 shadowCoord = float4(0, 0, 0, 0);
                #endif
                Light mainLight = GetMainLight(shadowCoord);
                float3 halfDir = SafeNormalize(float3(mainLight.direction) + viewDirWS);
                float  NdotH   = saturate(dot(normalWS, halfDir));
                float  glint   = pow(NdotH, max(_GlintPower, 1.0)) * _GlintStrength;
                float  lobe    = pow(NdotH, max(_SunLobePower, 1.0)) * _SunLobeStrength;
                color += (glint + lobe) * mainLight.color * mainLight.shadowAttenuation * fresnel;

                // ── shoreline foam ────────────────────────────────────────────
                float foamBand = saturate(1.0 - waterDepth / max(_FoamDepth, 0.01));
                float n = ValueNoise(IN.positionWS.xz * 3.0 + _Time.y * 0.35)
                        * ValueNoise(IN.positionWS.xz * 9.0 - _Time.y * 0.6);
                float foam = smoothstep(0.15, 0.6, n * 2.0 + foamBand - 0.6) * foamBand * _FoamStrength * detailFade;
                float3 foamColor = (mainLight.color * mainLight.shadowAttenuation * 0.8 + ambient) * 0.9;
                color = lerp(color, foamColor, saturate(foam));

                color = MixFog(color, IN.fogFactor);

                // Opaque everywhere the water has depth: the bottom was composited
                // here from the opaque pass. Only the last few centimetres at the
                // bank fade out, so the sheet meets the terrain as a wet edge.
                float alpha = saturate(waterDepth / max(_ShoreFadeDepth, 0.001));
                return half4(color, alpha);
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
