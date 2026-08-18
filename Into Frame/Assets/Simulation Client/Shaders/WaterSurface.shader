Shader "IntoFrame/WaterSurface"
{
    Properties
    {
        // Mirrors the water_material PBRMaterial baked server-side in
        // TerrainMeshGenerator.generate (terrain_generator.py) -- baseColorFactor
        // [0.10, 0.30, 0.45, 0.90], roughnessFactor 0.08 (smoothness 0.92). Kept as a
        // literal default here rather than read back from the GLTFast-generated
        // material at runtime (see WaterMaterialManager): this shader replaces that
        // material outright, so there is nothing reliable to read the color from
        // beyond re-deriving it from the same two numbers python already picked.
        // Keep the two in step -- the baked material is what renders on a client with
        // no water shader of its own, which today means the visionOS app.
        // Alpha raised 0.75 -> 0.90. At 0.75 a quarter of the terrain under the
        // water read straight through it, and that terrain is a lakebed carved
        // water_depression_m below the surface with the panorama's own water pixels
        // painted on it -- so what showed through was a second, darker, wrong-scale
        // copy of the water, which is what made it read as tinted glass rather than
        // as water. Real water is not transparent at grazing incidence, which is
        // nearly the whole surface from a standing eye height.
        //
        // Not fully opaque: the shoreline is where the terrain mesh's water hole and
        // this sheet meet, and a little transmission there is what keeps that seam
        // from reading as a hard cut.
        _BaseColor ("Base Color", Color) = (0.10, 0.30, 0.45, 0.90)
        _Smoothness ("Smoothness", Range(0,1)) = 0.92

        // Sum-of-sines wave parameters. Three fixed, non-harmonic octaves (see
        // ComputeWave) give a gentle, non-repeating ripple without needing a normal
        // map or any authored texture -- the water mesh is flat-shaded PBR only.
        _WaveAmplitude ("Wave Amplitude (m)", Float) = 0.04
        _WaveFrequency ("Wave Frequency (rad/m)", Float) = 0.8
        _WaveSpeed ("Wave Speed", Float) = 0.6

        // ── Surface detail ───────────────────────────────────────────────────
        // The waves above are VERTEX displacement, and the water mesh is Poisson-disc
        // sampled with a near-camera density bias (see TerrainMeshGenerator.generate's
        // inner/outer_min_dist) -- so beyond a few tens of metres its vertices are
        // metres apart and the wave carries no detail at all. Past that distance the
        // old surface was a single flat quad's worth of normal, which is most of why
        // it read as a sheet of plastic rather than as water.
        //
        // These octaves are evaluated PER PIXEL and perturb the normal only; they
        // never displace geometry, so they cost nothing in the vertex stage and stay
        // sharp at any distance. Frequencies are much higher than the swell's and
        // amplitudes far smaller: this is the capillary ripple that makes the sun
        // glitter, not more swell.
        _DetailAmplitude ("Detail Amplitude", Range(0,1)) = 0.35
        _DetailFrequency ("Detail Frequency (rad/m)", Float) = 7.0
        _DetailSpeed ("Detail Speed", Float) = 1.4
        // Distance over which detail ripple fades to flat. Past roughly this range a
        // ripple is smaller than a pixel, and keeping it only buys shimmering
        // aliasing that reads as noise -- especially in stereo, where the two eyes
        // alias differently and the surface stops fusing.
        _DetailFadeDistance ("Detail Fade Distance (m)", Float) = 45.0

        // ── Fresnel ──────────────────────────────────────────────────────────
        // 0.02 is water's real normal-incidence reflectance (IOR 1.33). Exposed
        // because it is the single number that decides whether this reads as water,
        // not because it should be changed.
        _FresnelF0 ("Fresnel F0", Range(0,0.2)) = 0.02
        // Sun glitter. URP's own GGX lobe at this smoothness gives a broad sheen;
        // real water gives a tight, very bright specular track toward the sun, and
        // that track is one of the strongest "this is water" cues there is.
        _GlintPower ("Glint Sharpness", Float) = 900.0
        _GlintStrength ("Glint Strength", Float) = 3.0
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
                float  _DetailAmplitude;
                float  _DetailFrequency;
                float  _DetailSpeed;
                float  _DetailFadeDistance;
                float  _FresnelF0;
                float  _GlintPower;
                float  _GlintStrength;
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

            // Per-pixel capillary ripple. Same analytic sum-of-sines idea as
            // ComputeWave, but it returns SLOPE ONLY -- nothing here moves geometry,
            // so it is safe to run at a frequency the mesh could never resolve.
            //
            // Four octaves rather than three, and the directions are deliberately not
            // the swell's: real water has fine chop running across the swell, not
            // along it, and reusing dir1..dir3 here would have produced a single
            // stretched pattern instead of two overlaid ones.
            float2 ComputeDetailSlope(float2 posXZ, float time)
            {
                float2 dir1 = normalize(float2( 0.80, -0.55));
                float2 dir2 = normalize(float2(-0.30, -0.95));
                float2 dir3 = normalize(float2( 0.95,  0.25));
                float2 dir4 = normalize(float2(-0.65,  0.75));

                float k1 = _DetailFrequency * 1.00;
                float k2 = _DetailFrequency * 1.70;
                float k3 = _DetailFrequency * 2.90;
                float k4 = _DetailFrequency * 4.60;

                float s1 = _DetailSpeed * 1.00;
                float s2 = _DetailSpeed * 1.40;
                float s3 = _DetailSpeed * 0.80;
                float s4 = _DetailSpeed * 1.90;

                // Amplitude falls with frequency (roughly 1/k) so no single octave
                // dominates the normal -- the same reason the swell weights taper.
                float a1 = _DetailAmplitude * 0.40;
                float a2 = _DetailAmplitude * 0.28;
                float a3 = _DetailAmplitude * 0.20;
                float a4 = _DetailAmplitude * 0.12;

                float p1 = dot(dir1, posXZ) * k1 + time * s1;
                float p2 = dot(dir2, posXZ) * k2 + time * s2;
                float p3 = dot(dir3, posXZ) * k3 + time * s3;
                float p4 = dot(dir4, posXZ) * k4 + time * s4;

                return dir1 * (a1 * cos(p1))
                     + dir2 * (a2 * cos(p2))
                     + dir3 * (a3 * cos(p3))
                     + dir4 * (a4 * cos(p4));
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

                // Per-pixel ripple, faded out with distance. The vertex wave gives the
                // swell; this gives the surface something to catch light with at
                // ranges where the mesh has no vertices left to bend.
                //
                // The fade is not just an optimisation. A ripple whose wavelength
                // falls below a pixel does not average to flat, it aliases -- and in
                // stereo the two eyes alias out of step, so the surface stops fusing
                // into one depth and starts to sparkle at the wrong distance.
                float viewDistance = distance(IN.positionWS, GetCameraPositionWS());
                float detailFade = saturate(1.0 - viewDistance / max(_DetailFadeDistance, 0.01));
                // Squared so the falloff is gentle near the viewer and quick at the
                // far end, rather than thinning out the detail you can actually see.
                detailFade *= detailFade;

                float3 normalWS = normalize(IN.normalWS);
                if (detailFade > 0.0)
                {
                    float2 detailSlope = ComputeDetailSlope(IN.positionWS.xz, _Time.y)
                                       * detailFade
                                       * WindEnvelope(IN.positionWS.xz, _Time.y);

                    // Combined in SLOPE space, not by blending the two normals.
                    // Both terms are gradients of the same height field, so their
                    // slopes add exactly, while summing unit normals does not: it
                    // shortens the result (darkening the surface once renormalised)
                    // and can collapse toward zero if both tilt hard in opposite
                    // directions. The vertex normal was built as (-sx, 1, -sz), so
                    // its own slope is recoverable as -n.xz / n.y.
                    float2 baseSlope = -normalWS.xz / max(normalWS.y, 1e-4);
                    float2 totalSlope = baseSlope + detailSlope;
                    normalWS = normalize(float3(-totalSlope.x, 1.0, -totalSlope.y));
                }

                // Kept as full-precision locals and used for the Fresnel and glint
                // maths below. URP's InputData declares normalWS/viewDirectionWS as
                // half3, and a half NdotH quantises to about 5e-4 -- which pow()ed to
                // _GlintPower turns the sun track into visible blocky steps rather
                // than a smooth glitter field. Fine for the PBR call, not for this.
                float3 viewDirWS = GetWorldSpaceNormalizeViewDir(IN.positionWS);

                InputData inputData = (InputData)0;
                inputData.positionWS              = IN.positionWS;
                inputData.normalWS                = normalWS;
                inputData.viewDirectionWS         = viewDirWS;
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

                // ── Fresnel-driven opacity ───────────────────────────────────
                // The one change that matters most. URP's BRDF already applies
                // Fresnel to its specular term, but the whole result was then
                // multiplied by a CONSTANT alpha -- so at grazing angles, where real
                // water turns into a mirror, this surface stayed 10% see-through and
                // its reflection stayed 10% washed out by the lakebed behind it. A
                // constant-alpha surface cannot look like water at any alpha: face-on
                // it should be nearly clear and edge-on nearly opaque, and picking one
                // number is picking which half to get wrong.
                //
                // Schlick against water's real F0 (0.02 at IOR 1.33), so the ramp is
                // the physical one rather than a tuned curve.
                float NdotV = saturate(dot(normalWS, viewDirWS));
                float fresnel = _FresnelF0 + (1.0 - _FresnelF0) * pow(1.0 - NdotV, 5.0);

                // ── Sun glitter ──────────────────────────────────────────────
                // A tight Blinn-Phong lobe on top of the PBR result. Deliberately not
                // just "raise smoothness": the glitter track toward the sun is much
                // sharper than the sheen over the rest of the surface, and one GGX
                // lobe cannot be both. Shadow-attenuated so the track breaks where
                // terrain shadows the water, which is what stops it reading as a
                // decal painted on the surface.
                Light mainLight = GetMainLight(inputData.shadowCoord);
                float3 halfDir = SafeNormalize(float3(mainLight.direction) + viewDirWS);
                float NdotH = saturate(dot(normalWS, halfDir));
                float glint = pow(NdotH, max(_GlintPower, 1.0)) * _GlintStrength;
                color.rgb += glint * mainLight.color * mainLight.shadowAttenuation * fresnel;

                // Opaque at grazing, near the authored alpha face-on.
                color.a = lerp(_BaseColor.a, 1.0, fresnel);

                color.rgb = MixFog(color.rgb, IN.fogFactor);
                return color;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
