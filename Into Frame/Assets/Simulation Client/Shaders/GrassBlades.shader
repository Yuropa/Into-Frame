Shader "IntoFrame/GrassBlades"
{
    // Instanced grass tufts grown by GrassField.cs from the server's grass field
    // (server/pipeline/grass_cover/grass_field.py). Each instance is one tuft of
    // blades whose UV.y runs 0 at the root to 1 at the tip; the instance colour is
    // the photograph's (delit) ground colour at that spot, darkened toward the root
    // and lifted toward the tip, and the blades bend with the same wind globals
    // WindField.cs sets for every other swaying thing in the scene.
    Properties
    {
        _BaseDarkness ("Root Darkness", Range(0, 1)) = 0.45
        _TipBrightness ("Tip Brightness", Range(0.5, 2)) = 1.2
        _Translucency ("Translucency", Range(0, 1)) = 0.35
        _SwayStrength ("Sway (m per unit wind at the tip)", Float) = 0.6
        _SwayFrequency ("Sway Frequency", Float) = 2.1
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"          = "Geometry"
        }

        Pass
        {
            Name "UniversalForward"
            Tags { "LightMode" = "UniversalForward" }
            Cull Off
            ZWrite On

            HLSLPROGRAM
            #pragma vertex   Vert
            #pragma fragment Frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _BaseDarkness;
                float _TipBrightness;
                float _Translucency;
                float _SwayStrength;
                float _SwayFrequency;
            CBUFFER_END

            UNITY_INSTANCING_BUFFER_START(GrassProps)
                UNITY_DEFINE_INSTANCED_PROP(float4, _GrassColor)
            UNITY_INSTANCING_BUFFER_END(GrassProps)

            // Set every frame by WindField.cs.
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
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float  height     : TEXCOORD2;
                float  fogFactor  : TEXCOORD3;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // WindField.GustPulse, per tuft root: the same gust front crosses the
            // grass as crosses every rigged plant.
            float GustPulse(float2 rootXZ)
            {
                float2 dir = _WindDirXZ.xy;
                float delay = dot(rootXZ, dir) / max(_WindGustSpeed, 0.01);
                float period = max(_WindGustInterval, 0.01);
                float u = _Time.y - delay;
                float phase = u - floor(u / period) * period;
                float duration = clamp(_WindGustDuration, 0.01, period);
                if (phase >= duration) return 0.0;
                return pow(sin(phase / duration * PI), max(_WindGustSharpness, 0.01));
            }

            Varyings Vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                float3 rootWS = TransformObjectToWorld(float3(0, 0, 0));
                float3 posWS  = TransformObjectToWorld(v.positionOS.xyz);
                float  h      = saturate(v.uv.y);

                float2 dir = _WindDirXZ.xy;
                float strength = _WindBaseStrength + _WindGustStrength * GustPulse(rootWS.xz);
                float flutter = sin(_Time.y * _SwayFrequency + dot(rootWS.xz, float2(1.7, 1.3)))
                              + 0.5 * sin(_Time.y * _SwayFrequency * 2.3 + dot(rootWS.xz, float2(-2.1, 0.9)));
                float bladeHeight = max(posWS.y - rootWS.y, 0.0);
                float bend = strength * _SwayStrength * (0.75 + 0.25 * flutter) * h * h;
                float2 offset = dir * bend * saturate(bladeHeight / 0.3 + 0.3);
                posWS.xz += offset;
                // Keep the blade's length roughly constant as it bends.
                posWS.y -= 0.5 * dot(offset, offset) / max(bladeHeight, 0.05);

                o.positionWS = posWS;
                o.positionCS = TransformWorldToHClip(posWS);
                // Soften toward up: a blade is too thin for its own normal to read,
                // and the meadow as a whole should shade like a surface.
                float3 n = TransformObjectToWorldNormal(v.normalOS);
                o.normalWS = normalize(lerp(n, float3(0, 1, 0), 0.55));
                o.height = h;
                o.fogFactor = ComputeFogFactor(o.positionCS.z);
                return o;
            }

            half4 Frag(Varyings i, bool frontFace : SV_IsFrontFace) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                float3 colour = UNITY_ACCESS_INSTANCED_PROP(GrassProps, _GrassColor).rgb;
                float3 albedo = colour * lerp(_BaseDarkness, _TipBrightness, i.height);

                float3 n = normalize(i.normalWS);
                float3 v = normalize(GetWorldSpaceViewDir(i.positionWS));
                Light mainLight = GetMainLight(TransformWorldToShadowCoord(i.positionWS));
                float3 l = mainLight.direction;
                float ndl = saturate(dot(n, l)) * 0.8 + 0.2 * saturate(abs(dot(n, l)));
                float trans = pow(saturate(dot(v, -l)), 4.0) * _Translucency * i.height;
                float3 direct = mainLight.color * (ndl * mainLight.shadowAttenuation + trans);
                float3 ambient = SampleSH(n) * lerp(0.6, 1.0, i.height);
                float3 c = albedo * (direct + ambient);
                c = MixFog(c, i.fogFactor);
                return half4(c, 1.0);
            }
            ENDHLSL
        }
    }
}
