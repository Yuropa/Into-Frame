// Equirectangular skybox with an arbitrary-axis rotation, so the sky can be spun about
// the sun direction rather than only about Y.
//
// Unity's built-in Skybox/Panoramic exposes _Rotation, which is a yaw only. Spinning
// about yaw moves the sun across the sky and desynchronises it from the directional
// light extracted from that same panorama. Rotating about the axis THROUGH the sun
// leaves the sun exactly where it is and turns everything else around it, which is the
// effect wanted here and also the one that keeps lighting consistent for free.
//
// The lat/long convention below deliberately matches Skybox/Panoramic's own
// ToRadialCoords (latitude from acos(y), longitude from atan2(z, x), then flipped into
// UV space) so that with _SkyRotation set to identity this renders identically to the
// built-in shader. That equivalence is what lets SkyboxSpin fall back to the built-in
// without the sky jumping, and it is the first thing to check if the sky comes out
// mirrored or offset: at t = 0 this must look exactly like the old static skybox.
Shader "Skybox/PanoramaSpin"
{
    Properties
    {
        _MainTex ("Panorama (equirectangular)", 2D) = "grey" {}
        _Exposure ("Exposure", Range(0, 8)) = 1.0
        // Yaw applied BEFORE _SkyRotation, matching Skybox/Panoramic's _Rotation, so the
        // server's skyboxRotation keeps its existing meaning and terrain alignment.
        _Rotation ("Yaw (degrees)", Range(0, 360)) = 0
    }

    SubShader
    {
        Tags { "Queue"="Background" "RenderType"="Background" "PreviewType"="Skybox" }
        Cull Off ZWrite Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 2.0

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            float  _Exposure;
            float  _Rotation;
            // Set from script each frame (SkyboxSpin). Identity until then, so a missing
            // or un-driven material degrades to the static panorama rather than to noise.
            float4x4 _SkyRotation;

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 dirOS      : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float3 RotateAboutY(float3 v, float degrees)
            {
                float rad = radians(degrees);
                float s, c;
                sincos(rad, s, c);
                return float3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
            }

            Varyings Vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.dirOS = IN.positionOS.xyz;
                return OUT;
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float3 dir = normalize(IN.dirOS);
                dir = RotateAboutY(dir, _Rotation);
                dir = normalize(mul((float3x3)_SkyRotation, dir));

                // Matches Skybox/Panoramic's ToRadialCoords.
                float latitude  = acos(dir.y);
                float longitude = atan2(dir.z, dir.x);
                float2 uv = float2(0.5, 1.0) - float2(longitude, latitude) * float2(0.5 / PI, 1.0 / PI);

                // Explicit gradients: uv wraps at the longitude seam, and the implicit
                // derivative there spans the whole texture, which selects the smallest
                // mip and draws a visible vertical band down the join. Deriving the
                // gradients from the (continuous) direction vector instead avoids it.
                float3 ddxDir = ddx(dir);
                float3 ddyDir = ddy(dir);
                float2 duvdx = float2(-ddxDir.z * 0.5 / PI, ddxDir.y * 1.0 / PI);
                float2 duvdy = float2(-ddyDir.z * 0.5 / PI, ddyDir.y * 1.0 / PI);

                half4 color = SAMPLE_TEXTURE2D_GRAD(_MainTex, sampler_MainTex, uv, duvdx, duvdy);
                color.rgb *= _Exposure;
                return color;
            }
            ENDHLSL
        }
    }

    Fallback "Skybox/Panoramic"
}
