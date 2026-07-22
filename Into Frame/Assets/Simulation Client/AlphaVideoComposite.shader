// Off-screen compositing shader used only via Graphics.Blit (see
// AnimatedBillboardVideo.cs) to merge a color video's RGB with a separate
// grayscale matte video's red channel into one RGBA texture, which is then
// fed into the billboard's normal URP Lit material as _BaseMap -- so the
// billboard's actual rendering, alpha-cutout, lighting, etc. are entirely
// unchanged from the static-texture path.
Shader "Hidden/IntoFrame/AlphaVideoComposite"
{
    Properties
    {
        _ColorTex ("Color", 2D) = "black" {}
        _AlphaTex ("Alpha", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_ColorTex);
            SAMPLER(sampler_ColorTex);
            TEXTURE2D(_AlphaTex);
            SAMPLER(sampler_AlphaTex);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                half3 color = SAMPLE_TEXTURE2D(_ColorTex, sampler_ColorTex, IN.uv).rgb;
                half alpha = SAMPLE_TEXTURE2D(_AlphaTex, sampler_AlphaTex, IN.uv).r;
                return half4(color, alpha);
            }
            ENDHLSL
        }
    }
}
