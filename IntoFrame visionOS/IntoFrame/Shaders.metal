//
//  Shaders.metal
//  InfoFrame
//
//  Created by Josh Ford on 5/21/26.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

// Metal has no built-in inverse() for matrices
float3x3 inverse3x3(float3x3 m) {
    float3 a = m[0], b = m[1], c = m[2];
    float det = dot(a, cross(b, c));
    float3 row0 = cross(b, c) / det;
    float3 row1 = cross(c, a) / det;
    float3 row2 = cross(a, b) / det;
    return float3x3(float3(row0.x, row1.x, row2.x),
                    float3(row0.y, row1.y, row2.y),
                    float3(row0.z, row1.z, row2.z));
}

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
    float3 worldNormal;
    float4 ambientColor [[flat]];  // rgb = ambient, a = env strength (0=unlit)
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               ushort amp_id [[amplification_id]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               constant ViewProjectionArray & viewProjectionArray [[ buffer(BufferIndexViewProjection) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = viewProjectionArray.viewProjectionMatrix[amp_id] * uniforms.modelMatrix * position;
    out.texCoord = in.texCoord;

    // Inverse-transpose of the upper-left 3x3 for correct non-uniform scale handling
    float3x3 modelMat3 = float3x3(uniforms.modelMatrix[0].xyz,
                                   uniforms.modelMatrix[1].xyz,
                                   uniforms.modelMatrix[2].xyz);
    out.worldNormal = normalize(transpose(inverse3x3(modelMat3)) * in.normal);
    out.ambientColor = uniforms.ambientColor;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               texture2d<half> colorMap [[ texture(TextureIndexColor) ]],
                               texture2d<half> envMap   [[ texture(TextureIndexEnvironment) ]])
{
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    constexpr sampler envSampler(mip_filter::linear, mag_filter::linear, min_filter::linear,
                                  address::repeat);

    half4 colorSample = colorMap.sample(colorSampler, in.texCoord.xy);

    float3 albedo    = float3(colorSample.rgb);
    float3 ambientRGB = float3(in.ambientColor.rgb);
    float  envStrength = in.ambientColor.a;

    float3 lighting;
    if (envStrength > 0.0) {
        float3 n = normalize(in.worldNormal);
        // Equirectangular UV from world-space normal (Y-up)
        float u = atan2(n.z, n.x) * (1.0 / (2.0 * M_PI_F)) + 0.5;
        float v = 0.5 - asin(clamp(n.y, -1.0f, 1.0f)) * (1.0 / M_PI_F);
        float3 envLight = float3(envMap.sample(envSampler, float2(u, v)).rgb);
        // Blend: full env-lit at envStrength=1, pure ambient at envStrength=0
        lighting = ambientRGB * (envLight * envStrength + (1.0 - envStrength));
    } else {
        // Unlit — skybox or no env map loaded
        lighting = ambientRGB;
    }

    return float4(albedo * lighting, colorSample.a);
}
