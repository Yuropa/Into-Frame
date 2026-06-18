//
//  ShaderTypes.h
//  InfoFrame
//
//  Created by Josh Ford on 5/21/26.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions  = 0,
    BufferIndexMeshGenerics   = 1,
    BufferIndexMeshNormals    = 2,
    BufferIndexUniforms       = 3,
    BufferIndexViewProjection = 4,
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition   = 0,
    VertexAttributeTexcoord   = 1,
    VertexAttributeNormal     = 2,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor         = 0,
    TextureIndexEnvironment   = 1,
};

typedef struct
{
    matrix_float4x4 viewProjectionMatrix[2];
} ViewProjectionArray;

typedef struct
{
    matrix_float4x4 modelMatrix;
    // rgb = ambient color, a = env map strength (0 = unlit/skybox, 1 = IBL)
    vector_float4   ambientColor;
} Uniforms;

#endif /* ShaderTypes_h */

