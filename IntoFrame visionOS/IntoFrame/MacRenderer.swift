import Metal
import MetalKit
import ModelIO
import simd

// Traditional Metal renderer for the macOS debug preview window.
// Uses the same shaders and uniform layout as the visionOS Renderer actor,
// but bound via setVertexBuffer/setFragmentTexture instead of MTL4 argument tables.
@MainActor
final class MacRenderer: NSObject, MTKViewDelegate {

    // MARK: - State

    struct PendingScene {
        let objects: [SceneObject]
        let assets: [String: Data]
        let params: SceneParams?
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let skyboxDepthState: MTLDepthStencilState
    let colorMap: MTLTexture
    var environmentTexture: MTLTexture
    var sceneAmbientColor: SIMD4<Float> = .init(1, 1, 1, 0)

    var sceneRenderables: [SceneRenderable] = []
    var skyboxRenderable: SceneRenderable?
    private var pendingScene: PendingScene?

    // Triple-buffered uniforms (same sizes as visionOS Renderer uses)
    private let uniformBuffers: [MTLBuffer]
    private let viewProjectionBuffers: [MTLBuffer]
    private var bufferIndex = 0
    private let frameSemaphore = DispatchSemaphore(value: 3)

    // Orbit camera
    var azimuth: Float = 0
    var elevation: Float = 0.3
    var radius: Float = 5.0
    var target: SIMD3<Float> = .zero
    var viewAspect: Float = 1.0


    // MARK: - Init

    init?(mtkView: MTKView) {
        guard let device = mtkView.device ?? MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        mtkView.device = device
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.clearColor = MTLClearColor(red: 0.08, green: 0.08, blue: 0.10, alpha: 1)
        mtkView.clearDepth = 1.0
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.preferredFramesPerSecond = 60

        let vtxDesc = Renderer.buildMetalVertexDescriptor()

        guard let library = device.makeDefaultLibrary(),
              let vertFn = library.makeFunction(name: "vertexShader"),
              let fragFn = library.makeFunction(name: "fragmentShader") else { return nil }

        let pipeDesc = MTLRenderPipelineDescriptor()
        pipeDesc.vertexFunction = vertFn
        pipeDesc.fragmentFunction = fragFn
        pipeDesc.vertexDescriptor = vtxDesc
        pipeDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        pipeDesc.depthAttachmentPixelFormat = mtkView.depthStencilPixelFormat
        pipeDesc.rasterSampleCount = mtkView.sampleCount
        pipeDesc.maxVertexAmplificationCount = 1

        guard let pipeline = try? device.makeRenderPipelineState(descriptor: pipeDesc) else { return nil }
        self.pipelineState = pipeline

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { return nil }
        self.depthState = ds

        let skyDepthDesc = MTLDepthStencilDescriptor()
        skyDepthDesc.depthCompareFunction = .always
        skyDepthDesc.isDepthWriteEnabled = false
        guard let sds = device.makeDepthStencilState(descriptor: skyDepthDesc) else { return nil }
        self.skyboxDepthState = sds

        let textureLoader = MTKTextureLoader(device: device)
        let texOptions: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
        ]
        guard let cm = try? textureLoader.newTexture(name: "ColorMap", scaleFactor: 1,
                                                      bundle: nil, options: texOptions) else { return nil }
        self.colorMap = cm
        self.environmentTexture = Renderer.makeDefaultEnvironmentTexture(device: device)

        // Triple-buffered uniforms + view-projection buffers
        let uniformsSize = alignedUniformsSize * maxUniformsPerFrame
        self.uniformBuffers = (0..<3).map { _ in
            device.makeBuffer(length: uniformsSize, options: .storageModeShared)!
        }
        self.viewProjectionBuffers = (0..<3).map { _ in
            device.makeBuffer(length: alignedViewProjectionArraySize, options: .storageModeShared)!
        }

        super.init()
    }

    // MARK: - Scene Loading

    func loadScene(objects: [SceneObject], assets: [String: Data], params: SceneParams?) {
        pendingScene = PendingScene(objects: objects, assets: assets, params: params)
    }

    // MARK: - Camera Input

    func handleMouseDrag(deltaX: Float, deltaY: Float) {
        azimuth   += deltaX * 0.002
        elevation  = max(-.pi / 2 + 0.01, min(.pi / 2 - 0.01, elevation - deltaY * 0.002))
    }

    func handleScroll(delta: Float) {
        radius = max(0.5, min(1000, radius - delta * 0.5))
    }

    func handleTranslate(dx: Float, dz: Float) {
        let step = max(0.1, radius * 0.03)
        let right   = SIMD3<Float>( cos(azimuth), 0, -sin(azimuth))
        let forward = SIMD3<Float>(-sin(azimuth), 0, -cos(azimuth))
        target += right * (dx * step) + forward * (dz * step)
    }

    // MARK: - MTKViewDelegate

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        MainActor.assumeIsolated {
            viewAspect = Float(size.width / max(size.height, 1))
        }
    }

    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            renderFrame(in: view)
        }
    }

    // MARK: - Render

    private func renderFrame(in view: MTKView) {
        frameSemaphore.wait()

        if let pending = pendingScene {
            pendingScene = nil
            processPendingScene(pending)
        }

        bufferIndex = (bufferIndex + 1) % 3
        updateViewProjection()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else {
            frameSemaphore.signal()
            return
        }

        let semaphore = frameSemaphore
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            commandBuffer.commit()
            return
        }
        encoder.label = "Mac Preview"
        encoder.setFrontFacing(.counterClockwise)
        encoder.setRenderPipelineState(pipelineState)

        let vpBuf = viewProjectionBuffers[bufferIndex]
        let uBuf  = uniformBuffers[bufferIndex]

        encoder.setVertexBuffer(vpBuf, offset: 0, index: BufferIndex.viewProjection.rawValue)
        encoder.setFragmentTexture(environmentTexture, index: TextureIndex.environment.rawValue)

        let hasScene = !sceneRenderables.isEmpty || skyboxRenderable != nil

        if hasScene {
            if let skybox = skyboxRenderable {
                encoder.pushDebugGroup("Skybox")
                encoder.setCullMode(.none)
                encoder.setDepthStencilState(skyboxDepthState)

                writeUniforms(to: uBuf, at: 0, modelMatrix: skybox.modelMatrix,
                              ambientColor: SIMD4<Float>(1, 1, 1, 0))
                encoder.setVertexBuffer(uBuf, offset: 0, index: BufferIndex.uniforms.rawValue)
                encoder.setFragmentTexture(skybox.texture, index: TextureIndex.color.rawValue)
                bindAndDraw(mesh: skybox.mesh, encoder: encoder)
                encoder.popDebugGroup()
            }

            encoder.setCullMode(.back)
            encoder.setDepthStencilState(depthState)

            for (i, renderable) in sceneRenderables.enumerated() {
                let offset = (i + 1) * alignedUniformsSize
                writeUniforms(to: uBuf, at: offset, modelMatrix: renderable.modelMatrix,
                              ambientColor: sceneAmbientColor)
                encoder.setVertexBuffer(uBuf, offset: offset, index: BufferIndex.uniforms.rawValue)
                encoder.setFragmentTexture(renderable.texture, index: TextureIndex.color.rawValue)
                bindAndDraw(mesh: renderable.mesh, encoder: encoder)
            }
        }

        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Helpers

    private func updateViewProjection() {
        let vp = viewProjectionBuffers[bufferIndex].contents()
            .bindMemory(to: ViewProjectionArray.self, capacity: 1)
        vp[0].viewProjectionMatrix.0 = projectionMatrix() * viewMatrix()
    }

    private func viewMatrix() -> float4x4 {
        let x = target.x + radius * cos(elevation) * sin(azimuth)
        let y = target.y + radius * sin(elevation)
        let z = target.z + radius * cos(elevation) * cos(azimuth)
        return matrix4x4_lookAt(eye: SIMD3<Float>(x, y, z),
                                 target: target,
                                 up: SIMD3<Float>(0, 1, 0))
    }

    private func projectionMatrix() -> float4x4 {
        matrix4x4_perspectiveFov(fovyRadians: .pi / 3,
                                  aspectRatio: max(viewAspect, 0.01),
                                  nearZ: 0.1, farZ: 2500)
    }

    private func writeUniforms(to buffer: MTLBuffer, at offset: Int,
                                modelMatrix: float4x4, ambientColor: SIMD4<Float>) {
        let ptr = (buffer.contents() + offset).bindMemory(to: Uniforms.self, capacity: 1)
        ptr[0].modelMatrix   = modelMatrix
        ptr[0].ambientColor  = ambientColor
    }

    private func bindAndDraw(mesh: MTKMesh, encoder: MTLRenderCommandEncoder) {
        for (index, vertexBuffer) in mesh.vertexBuffers.enumerated() {
            encoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
        }
        for submesh in mesh.submeshes {
            encoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                          indexCount: submesh.indexCount,
                                          indexType: submesh.indexType,
                                          indexBuffer: submesh.indexBuffer.buffer,
                                          indexBufferOffset: submesh.indexBuffer.offset)
        }
    }

    // MARK: - Scene Processing

    private func processPendingScene(_ data: PendingScene) {
        let vtxDesc = Renderer.buildMetalVertexDescriptor()
        let mdlDesc = MTKModelIOVertexDescriptorFromMetal(vtxDesc)
        guard let attrs = mdlDesc.attributes as? [MDLVertexAttribute] else { return }
        attrs[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attrs[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        attrs[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal

        let textureLoader = MTKTextureLoader(device: device)
        let texOptions: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue)
        ]

        print("[MacRenderer] processPendingScene: \(data.objects.count) objects, \(data.assets.count) assets [\(data.assets.keys.joined(separator: ", "))]")

        var newRenderables: [SceneRenderable] = []
        for obj in data.objects {
            let objMesh: MTKMesh
            var meshTexture: MTLTexture? = nil

            if obj.type == "billboard" {
                guard let m = makeBillboardMesh(vertexDescriptor: mdlDesc) else {
                    print("[MacRenderer] billboard mesh creation failed for \(obj.id)")
                    continue
                }
                objMesh = m
            } else if let meshName = obj.mesh, !meshName.isEmpty, let meshData = data.assets[meshName] {
                guard let result = loadMeshFromData(meshData, name: meshName,
                                                    vertexDescriptor: mdlDesc) else {
                    print("[MacRenderer] mesh load failed for \(meshName) (type=\(obj.type))")
                    continue
                }
                objMesh = result.mesh
                meshTexture = result.texture
            } else {
                let meshVal = obj.mesh ?? "nil"
                let hasAsset = obj.mesh.flatMap { data.assets[$0] } != nil
                print("[MacRenderer] skipping obj \(obj.id) type=\(obj.type) mesh='\(meshVal)' assetPresent=\(hasAsset)")
                continue
            }

            let objTexture: MTLTexture
            if let tex = meshTexture {
                objTexture = tex
            } else if let texName = obj.texture, !texName.isEmpty, let texData = data.assets[texName] {
                objTexture = (try? textureLoader.newTexture(data: texData, options: texOptions)) ?? colorMap
            } else {
                objTexture = colorMap
            }
            newRenderables.append(SceneRenderable(mesh: objMesh, texture: objTexture,
                                                   modelMatrix: sceneObjectModelMatrix(obj)))
        }
        print("[MacRenderer] built \(newRenderables.count) renderables")

        var newSkybox: SceneRenderable?
        if let skyboxName = data.params?.skybox, let skyboxData = data.assets[skyboxName],
           let skyboxTex = try? textureLoader.newTexture(data: skyboxData, options: texOptions),
           let skyboxMesh = makeSkyboxMesh(vertexDescriptor: mdlDesc) {
            let rotDeg = data.params?.skyboxRotation ?? 0
            let skyboxMatrix = matrix4x4_rotation(radians: rotDeg * .pi / 180,
                                                   axis: SIMD3<Float>(0, 1, 0))
            newSkybox = SceneRenderable(mesh: skyboxMesh, texture: skyboxTex,
                                        modelMatrix: skyboxMatrix)
        }

        var newAmbient = SIMD3<Float>(1, 1, 1)
        if let hex = data.params?.ambientColor, let parsed = Renderer.parseHexColor(hex) {
            newAmbient = parsed
        }

        var newEnvTex: MTLTexture? = nil
        if let ldr = data.params?.lighting?.ldr, let imgData = Data(base64Encoded: ldr) {
            let envOptions: [MTKTextureLoader.Option: Any] = [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue),
                .SRGB: NSNumber(value: false)
            ]
            newEnvTex = try? textureLoader.newTexture(data: imgData, options: envOptions)
        }

        sceneRenderables = newRenderables
        skyboxRenderable = newSkybox
        sceneAmbientColor = SIMD4<Float>(newAmbient.x, newAmbient.y, newAmbient.z,
                                         newEnvTex != nil ? 1.0 : 0.0)
        if let tex = newEnvTex { environmentTexture = tex }
    }

    private func loadMeshFromData(_ data: Data, name: String,
                                   vertexDescriptor: MDLVertexDescriptor) -> (mesh: MTKMesh, texture: MTLTexture?)? {
        let ext = (name as NSString).pathExtension.isEmpty ? "usdz" : (name as NSString).pathExtension
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "." + ext)
        do {
            try data.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let allocator = MTKMeshBufferAllocator(device: device)
            let asset = MDLAsset(url: url, vertexDescriptor: vertexDescriptor, bufferAllocator: allocator)
            asset.loadTextures()
            guard let mdlMesh = asset.childObjects(of: MDLMesh.self).first as? MDLMesh else { return nil }
            mdlMesh.vertexDescriptor = vertexDescriptor
            let mtkMesh = try MTKMesh(mesh: mdlMesh, device: device)
            let texture = Renderer.extractTexture(from: mdlMesh, device: device)
            return (mesh: mtkMesh, texture: texture)
        } catch { return nil }
    }

    private func makeBillboardMesh(vertexDescriptor: MDLVertexDescriptor) -> MTKMesh? {
        let alloc = MTKMeshBufferAllocator(device: device)
        let mdl = MDLMesh(planeWithExtent: SIMD3<Float>(1, 1, 0),
                          segments: SIMD2<UInt32>(1, 1), geometryType: .triangles, allocator: alloc)
        mdl.vertexDescriptor = vertexDescriptor
        return try? MTKMesh(mesh: mdl, device: device)
    }

    private func makeSkyboxMesh(vertexDescriptor: MDLVertexDescriptor) -> MTKMesh? {
        let alloc = MTKMeshBufferAllocator(device: device)
        let mdl = MDLMesh(sphereWithExtent: SIMD3<Float>(1800, 1800, 1800),
                          segments: SIMD2<UInt32>(64, 32), inwardNormals: true,
                          geometryType: .triangles, allocator: alloc)
        mdl.vertexDescriptor = vertexDescriptor
        return try? MTKMesh(mesh: mdl, device: device)
    }
}

// MARK: - Camera math

private func matrix4x4_lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let z = normalize(eye - target)
    let x = normalize(cross(up, z))
    let y = cross(z, x)
    return .init(columns: (
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
    ))
}

private func matrix4x4_perspectiveFov(fovyRadians fovy: Float,
                                       aspectRatio: Float,
                                       nearZ: Float, farZ: Float) -> float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return .init(columns: (
        SIMD4<Float>(xs,  0,  0,  0),
        SIMD4<Float>( 0, ys,  0,  0),
        SIMD4<Float>( 0,  0, zs, -1),
        SIMD4<Float>( 0,  0, nearZ * zs, 0)
    ))
}
