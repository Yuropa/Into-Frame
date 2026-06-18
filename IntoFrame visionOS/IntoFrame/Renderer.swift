import CompositorServices
import Metal
import MetalKit
import simd
import os

nonisolated let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100
nonisolated let alignedViewProjectionArraySize = (MemoryLayout<ViewProjectionArray>.size + 0xFF) & -0x100

nonisolated let maxBuffersInFlight = 3
nonisolated let maxRenderableObjects = 128
nonisolated let maxUniformsPerFrame = maxRenderableObjects + 1

enum RendererError: Error {
    case badVertexDescriptor
}

struct SceneRenderable {
    let mesh: MTKMesh
    let texture: MTLTexture
    var modelMatrix: float4x4
}

struct PendingSceneData: Sendable {
    let objects: [SceneObject]
    let assets: [String: Data]
    let params: SceneParams?
}

extension MTLDevice {
    nonisolated var supportsMSAA: Bool {
        supports32BitMSAA && supportsTextureSampleCount(4)
    }

    nonisolated var rasterSampleCount: Int {
        supportsMSAA ? 4 : 1
    }
}

extension LayerRenderer.Clock.Instant {
    nonisolated var timeInterval: TimeInterval {
        let components = LayerRenderer.Clock.Instant.epoch.duration(to: self).components
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

final class RendererTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
          job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    nonisolated func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        return UnownedTaskExecutor(ordinary: self)
    }

    static var shared: RendererTaskExecutor = RendererTaskExecutor()
}

actor Renderer {

    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let vertexArgumentTable: MTL4ArgumentTable
    let fragmentArgumentTable: MTL4ArgumentTable
    #if !targetEnvironment(simulator)
    let residencySets: [MTLResidencySet]
    let commandQueueResidencySet: MTLResidencySet
    #endif

    let dynamicUniformBuffer: MTLBuffer
    let pipelineState: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let skyboxDepthState: MTLDepthStencilState
    let colorMap: MTLTexture
    var environmentTexture: MTLTexture
    var sceneAmbientColor: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 0)

    let endFrameEvent: MTLSharedEvent
    var committedFrameIndex: UInt64 = 0

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var perDrawableTarget = [LayerRenderer.Drawable.Target: DrawableTarget]()

    var sceneRenderables: [SceneRenderable] = []
    var skyboxRenderable: SceneRenderable?
    nonisolated(unsafe) let pendingSceneLock = OSAllocatedUnfairLock<PendingSceneData?>(initialState: nil)
    private var retiredRenderables: [([SceneRenderable], SceneRenderable?, UInt64)] = []

    let worldTracking: WorldTrackingProvider
    let layerRenderer: LayerRenderer
    let appModel: AppModel

    init(_ layerRenderer: LayerRenderer, appModel: AppModel) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.appModel = appModel

        let device = self.device
        self.commandQueue = layerRenderer.commandQueue
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0...maxBuffersInFlight).map { _ in device.makeCommandAllocator()! }

        let argTableDesc = MTL4ArgumentTableDescriptor()
        argTableDesc.maxBufferBindCount = 5
        self.vertexArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)
        argTableDesc.maxBufferBindCount = 0
        argTableDesc.maxTextureBindCount = 2
        self.fragmentArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)

        #if !targetEnvironment(simulator)
        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = 3
        self.residencySets = (0...maxBuffersInFlight).map { _ in try! device.makeResidencySet(descriptor: residencySetDesc) }
        #endif

        self.endFrameEvent = device.makeSharedEvent()!
        self.endFrameEvent.signaledValue = UInt64(maxBuffersInFlight)
        committedFrameIndex = UInt64(maxBuffersInFlight)

        let uniformBufferSize = alignedUniformsSize * maxUniformsPerFrame * maxBuffersInFlight

        self.dynamicUniformBuffer = self.device.makeBuffer(length: uniformBufferSize,
                                                           options: [MTLResourceOptions.storageModeShared])!

        self.dynamicUniformBuffer.label = "UniformBuffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to: Uniforms.self, capacity: 1)

        let mtlVertexDescriptor = Self.buildMetalVertexDescriptor()

        do {
            pipelineState = try Self.buildRenderPipeline(device: device,
                                                         layerRenderer: layerRenderer,
                                                         mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }

        self.depthState = Self.buildDepthStencilState(device: device)

        let skyboxDepthDesc = MTLDepthStencilDescriptor()
        skyboxDepthDesc.depthCompareFunction = .always
        skyboxDepthDesc.isDepthWriteEnabled = false
        self.skyboxDepthState = device.makeDepthStencilState(descriptor: skyboxDepthDesc)!

        do {
            colorMap = try Self.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            fatalError("Unable to load texture. Error info: \(error)")
        }

        environmentTexture = Self.makeDefaultEnvironmentTexture(device: device)

        #if !targetEnvironment(simulator)
        residencySetDesc.initialCapacity = 3
        let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
        residencySet.addAllocations([colorMap, dynamicUniformBuffer, environmentTexture])
        residencySet.commit()
        commandQueueResidencySet = residencySet
        commandQueue.addResidencySet(residencySet)
        #endif

        worldTracking = WorldTrackingProvider()
    }

    private func startARSession(_ arSession: ARKitSession) async {
        do {
            try await arSession.run([worldTracking])
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }

    @MainActor
    static func startRenderLoop(_ layerRenderer: LayerRenderer, appModel: AppModel, arSession: ARKitSession) {
        Task(executorPreference: RendererTaskExecutor.shared) {
            let renderer = Renderer(layerRenderer, appModel: appModel)
            Task { @MainActor in appModel.renderer = renderer }
            await renderer.startARSession(arSession)
            await renderer.renderLoop()
        }
    }

    static func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.meshNormals.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    static func buildRenderPipeline(device: MTLDevice,
                                    layerRenderer: LayerRenderer,
                                    mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.rasterSampleCount = device.rasterSampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    static func buildDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptor.isDepthWriteEnabled = true
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    static func loadTexture(device: MTLDevice,
                            textureName: String) throws -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]

        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
    }

    private func updateDynamicBufferState(frameIndex: UInt64) {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * maxUniformsPerFrame * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to: Uniforms.self, capacity: 1)

        #if !targetEnvironment(simulator)
        residencySets[uniformBufferIndex].removeAllAllocations()
        residencySets[uniformBufferIndex].commit()
        #endif
        commandAllocators[uniformBufferIndex].reset()

        perDrawableTarget = perDrawableTarget.filter { $0.value.lastUsedFrameIndex + 90 > frameIndex }
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        guard self.endFrameEvent.wait(untilSignaledValue: committedFrameIndex - UInt64(maxBuffersInFlight), timeoutMS: 10000) else {
            return
        }

        retiredRenderables.removeAll { endFrameEvent.signaledValue >= $0.2 }

        if let pending = pendingSceneLock.withLock({ let v = $0; $0 = nil; return v }) {
            processPendingScene(pending)
        }

        frame.startUpdate()

        self.updateDynamicBufferState(frameIndex: frame.frameIndex)

        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }

        frame.startSubmission()

        for drawable in drawables {
            render(drawable: drawable, frameIndex: frame.frameIndex)
        }

        committedFrameIndex += 1

        commandQueue.signalEvent(self.endFrameEvent, value: committedFrameIndex)

        frame.endSubmission()
    }

    func render(drawable: LayerRenderer.Drawable, frameIndex: UInt64) {
        let time = drawable.frameTiming.presentationTime.timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        if perDrawableTarget[drawable.target] == nil {
            perDrawableTarget[drawable.target] = .init(drawable: drawable)
        }
        let drawableTarget = perDrawableTarget[drawable.target]!

        drawableTarget.updateBufferState(uniformBufferIndex: uniformBufferIndex, frameIndex: frameIndex)

        drawableTarget.updateViewProjectionArray(drawable: drawable)

        let renderPassDescriptor = MTL4RenderPassDescriptor()

        if device.supportsMSAA {
            let renderTargets = drawableTarget.memorylessTargets[uniformBufferIndex]
            assert(renderTargets.color.width == drawable.colorTextures[0].width)
            assert(renderTargets.color.height == drawable.colorTextures[0].height)

            renderPassDescriptor.colorAttachments[0].resolveTexture = drawable.colorTextures[0]
            renderPassDescriptor.colorAttachments[0].texture = renderTargets.color
            renderPassDescriptor.depthAttachment.resolveTexture = drawable.depthTextures[0]
            renderPassDescriptor.depthAttachment.texture = renderTargets.depth

            renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve
            renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
        } else {
            renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
            renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]

            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.depthAttachment.storeAction = .store
        }

        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
        if layerRenderer.configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = drawable.views.count
        }

        #if !targetEnvironment(simulator)
        let residencySet = self.residencySets[uniformBufferIndex]
        residencySet.addAllocations([
            drawable.colorTextures[0],
            drawable.depthTextures[0],
            drawableTarget.viewProjectionBuffer,
            environmentTexture
        ])

        for renderable in sceneRenderables {
            residencySet.addAllocations(renderable.mesh.vertexBuffers.map { $0.buffer })
            residencySet.addAllocations(renderable.mesh.submeshes.map { $0.indexBuffer.buffer })
            residencySet.addAllocations([renderable.texture])
        }
        if let skybox = skyboxRenderable {
            residencySet.addAllocations(skybox.mesh.vertexBuffers.map { $0.buffer })
            residencySet.addAllocations(skybox.mesh.submeshes.map { $0.indexBuffer.buffer })
            residencySet.addAllocations([skybox.texture])
        }

        residencySet.commit()
        #endif

        let commandAllocator = self.commandAllocators[uniformBufferIndex]
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        commandBuffer.useResidencySet(residencySet)

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.setFrontFacing(.counterClockwise)

        renderEncoder.setRenderPipelineState(pipelineState)

        let viewports = drawable.views.map { $0.textureMap.viewport }

        renderEncoder.setViewports(viewports)

        if drawable.views.count > 1 {
            let viewMappings = (0..<drawable.views.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewMappings)
        }

        renderEncoder.setArgumentTable(self.vertexArgumentTable, stages: .vertex)
        renderEncoder.setArgumentTable(self.fragmentArgumentTable, stages: .fragment)

        self.vertexArgumentTable.setAddress(drawableTarget.viewProjectionBuffer.gpuAddress + UInt64(drawableTarget.viewProjectionBufferOffset), index: BufferIndex.viewProjection.rawValue)
        self.fragmentArgumentTable.setTexture(environmentTexture.gpuResourceID, index: TextureIndex.environment.rawValue)

        let hasScene = !sceneRenderables.isEmpty || skyboxRenderable != nil

        if hasScene {
            if let skybox = skyboxRenderable {
                renderEncoder.pushDebugGroup("Skybox")
                renderEncoder.setCullMode(.none)
                renderEncoder.setDepthStencilState(skyboxDepthState)

                let ptr = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset)
                    .bindMemory(to: Uniforms.self, capacity: 1)
                ptr.pointee.modelMatrix = skybox.modelMatrix
                ptr.pointee.ambientColor = SIMD4<Float>(1, 1, 1, 0)

                self.vertexArgumentTable.setAddress(
                    dynamicUniformBuffer.gpuAddress + UInt64(uniformBufferOffset),
                    index: BufferIndex.uniforms.rawValue)
                self.fragmentArgumentTable.setTexture(skybox.texture.gpuResourceID, index: TextureIndex.color.rawValue)

                for (index, element) in skybox.mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout, layout.stride != 0 else { continue }
                    let buffer = skybox.mesh.vertexBuffers[index]
                    self.vertexArgumentTable.setAddress(buffer.buffer.gpuAddress + UInt64(buffer.offset), index: index)
                }

                for submesh in skybox.mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(primitiveType: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer.gpuAddress + UInt64(submesh.indexBuffer.offset),
                                                        indexBufferLength: submesh.indexBuffer.buffer.length)
                }

                renderEncoder.popDebugGroup()
            }

            renderEncoder.setCullMode(.back)
            renderEncoder.setDepthStencilState(depthState)

            for (i, renderable) in sceneRenderables.enumerated() {
                let objOffset = uniformBufferOffset + (i + 1) * alignedUniformsSize
                let ptr = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + objOffset)
                    .bindMemory(to: Uniforms.self, capacity: 1)
                ptr.pointee.modelMatrix = renderable.modelMatrix
                ptr.pointee.ambientColor = sceneAmbientColor

                self.vertexArgumentTable.setAddress(
                    dynamicUniformBuffer.gpuAddress + UInt64(objOffset),
                    index: BufferIndex.uniforms.rawValue)
                self.fragmentArgumentTable.setTexture(renderable.texture.gpuResourceID, index: TextureIndex.color.rawValue)

                for (index, element) in renderable.mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout, layout.stride != 0 else { continue }
                    let buffer = renderable.mesh.vertexBuffers[index]
                    self.vertexArgumentTable.setAddress(buffer.buffer.gpuAddress + UInt64(buffer.offset), index: index)
                }

                for submesh in renderable.mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(primitiveType: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer.gpuAddress + UInt64(submesh.indexBuffer.offset),
                                                        indexBufferLength: submesh.indexBuffer.buffer.length)
                }
            }
        }

        renderEncoder.endEncoding()

        commandBuffer.endCommandBuffer()

        self.commandQueue.commit([commandBuffer])

        drawable.encodePresent()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                Task { @MainActor in
                    appModel.immersiveSpaceState = .closed
                }
                return
            } else if layerRenderer.state == .paused {
                Task { @MainActor in
                    appModel.immersiveSpaceState = .inTransition
                }
                layerRenderer.waitUntilRunning()
                continue
            } else {
                Task { @MainActor in
                    if appModel.immersiveSpaceState != .open {
                        appModel.immersiveSpaceState = .open
                    }
                }
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }

    // MARK: - Scene Loading

    nonisolated func loadScene(objects: [SceneObject], assets: [String: Data], params: SceneParams?) {
        pendingSceneLock.withLock { $0 = PendingSceneData(objects: objects, assets: assets, params: params) }
    }

    private func processPendingScene(_ data: PendingSceneData) {
        let mtlVertexDescriptor = Self.buildMetalVertexDescriptor()
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else { return }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        attributes[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal

        let textureLoader = MTKTextureLoader(device: device)
        let textureOptions: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue)
        ]

        var newRenderables: [SceneRenderable] = []

        for obj in data.objects {
            let objMesh: MTKMesh
            var meshTexture: MTLTexture? = nil

            if obj.type == "billboard" {
                guard let m = createBillboardMesh(vertexDescriptor: mdlVertexDescriptor) else { continue }
                objMesh = m
            } else if let meshName = obj.mesh, !meshName.isEmpty, let meshData = data.assets[meshName] {
                guard let result = loadMeshFromData(meshData, name: meshName, vertexDescriptor: mdlVertexDescriptor) else { continue }
                objMesh = result.mesh
                meshTexture = result.texture
            } else {
                continue
            }

            let objTexture: MTLTexture
            if let tex = meshTexture {
                objTexture = tex
            } else if let texName = obj.texture, !texName.isEmpty, let texData = data.assets[texName] {
                objTexture = (try? textureLoader.newTexture(data: texData, options: textureOptions)) ?? colorMap
            } else {
                objTexture = colorMap
            }

            newRenderables.append(SceneRenderable(
                mesh: objMesh,
                texture: objTexture,
                modelMatrix: computeModelMatrix(obj)
            ))
        }

        var newSkybox: SceneRenderable?
        if let skyboxName = data.params?.skybox, let skyboxData = data.assets[skyboxName] {
            if let skyboxTexture = try? textureLoader.newTexture(data: skyboxData, options: textureOptions),
               let skyboxMesh = createSkyboxMesh(vertexDescriptor: mdlVertexDescriptor) {
                let rotDeg = data.params?.skyboxRotation ?? 0
                let skyboxMatrix = matrix4x4_rotation(radians: rotDeg * .pi / 180,
                                                       axis: SIMD3<Float>(0, 1, 0))
                newSkybox = SceneRenderable(mesh: skyboxMesh, texture: skyboxTexture, modelMatrix: skyboxMatrix)
            }
        }

        // Parse ambient color from params (default: white)
        var newAmbient = SIMD3<Float>(1, 1, 1)
        if let hexStr = data.params?.ambientColor,
           let parsed = Self.parseHexColor(hexStr) {
            newAmbient = parsed
        }

        // Decode environment lighting map if present
        var newEnvTexture: MTLTexture? = nil
        if let ldrBase64 = data.params?.lighting?.ldr,
           let imgData = Data(base64Encoded: ldrBase64) {
            let envOptions: [MTKTextureLoader.Option: Any] = [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue),
                .SRGB: NSNumber(value: false)
            ]
            newEnvTexture = try? textureLoader.newTexture(data: imgData, options: envOptions)
        }

        if !sceneRenderables.isEmpty || skyboxRenderable != nil {
            retiredRenderables.append((sceneRenderables, skyboxRenderable, committedFrameIndex))
        }

        sceneRenderables = newRenderables
        skyboxRenderable = newSkybox

        // Update ambient color: enable env map IBL (a=1) only if we loaded one
        let envStrength: Float = newEnvTexture != nil ? 1.0 : 0.0
        sceneAmbientColor = SIMD4<Float>(newAmbient.x, newAmbient.y, newAmbient.z, envStrength)
        if let tex = newEnvTexture {
            environmentTexture = tex
        }
    }

    static func parseHexColor(_ hex: String) -> SIMD3<Float>? {
        let h = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
        guard h.count == 6, let value = UInt32(h, radix: 16) else { return nil }
        return SIMD3<Float>(
            Float((value >> 16) & 0xFF) / 255.0,
            Float((value >> 8)  & 0xFF) / 255.0,
            Float( value        & 0xFF) / 255.0
        )
    }

    static func makeDefaultEnvironmentTexture(device: MTLDevice) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        desc.usage = .shaderRead
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)!
        var pixel: [UInt8] = [255, 255, 255, 255]
        tex.replace(region: MTLRegionMake2D(0, 0, 1, 1),
                    mipmapLevel: 0, withBytes: &pixel, bytesPerRow: 4)
        return tex
    }

    private func loadMeshFromData(_ data: Data, name: String, vertexDescriptor: MDLVertexDescriptor) -> (mesh: MTKMesh, texture: MTLTexture?)? {
        let nameExt = (name as NSString).pathExtension
        let ext = nameExt.isEmpty ? Self.detectMeshExtension(from: data) : nameExt
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + "." + ext)
        do {
            try data.write(to: tempURL)
            defer { try? FileManager.default.removeItem(at: tempURL) }

            let allocator = MTKMeshBufferAllocator(device: device)
            let asset = MDLAsset(url: tempURL, vertexDescriptor: vertexDescriptor, bufferAllocator: allocator)
            asset.loadTextures()
            guard let mdlMesh = asset.childObjects(of: MDLMesh.self).first as? MDLMesh else { return nil }
            mdlMesh.vertexDescriptor = vertexDescriptor
            let mtkMesh = try MTKMesh(mesh: mdlMesh, device: device)
            let texture = Self.extractTexture(from: mdlMesh, device: device)
            return (mesh: mtkMesh, texture: texture)
        } catch {
            return nil
        }
    }

    static func detectMeshExtension(from data: Data) -> String {
        // GLB binary magic: "glTF" = 0x67 0x6C 0x54 0x46
        if data.count >= 4 && data[0] == 0x67 && data[1] == 0x6C && data[2] == 0x54 && data[3] == 0x46 {
            return "glb"
        }
        return "usdz"
    }

    static func extractTexture(from mdlMesh: MDLMesh, device: MTLDevice) -> MTLTexture? {
        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue),
            .SRGB: NSNumber(value: true)
        ]
        guard let submeshes = mdlMesh.submeshes else { return nil }
        for case let sub as MDLSubmesh in submeshes {
            guard let mat = sub.material else { continue }
            for semantic in [MDLMaterialSemantic.baseColor, .emission, .ambientOcclusion] {
                guard let prop = mat.property(with: semantic),
                      prop.type == MDLMaterialPropertyType.texture,
                      let sampler = prop.textureSamplerValue,
                      let mdlTex = sampler.texture,
                      let tex = try? loader.newTexture(texture: mdlTex, options: options) else { continue }
                return tex
            }
        }
        return nil
    }

    private func createBillboardMesh(vertexDescriptor: MDLVertexDescriptor) -> MTKMesh? {
        let allocator = MTKMeshBufferAllocator(device: device)
        let mdlMesh = MDLMesh(planeWithExtent: SIMD3<Float>(1, 1, 0),
                              segments: SIMD2<UInt32>(1, 1),
                              geometryType: .triangles,
                              allocator: allocator)
        mdlMesh.vertexDescriptor = vertexDescriptor
        return try? MTKMesh(mesh: mdlMesh, device: device)
    }

    private func createSkyboxMesh(vertexDescriptor: MDLVertexDescriptor) -> MTKMesh? {
        let allocator = MTKMeshBufferAllocator(device: device)
        let mdlMesh = MDLMesh(sphereWithExtent: SIMD3<Float>(500, 500, 500),
                              segments: SIMD2<UInt32>(64, 32),
                              inwardNormals: true,
                              geometryType: .triangles,
                              allocator: allocator)
        mdlMesh.vertexDescriptor = vertexDescriptor
        return try? MTKMesh(mesh: mdlMesh, device: device)
    }

    private func computeModelMatrix(_ obj: SceneObject) -> float4x4 {
        sceneObjectModelMatrix(obj)
    }
}

extension Renderer {
    class DrawableTarget {
        var lastUsedFrameIndex: UInt64

        let memorylessTargets: [(color: MTLTexture, depth: MTLTexture)]

        let viewProjectionBuffer: MTLBuffer

        var viewProjectionBufferOffset = 0

        var viewProjectionArray: UnsafeMutablePointer<ViewProjectionArray>

        nonisolated init(drawable: LayerRenderer.Drawable) {
            lastUsedFrameIndex = 0

            let device = drawable.colorTextures[0].device
            nonisolated func renderTarget(resolveTexture: MTLTexture) -> MTLTexture {
                assert(device.supportsMSAA)

                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: resolveTexture.pixelFormat,
                                                                          width: resolveTexture.width,
                                                                          height: resolveTexture.height,
                                                                          mipmapped: false)
                descriptor.usage = .renderTarget
                descriptor.textureType = .type2DMultisampleArray
                descriptor.sampleCount = device.rasterSampleCount
                descriptor.storageMode = .memoryless
                descriptor.arrayLength = resolveTexture.arrayLength
                return device.makeTexture(descriptor: descriptor)!
            }

            if device.supportsMSAA {
                memorylessTargets = .init(repeating: (renderTarget(resolveTexture: drawable.colorTextures[0]),
                                                      renderTarget(resolveTexture: drawable.depthTextures[0])),
                                          count: maxBuffersInFlight)
            } else {
                memorylessTargets = []
            }

            let bufferSize = alignedViewProjectionArraySize * maxBuffersInFlight

            viewProjectionBuffer = device.makeBuffer(length: bufferSize,
                                                     options: [MTLResourceOptions.storageModeShared])!
            viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)
        }
    }
}

extension Renderer.DrawableTarget {
    nonisolated func updateBufferState(uniformBufferIndex: Int, frameIndex: UInt64) {
        viewProjectionBufferOffset = alignedViewProjectionArraySize * uniformBufferIndex

        viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)

        lastUsedFrameIndex = frameIndex
    }

    nonisolated func updateViewProjectionArray(drawable: LayerRenderer.Drawable) {
        let simdDeviceAnchor = drawable.deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        nonisolated func viewProjection(forViewIndex viewIndex: Int) -> float4x4 {
            let view = drawable.views[viewIndex]
            let viewMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = drawable.computeProjection(viewIndex: viewIndex)

            return projectionMatrix * viewMatrix
        }

        viewProjectionArray[0].viewProjectionMatrix.0 = viewProjection(forViewIndex: 0)
        if drawable.views.count > 1 {
            viewProjectionArray[0].viewProjectionMatrix.1 = viewProjection(forViewIndex: 1)
        }
    }
}

nonisolated func sceneObjectModelMatrix(_ obj: SceneObject) -> float4x4 {
    let t = obj.position?.simd ?? SIMD3<Float>(0, 0, 0)
    let r = obj.rotation?.quaternion ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    let s = obj.scale?.simd ?? SIMD3<Float>(1, 1, 1)
    return matrix4x4_translation(t.x, t.y, t.z) * float4x4(r) * float4x4(diagonal: SIMD4<Float>(s.x, s.y, s.z, 1))
}

nonisolated func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return .init(columns: (vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                           vector_float4(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
                           vector_float4(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
                           vector_float4(                  0, 0, 0, 1)))
}

nonisolated func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return .init(columns: (vector_float4(1, 0, 0, 0),
                           vector_float4(0, 1, 0, 0),
                           vector_float4(0, 0, 1, 0),
                           vector_float4(translationX, translationY, translationZ, 1)))
}
