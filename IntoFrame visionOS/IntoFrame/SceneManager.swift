import Foundation
import os

@MainActor
@Observable
class SceneManager {
    let assetServer = AssetServer()
    let client = SceneClient()

    var sceneObjects: [String: SceneObject] = [:]
    var sceneParams: SceneParams?
    var downloadedAssets: [String: Data] = [:]
    var totalAssets = 0
    var completedAssets = 0
    var isLoading = false

    var onSceneReady: (([SceneObject], [String: Data], SceneParams?) -> Void)?

    private let logger = Logger(subsystem: "com.intoframe", category: "SceneManager")

    init() {
        client.onSceneInit = { [weak self] payload in
            self?.handleSceneInit(payload)
        }
        client.onObjectSpawn = { [weak self] obj in
            self?.handleSpawn(obj)
        }
        client.onObjectUpdate = { [weak self] update in
            self?.handleUpdate(update)
        }
        client.onObjectDestroy = { [weak self] id in
            self?.handleDestroy(id)
        }
    }

    func connect() {
        client.connect()
    }

    // MARK: - Scene Init

    private func handleSceneInit(_ payload: SceneInitPayload) {
        sceneObjects.removeAll()
        downloadedAssets.removeAll()
        sceneParams = payload.scene

        guard let objects = payload.scene.objects else {
            isLoading = false
            return
        }

        var assetNames: Set<String> = []
        for obj in objects {
            if let mesh = obj.mesh { assetNames.insert(mesh) }
            if let texture = obj.texture { assetNames.insert(texture) }
        }
        if let skybox = payload.scene.skybox {
            assetNames.insert(skybox)
        }

        totalAssets = assetNames.count
        completedAssets = 0
        isLoading = assetNames.count > 0

        for obj in objects {
            sceneObjects[obj.id] = obj
        }

        logger.info("Scene init: \(objects.count) objects, \(assetNames.count) assets")

        Task {
            await downloadAllAssets(assetNames)
            onSceneReady?(Array(sceneObjects.values), downloadedAssets, sceneParams)
        }
    }

    private func downloadAllAssets(_ names: Set<String>) async {
        let server = assetServer

        await withTaskGroup(of: (String, Data?).self) { group in
            for name in names {
                group.addTask {
                    do {
                        let data = try await server.fetchResource(name)
                        return (name, data)
                    } catch {
                        return (name, nil)
                    }
                }
            }

            for await (name, data) in group {
                if let data {
                    downloadedAssets[name] = data
                    logger.info("Downloaded: \(name) (\(data.count) bytes)")
                } else {
                    logger.error("Failed to download: \(name)")
                }
                completedAssets += 1
            }
        }

        isLoading = false
        logger.info("All assets downloaded")
    }

    // MARK: - Object Lifecycle

    private func handleSpawn(_ obj: SceneObject) {
        if sceneObjects[obj.id] != nil {
            handleUpdate(ObjectUpdatePayload(id: obj.id, changes: obj))
            return
        }

        sceneObjects[obj.id] = obj

        Task {
            let server = assetServer
            if let mesh = obj.mesh, downloadedAssets[mesh] == nil {
                if let data = try? await server.fetchResource(mesh) {
                    downloadedAssets[mesh] = data
                }
            }
            if let texture = obj.texture, downloadedAssets[texture] == nil {
                if let data = try? await server.fetchResource(texture) {
                    downloadedAssets[texture] = data
                }
            }
            onSceneReady?(Array(sceneObjects.values), downloadedAssets, sceneParams)
        }
    }

    private func handleUpdate(_ update: ObjectUpdatePayload) {
        guard var existing = sceneObjects[update.id], let changes = update.changes else { return }

        if let pos = changes.position { existing.position = pos }
        if let rot = changes.rotation { existing.rotation = rot }
        if let scale = changes.scale { existing.scale = scale }
        if let texture = changes.texture { existing.texture = texture }
        if let mesh = changes.mesh { existing.mesh = mesh }

        sceneObjects[update.id] = existing
        onSceneReady?(Array(sceneObjects.values), downloadedAssets, sceneParams)
    }

    private func handleDestroy(_ id: String) {
        sceneObjects.removeValue(forKey: id)
        onSceneReady?(Array(sceneObjects.values), downloadedAssets, sceneParams)
    }
}
