import SwiftUI

@MainActor
@Observable
class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"
    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }
    var immersiveSpaceState = ImmersiveSpaceState.closed

    let sceneManager = SceneManager()
    var renderer: Renderer?
    var macRenderer: MacRenderer?

    init() {
        sceneManager.onSceneReady = { [weak self] objects, assets, params in
            guard let self else { return }
            if let renderer = self.renderer {
                Task { await renderer.loadScene(objects: objects, assets: assets, params: params) }
            }
            self.macRenderer?.loadScene(objects: objects, assets: assets, params: params)
        }
    }
}
