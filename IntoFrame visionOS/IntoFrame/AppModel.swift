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

    init() {
        sceneManager.onSceneReady = { [weak self] objects, assets, params in
            guard let renderer = self?.renderer else { return }
            Task {
                await renderer.loadScene(objects: objects, assets: assets, params: params)
            }
        }
    }
}
