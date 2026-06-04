import Foundation
import simd

struct Vec3: Codable, Sendable {
    var x: Float = 0
    var y: Float = 0
    var z: Float = 0

    var simd: SIMD3<Float> { SIMD3(x, y, z) }

    var quaternion: simd_quatf {
        let rx = simd_quatf(angle: x * .pi / 180, axis: SIMD3(1, 0, 0))
        let ry = simd_quatf(angle: y * .pi / 180, axis: SIMD3(0, 1, 0))
        let rz = simd_quatf(angle: z * .pi / 180, axis: SIMD3(0, 0, 1))
        return ry * rx * rz
    }
}

struct SceneObject: Codable, Identifiable, Sendable {
    var id: String
    var type: String
    var name: String?
    var texture: String?
    var mesh: String?
    var position: Vec3?
    var rotation: Vec3?
    var scale: Vec3?
}

struct ExtrinsicsParams: Codable, Sendable {
    var rotation: [Float]
    var translation: [Float]
}

struct SceneLightingData: Codable, Sendable {
    var ldr: String?
    var log: String?
    var width: Int?
    var height: Int?
}

struct SceneParams: Codable, Sendable {
    var ambientColor: String?
    var gravity: Float?
    var timeScale: Float?
    var skybox: String?
    var skyboxRotation: Float?
    var nearClipPlane: Float?
    var farClipPlane: Float?
    var objects: [SceneObject]?
    var extrinsics: ExtrinsicsParams?
    var lighting: SceneLightingData?
}

struct SceneInitPayload: Codable, Sendable {
    var scene: SceneParams
}

struct ObjectUpdatePayload: Codable, Sendable {
    var id: String
    var changes: SceneObject?
}

struct ObjectDestroyPayload: Codable, Sendable {
    var id: String
}

struct SceneProgressPayload: Codable, Sendable {
    var step: String
    var detail: String?
    var percent: Float
}

struct ServerEnvelope: Sendable {
    let type: String
    let payloadData: Data

    init?(from data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return nil
        }
        self.type = type
        if let payload = json["payload"] {
            self.payloadData = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
        } else {
            self.payloadData = Data()
        }
    }

    func decode<T: Decodable>() -> T? {
        try? JSONDecoder().decode(T.self, from: payloadData)
    }
}
