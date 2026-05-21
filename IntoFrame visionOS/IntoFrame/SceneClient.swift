import Foundation
import os

@Observable
class SceneClient {
    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case connected
        case reconnecting
    }

    var connectionState: ConnectionState = .disconnected
    var statusText: String = "Disconnected"
    var progressStep: String = ""
    var progressPercent: Float = 0

    private let serverURL: URL
    private let reconnectDelay: TimeInterval
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "com.intoframe", category: "SceneClient")

    var onSceneInit: ((SceneInitPayload) -> Void)?
    var onObjectSpawn: ((SceneObject) -> Void)?
    var onObjectUpdate: ((ObjectUpdatePayload) -> Void)?
    var onObjectDestroy: ((String) -> Void)?
    var onProgress: ((SceneProgressPayload) -> Void)?

    private static func configuredWebSocketURL() -> URL {
        if let dict = Bundle.main.infoDictionary,
           let urlString = dict["ServerWSURL"] as? String,
           let url = URL(string: urlString) {
            return url
        }
        return URL(string: "ws://localhost:8080")!
    }

    init(serverURL: URL = SceneClient.configuredWebSocketURL(), reconnectDelay: TimeInterval = 3) {
        self.serverURL = serverURL
        self.reconnectDelay = reconnectDelay
    }

    func connect() {
        teardown()

        connectionState = .connecting
        statusText = "Connecting..."
        logger.info("Connecting to \(self.serverURL)...")

        let task = URLSession.shared.webSocketTask(with: serverURL)
        self.webSocketTask = task
        task.resume()

        receiveTask = Task { [weak self] in
            guard let self else { return }

            do {
                try await task.send(.string("{\"type\":\"CLIENT_READY\"}"))
            } catch {
                self.logger.error("Connection failed: \(error.localizedDescription)")
                self.scheduleReconnect()
                return
            }

            self.connectionState = .connected
            self.statusText = "Connected"
            self.logger.info("Connected to server")

            while !Task.isCancelled {
                do {
                    let message = try await task.receive()
                    self.handleMessage(message)
                } catch {
                    if !Task.isCancelled {
                        self.logger.error("Receive error: \(error.localizedDescription)")
                        self.scheduleReconnect()
                    }
                    break
                }
            }
        }
    }

    func disconnect() {
        teardown()
        connectionState = .disconnected
        statusText = "Disconnected"
    }

    private func teardown() {
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
    }

    private func scheduleReconnect() {
        teardown()
        connectionState = .reconnecting
        statusText = "Reconnecting..."

        receiveTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(self.reconnectDelay))
            guard !Task.isCancelled else { return }
            self.connect()
        }
    }

    // MARK: - Message Handling

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .string(let text):
            data = Data(text.utf8)
        case .data(let d):
            data = d
        @unknown default:
            return
        }

        guard let envelope = ServerEnvelope(from: data) else {
            logger.warning("Failed to parse server message")
            return
        }

        route(envelope)
    }

    private func route(_ envelope: ServerEnvelope) {
        switch envelope.type {
        case "SCENE_INIT":
            if let payload: SceneInitPayload = envelope.decode() {
                logger.info("SCENE_INIT with \(payload.scene.objects?.count ?? 0) objects")
                onSceneInit?(payload)
            }

        case "OBJECT_SPAWN":
            if let obj: SceneObject = envelope.decode() {
                onObjectSpawn?(obj)
            }

        case "OBJECT_UPDATE":
            if let update: ObjectUpdatePayload = envelope.decode() {
                onObjectUpdate?(update)
            }

        case "OBJECT_DESTROY":
            if let destroy: ObjectDestroyPayload = envelope.decode() {
                onObjectDestroy?(destroy.id)
            }

        case "PROGRESS":
            if let progress: SceneProgressPayload = envelope.decode() {
                progressStep = progress.step
                progressPercent = progress.percent
                logger.info("Progress: \(progress.step) (\(Int(progress.percent * 100))%)")
                onProgress?(progress)
            }

        default:
            logger.warning("Unknown message type: \(envelope.type)")
        }
    }

    // MARK: - Sending

    func sendJSON(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        let logger = self.logger
        webSocketTask?.send(.string(text)) { error in
            if let error {
                logger.error("Send error: \(error.localizedDescription)")
            }
        }
    }

    func sendObjectEvent(objectId: String, event: String) {
        sendJSON([
            "type": "OBJECT_EVENT",
            "payload": ["objectId": objectId, "event": event]
        ])
    }
}
