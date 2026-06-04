import SwiftUI

struct ContentView: View {

    @Environment(AppModel.self) private var appModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 20) {
            Text("Into Frame")
                .font(.largeTitle)
                .bold()

            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(appModel.sceneManager.client.statusText)
                    .foregroundStyle(.secondary)
            }

            if appModel.sceneManager.client.progressPercent > 0
                && appModel.sceneManager.client.progressPercent < 1 {
                VStack(spacing: 4) {
                    ProgressView(value: Double(appModel.sceneManager.client.progressPercent))
                    Text(appModel.sceneManager.client.progressStep)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            }

            if appModel.sceneManager.isLoading {
                VStack(spacing: 4) {
                    ProgressView(
                        value: Double(appModel.sceneManager.completedAssets),
                        total: Double(max(appModel.sceneManager.totalAssets, 1))
                    )
                    Text("Downloading assets \(appModel.sceneManager.completedAssets)/\(appModel.sceneManager.totalAssets)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            }

            ToggleImmersiveSpaceButton()

            Button("Open Preview Window") {
                openWindow(id: "mac-preview")
            }
            .buttonStyle(.bordered)

            if !appModel.sceneManager.sceneObjects.isEmpty {
                Text("\(appModel.sceneManager.sceneObjects.count) objects in scene")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .task {
            appModel.sceneManager.connect()
        }
    }

    private var statusColor: Color {
        switch appModel.sceneManager.client.connectionState {
        case .connected: .green
        case .connecting, .reconnecting: .yellow
        case .disconnected: .red
        }
    }
}

#Preview {
    ContentView()
        .environment(AppModel())
}
