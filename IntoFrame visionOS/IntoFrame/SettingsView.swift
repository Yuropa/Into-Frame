import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            ServerSettingsPane()
                .tabItem {
                    Label("Server", systemImage: "network")
                }
        }
        .frame(width: 420, height: 220)
    }
}

private struct ServerSettingsPane: View {
    @Environment(AppModel.self) private var appModel

    @State private var wsURL    = ""
    @State private var assetURL = ""
    @State private var didReconnect = false

    var body: some View {
        Form {
            TextField("WebSocket URL", text: $wsURL)
                .fontDesign(.monospaced)
                .autocorrectionDisabled()

            TextField("Asset Server URL", text: $assetURL)
                .fontDesign(.monospaced)
                .autocorrectionDisabled()

            HStack {
                Button("Reset Defaults") {
                    wsURL    = SceneClient.defaultServerURL
                    assetURL = AssetServer.defaultBaseURL
                }

                Spacer()

                Button {
                    saveAndReconnect()
                } label: {
                    Label(
                        didReconnect ? "Reconnected" : "Save & Reconnect",
                        systemImage: didReconnect ? "checkmark" : "arrow.clockwise"
                    )
                }
                .buttonStyle(.borderedProminent)
                .disabled(didReconnect)
            }
        }
        .formStyle(.grouped)
        .onAppear { loadSaved() }
    }

    private func loadSaved() {
        wsURL    = UserDefaults.standard.string(forKey: "serverWSURL")  ?? SceneClient.defaultServerURL
        assetURL = UserDefaults.standard.string(forKey: "assetBaseURL") ?? AssetServer.defaultBaseURL
    }

    private func saveAndReconnect() {
        let ws    = wsURL.trimmingCharacters(in: .whitespaces).isEmpty    ? SceneClient.defaultServerURL : wsURL.trimmingCharacters(in: .whitespaces)
        let asset = assetURL.trimmingCharacters(in: .whitespaces).isEmpty ? AssetServer.defaultBaseURL   : assetURL.trimmingCharacters(in: .whitespaces)

        UserDefaults.standard.set(ws,    forKey: "serverWSURL")
        UserDefaults.standard.set(asset, forKey: "assetBaseURL")

        appModel.sceneManager.reconnect(websocketURL: ws, assetBaseURL: asset)

        withAnimation { didReconnect = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation { didReconnect = false }
        }
    }
}

#Preview {
    SettingsView()
        .environment(AppModel())
}
