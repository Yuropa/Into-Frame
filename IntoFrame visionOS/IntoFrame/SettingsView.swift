import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var appModel

    @State private var wsURL    = ""
    @State private var assetURL = ""
    @State private var didReconnect = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Server Settings")
                .font(.headline)

            field("WebSocket URL",
                  placeholder: SceneClient.defaultServerURL,
                  text: $wsURL)

            field("Asset Server URL",
                  placeholder: AssetServer.defaultBaseURL,
                  text: $assetURL)

            Divider()

            HStack(spacing: 10) {
                Button("Reset Defaults") {
                    wsURL    = SceneClient.defaultServerURL
                    assetURL = AssetServer.defaultBaseURL
                }
                .buttonStyle(.bordered)

                Spacer()

                Button {
                    saveAndReconnect()
                } label: {
                    Label(didReconnect ? "Reconnected" : "Save & Reconnect",
                          systemImage: didReconnect ? "checkmark" : "arrow.clockwise")
                }
                .buttonStyle(.borderedProminent)
                .disabled(didReconnect)
            }
        }
        .padding(16)
        .frame(width: 300)
        .onAppear { loadSaved() }
    }

    @ViewBuilder
    private func field(_ label: String, placeholder: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .fontDesign(.monospaced)
                .autocorrectionDisabled()
        }
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

        withAnimation {
            didReconnect = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation { didReconnect = false }
        }
    }
}

#Preview {
    SettingsView()
        .environment(AppModel())
}
