import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var appModel
    @Environment(\.openWindow) private var openWindow

    @State private var showStatusPopover = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            statusCard
            if showPipelineProgress { pipelineProgressCard }
            if appModel.sceneManager.isLoading { downloadProgressCard }
            actionButtons
        }
        .padding(16)
        .frame(width: 300)
        .task { appModel.sceneManager.connect() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: "viewfinder.circle.fill")
                .font(.system(size: 38))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.tint)

            VStack(alignment: .leading, spacing: 2) {
                Text("Into Frame")
                    .font(.title2)
                    .fontWeight(.bold)
                    .tracking(-0.5)
                Text("Spatial Visualization")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.bottom, 4)
    }

    // MARK: - Status Card

    private var statusCard: some View {
        Button { showStatusPopover.toggle() } label: {
            HStack(spacing: 12) {
                statusDot
                VStack(alignment: .leading, spacing: 1) {
                    Text(appModel.sceneManager.client.statusText)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text("Into Frame server")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if !appModel.sceneManager.sceneObjects.isEmpty {
                    Text("\(appModel.sceneManager.sceneObjects.count)")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.secondary.opacity(0.12), in: Capsule())
                }
                Image(systemName: "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12))
        .popover(isPresented: $showStatusPopover, arrowEdge: .bottom) {
            ConnectionDetailView()
                .environment(appModel)
        }
    }

    private var statusDot: some View {
        ZStack {
            Circle()
                .fill(statusColor.opacity(0.18))
                .frame(width: 26, height: 26)
            Circle()
                .fill(statusColor)
                .frame(width: 9, height: 9)
        }
    }

    // MARK: - Progress Cards

    private var showPipelineProgress: Bool {
        appModel.sceneManager.client.progressPercent > 0
            && appModel.sceneManager.client.progressPercent < 1
    }

    private var pipelineProgressCard: some View {
        let label = appModel.sceneManager.client.progressStep.isEmpty
            ? "Processing…"
            : appModel.sceneManager.client.progressStep
        return progressCard(label: label,
                            value: Double(appModel.sceneManager.client.progressPercent),
                            total: 1.0)
    }

    private var downloadProgressCard: some View {
        progressCard(
            label: "Assets \(appModel.sceneManager.completedAssets) / \(appModel.sceneManager.totalAssets)",
            value: Double(appModel.sceneManager.completedAssets),
            total: Double(max(appModel.sceneManager.totalAssets, 1))
        )
    }

    private func progressCard(label: String, value: Double, total: Double) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Text("\(Int((value / max(total, 1)) * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            ProgressView(value: value, total: total)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Actions

    private var actionButtons: some View {
        HStack(spacing: 8) {
            ToggleImmersiveSpaceButton()
                .frame(maxWidth: .infinity)

            Button {
                openWindow(id: "mac-preview")
            } label: {
                Label("Preview", systemImage: "rectangle.on.rectangle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
        }
        .controlSize(.large)
        .padding(.top, 2)
    }

    // MARK: - Colors

    private var statusColor: Color {
        switch appModel.sceneManager.client.connectionState {
        case .connected:                .green
        case .connecting, .reconnecting: .yellow
        case .disconnected:             .red
        }
    }
}

// MARK: - Connection Detail Popover

private struct ConnectionDetailView: View {
    @Environment(AppModel.self) private var appModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Connection")
                .font(.headline)

            detailSection("Server") {
                row("WebSocket", appModel.sceneManager.client.serverURLString)
                row("Assets", appModel.sceneManager.assetServer.baseURL.absoluteString)
            }

            let objects = Array(appModel.sceneManager.sceneObjects.values)
            if !objects.isEmpty {
                let billboards = objects.filter { $0.type == "billboard" }.count
                let meshes     = objects.filter { $0.type == "mesh" }.count

                detailSection("Scene") {
                    row("Total objects", "\(objects.count)")
                    if billboards > 0 { row("Billboards", "\(billboards)") }
                    if meshes > 0     { row("Meshes", "\(meshes)") }
                }
            }

            let assets = appModel.sceneManager.downloadedAssets
            if !assets.isEmpty {
                let totalBytes = assets.values.reduce(0) { $0 + $1.count }

                detailSection("Downloads") {
                    row("Files", "\(assets.count)")
                    row("Total size", formatBytes(totalBytes))
                }
            }
        }
        .padding(16)
        .frame(minWidth: 260, maxWidth: 320)
    }

    @ViewBuilder
    private func detailSection<Content: View>(_ title: String,
                                              @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .tracking(0.5)
            content()
        }
    }

    private func row(_ key: String, _ value: String) -> some View {
        HStack {
            Text(key)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontDesign(.monospaced)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private func formatBytes(_ bytes: Int) -> String {
        switch bytes {
        case ..<1024:              return "\(bytes) B"
        case ..<(1024 * 1024):    return String(format: "%.1f KB", Double(bytes) / 1024)
        default:                   return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        }
    }
}

#Preview {
    ContentView()
        .environment(AppModel())
}
