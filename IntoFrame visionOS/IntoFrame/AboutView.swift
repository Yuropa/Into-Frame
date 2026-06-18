import SwiftUI

struct AboutView: View {
    private var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }
    private var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
    }

    var body: some View {
        VStack(spacing: 0) {
            // Icon + identity
            VStack(spacing: 14) {
                Image(systemName: "viewfinder.circle.fill")
                    .font(.system(size: 72))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.tint)

                VStack(spacing: 4) {
                    Text("Into Frame")
                        .font(.title)
                        .fontWeight(.bold)
                        .tracking(-0.5)
                    Text("Version \(version) (\(build))")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 36)
            .padding(.horizontal, 32)

            Divider()
                .padding(.vertical, 22)
                .padding(.horizontal, 24)

            // Description
            Text("Visualize AI-generated 3D scenes on Apple Vision Pro and macOS. Streams live scene data from the Into Frame pipeline server.")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 28)

            Spacer(minLength: 24)

            // Website button
            Button {
                NSWorkspace.shared.open(URL(string: "https://jford.link/Into-Frame/")!)
            } label: {
                Label("Visit Website", systemImage: "safari")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.horizontal, 28)

            // Copyright
            Text("© 2026 Josh Ford")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.top, 14)
                .padding(.bottom, 28)
        }
        .frame(width: 320, height: 370)
    }
}

#Preview {
    AboutView()
}
