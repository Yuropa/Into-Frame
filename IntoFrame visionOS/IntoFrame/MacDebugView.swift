import SwiftUI
import Metal
import MetalKit
import AppKit

// MARK: - Debug Window Root View

struct MacDebugView: View {
    @Environment(AppModel.self) private var appModel

    var body: some View {
        ZStack(alignment: .topLeading) {
            MacMetalContainerView(appModel: appModel)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 8, height: 8)
                    Text(appModel.sceneManager.client.statusText)
                        .font(.caption)
                        .foregroundStyle(.white)
                }

                if showPipelineProgress {
                    let step = appModel.sceneManager.client.progressStep
                    let pct  = appModel.sceneManager.client.progressPercent
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(step.isEmpty ? "Processing…" : step)
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.85))
                                .lineLimit(1)
                            Spacer()
                            Text("\(Int(pct * 100))%")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.white.opacity(0.6))
                        }
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(.white.opacity(0.15)).frame(height: 3)
                                Capsule().fill(.white.opacity(0.7))
                                    .frame(width: geo.size.width * CGFloat(pct), height: 3)
                            }
                        }
                        .frame(height: 3)
                    }
                }

                if appModel.sceneManager.isLoading {
                    let done  = appModel.sceneManager.completedAssets
                    let total = appModel.sceneManager.totalAssets
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Downloading assets")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.85))
                            Spacer()
                            Text("\(done)/\(total)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.white.opacity(0.6))
                        }
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(.white.opacity(0.15)).frame(height: 3)
                                Capsule().fill(.white.opacity(0.7))
                                    .frame(width: geo.size.width * CGFloat(done) / CGFloat(max(total, 1)), height: 3)
                            }
                        }
                        .frame(height: 3)
                    }
                }

                if !appModel.sceneManager.sceneObjects.isEmpty {
                    Text("\(appModel.sceneManager.sceneObjects.count) objects")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.5))
                }

                Divider().frame(width: 80)

                Text("Drag to orbit · Scroll to zoom · Arrows to pan")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.4))
            }
            .padding(12)
            .background(.black.opacity(0.45))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(12)
        }
        .frame(minWidth: 640, minHeight: 480)
    }

    private var showPipelineProgress: Bool {
        let p = appModel.sceneManager.client.progressPercent
        return p > 0 && p < 1
    }

    private var statusColor: Color {
        switch appModel.sceneManager.client.connectionState {
        case .connected:              .green
        case .connecting, .reconnecting: .yellow
        case .disconnected:           .red
        }
    }
}

// MARK: - NSViewRepresentable wrapper

struct MacMetalContainerView: NSViewRepresentable {
    let appModel: AppModel

    func makeNSView(context: Context) -> MacMetalViewHost {
        let view = MacMetalViewHost()

        if let renderer = MacRenderer(mtkView: view) {
            view.delegate    = renderer
            view.onDrag      = { dx, dy in renderer.handleMouseDrag(deltaX: Float(dx), deltaY: Float(dy)) }
            view.onScroll    = { delta  in renderer.handleScroll(delta: Float(delta)) }
            view.onTranslate = { dx, dz in renderer.handleTranslate(dx: Float(dx), dz: Float(dz)) }

            appModel.macRenderer = renderer

            // Replay whatever scene is already loaded
            let objects = Array(appModel.sceneManager.sceneObjects.values)
            if !objects.isEmpty {
                renderer.loadScene(objects: objects,
                                   assets: appModel.sceneManager.downloadedAssets,
                                   params: appModel.sceneManager.sceneParams)
            }
        }

        return view
    }

    func updateNSView(_ nsView: MacMetalViewHost, context: Context) {}
}

// MARK: - Custom MTKView subclass for mouse/scroll input

final class MacMetalViewHost: MTKView {
    var onDrag:      ((CGFloat, CGFloat) -> Void)?
    var onScroll:    ((CGFloat) -> Void)?
    var onTranslate: ((CGFloat, CGFloat) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        onDrag?(event.deltaX, event.deltaY)
    }

    override func scrollWheel(with event: NSEvent) {
        let delta = event.hasPreciseScrollingDeltas
            ? event.scrollingDeltaY * 0.1
            : event.scrollingDeltaY
        onScroll?(delta)
    }

    override func keyDown(with event: NSEvent) {
        switch event.specialKey {
        case .leftArrow:  onTranslate?(-1, 0)
        case .rightArrow: onTranslate?( 1, 0)
        case .upArrow:    onTranslate?(0,  1)
        case .downArrow:  onTranslate?(0, -1)
        default:          super.keyDown(with: event)
        }
    }
}
