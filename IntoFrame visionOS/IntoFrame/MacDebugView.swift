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

                if appModel.sceneManager.isLoading {
                    Text("Downloading \(appModel.sceneManager.completedAssets)/\(appModel.sceneManager.totalAssets)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.7))
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
