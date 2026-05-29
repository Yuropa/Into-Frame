# Into Frame

**Into Frame** transforms a single image into a fully explorable, interactive 3D environment using Generative AI.

Upload a photo, painting, or any scene — and step inside it.

---

## How It Works

Into Frame is built around a two-part architecture: a Python server that runs the AI generation pipeline, and one of two clients that render the resulting scene.

| Component | Description |
|-----------|-------------|
| **`server/`** | A Python server that runs the AI generation pipeline. Handles model inference, scene construction, and serves assets to connected clients. |
| **`Into Frame/`** | A Unity project (C#) that connects to the Python server and renders the generated 3D scene in real time. |
| **`IntoFrame visionOS/`** | A native macOS app (Swift + Metal 4) that connects to the Python server, renders the generated scene, and streams it as a full-immersion experience to an Apple Vision Pro via Remote Immersive Space. |

---

## Getting Started

### Prerequisites

- Python 3.12
- Conda (Miniconda or Anaconda)
- CUDA 13.0 + CUDA Toolkit *(recommended — see note below)*
- **For the Unity client:** Unity 2022.3 LTS or later
- **For the macOS app:** Xcode, macOS 26.4+

> **Note on CUDA:** The pipeline can run without a CUDA-compatible GPU, but output quality and performance may vary. macOS MPS is supported automatically where available.

### 1. Setup

Run the setup script to configure the Python environment. **This will take a while** as it installs all required dependencies and downloads AI models:

```bash
chmod +x frame.sh
./frame.sh setup
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f` | Force clean installation — removes existing libraries and conda environments |
| `-s` | Save existing installation logs (default wipes the logs directory) |
| `-v` | Verbose mode — dump output to terminal instead of a log file |

> **Note:** The setup script downloads all required models automatically. If that step fails (e.g. missing Hugging Face access for gated models), you can re-run the download manually:
> ```bash
> ./frame.sh download
> ```

### 2. Start the Generation Server

```bash
./frame.sh server
```

By default this binds to `localhost:8080`, with an asset server on port `3000`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Host to bind the server |
| `--port` | `8080` | Generation server port |
| `--asset-port` | `3000` | Asset server port |
| `--output`, `-o` | `./output` | Output directory |
| `--debug`, `-d` | `False` | Save intermediate pipeline files |

### 3. Connect a Client

#### Unity

Open the `Into Frame/` folder as a Unity project. With the Python server running, press **Play** to connect and explore your generated scene.

#### macOS App (Apple Vision Pro)

Open `IntoFrame visionOS/IntoFrame.xcodeproj` in Xcode, build, and run. The app connects to the Python server, downloads the generated scene assets, and renders them using a custom Metal 4 renderer. Once a scene is ready, tap **Enter Immersive Space** to stream a full-immersion view of the environment to a nearby Apple Vision Pro via [Remote Immersive Space](https://developer.apple.com/documentation/visionos/creating-immersive-spaces-in-visionos-with-swiftui).

The main window shows connection status, scene generation progress, and asset download progress before the immersive view is available.

The server URL defaults to `ws://localhost:8080` and can be overridden via the `ServerWSURL` key in the app's `Info.plist`.

---

## Running the Pipeline Directly

Run the generation pipeline on a single image without starting the server:

```bash
./frame.sh run path/to/image.jpg
```

The default image (when called with no arguments) is `samples/Paris.jpg`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./output` | Output directory |
| `--debug`, `-d` | `True` | Save intermediate pipeline files |
| `--config` | `server/config.yaml` | Pipeline configuration YAML |

---

## Remote Server

To run the generation server on a remote machine with local port forwarding (useful when the GPU is on a separate box):

```bash
./frame.sh remote --host 192.168.1.10 --user admin
```

This SSH-tunnels ports `8080` and `3000` to `localhost` so the clients connect as if the server were local.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `your-remote-host` | Remote hostname or IP |
| `--user` | `admin` | Remote SSH username |
| `--dir` | `~/Research/Into-Frame/server` | Remote project directory |
| `--port` | `8080` | Server port |
| `--asset-port` | `3000` | Asset server port |
| `--ssh-port` | `22` | SSH port |
| `--key` | — | Path to SSH private key |

`REMOTE_USER`, `REMOTE_HOST`, `REMOTE_DIR`, `PORT`, and `ASSET_PORT` can also be set as environment variables.

---

## CLI Reference

`frame.sh` is the unified entry point for all server operations. Global options apply to all subcommands.

```
./frame.sh [--env ENV] [--seed VALUE] <command> [options]

Commands:
  run       Run the pipeline on an image (default: samples/Paris.jpg)
  server    Start the local generation server
  download  Download all models required by the pipeline
  remote    Start the server on a remote machine via SSH
  setup     Install dependencies and configure conda environments
```

`--env` overrides the conda environment name (default: `frame`). `--seed` accepts a bare integer or `STAGE:VALUE` pairs and is repeatable.

Pass `-h` or `--help` anywhere to see options:

```bash
./frame.sh -h              # global help
./frame.sh run -h          # run options
./frame.sh server -h       # server options
./frame.sh remote -h       # remote options
```

---

## Project Structure

```
into-frame/
├── frame.sh             # Unified CLI — setup, run, server, remote
├── server/
│   ├── main.py          # Python entry point (server, run, download)
│   ├── pipeline/        # AI generation pipeline stages
│   ├── scene/           # 3D scene representation
│   ├── server/          # WebSocket/HTTP server logic
│   └── samples/         # Sample input images
├── scripts/
│   ├── setup.sh         # Full environment install script
│   └── remote-server.sh # Standalone remote SSH launcher
├── Into Frame/          # Unity client (C#)
├── IntoFrame visionOS/  # macOS client (Swift + Metal, Xcode)
├── docs/                # Project website
└── requirements*.txt    # Per-environment pip requirements
```
