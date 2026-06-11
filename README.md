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
> # or with a specific config:
> ./frame.sh download --config server/config_alt.yaml
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
| `--debug`, `-d` | `false` | Save intermediate pipeline files |
| `--config` | `server/config.yaml` | Pipeline configuration YAML |

### 3. Connect a Client

#### Unity

Open the `Into Frame/` folder as a Unity project. With the Python server running, press **Play** to connect and explore your generated scene.

#### macOS App (Apple Vision Pro)

Open `IntoFrame visionOS/IntoFrame.xcodeproj` in Xcode, build, and run. The app connects to the Python server, downloads the generated scene assets, and renders them using a custom Metal 4 renderer. Once a scene is ready, tap **Enter Immersive Space** to stream a full-immersion view of the environment to a nearby Apple Vision Pro via [Remote Immersive Space](https://developer.apple.com/documentation/visionos/creating-immersive-spaces-in-visionos-with-swiftui).

The main window shows connection status, scene generation progress, and asset download progress before the immersive view is available.

The server URL defaults to `ws://localhost:8080` and can be overridden via the `ServerWSURL` key in the app's `Info.plist`.

---

## Running the Pipeline Directly

Run the generation pipeline on a single image (or directory of images) and produce a `.frame` archive, without starting the full server:

```bash
./frame.sh run path/to/image.jpg
```

The input path is optional and defaults to `samples/Paris.jpg`. You can also pass a directory to batch-process multiple images.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./output` | Output directory |
| `--debug`, `-d` | `true` | Save intermediate pipeline files |
| `--config` | `server/config.yaml` | Pipeline configuration YAML |

---

## Serving a Scene Locally

If you already have a `.frame` archive (produced by `run` or pulled from a remote), you can serve it as a local scene server without re-running the pipeline:

```bash
./frame.sh local ./output/Paris.frame
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Host to bind the server |
| `--port` | `8080` | Server port |
| `--asset-port` | `3000` | Asset server port |

---

## Remote Server

To run the generation server on a remote machine with local port forwarding (useful when the GPU is on a separate box):

```bash
./frame.sh remote --host 192.168.1.10 --user admin
```

This SSH-tunnels ports `8080` and `3000` to `localhost` so the clients connect as if the server were local.

`remote` accepts an optional subcommand as its first argument:

| Subcommand | Description |
|------------|-------------|
| `server` *(default)* | Start the remote server with port forwarding |
| `pull` | Pull generated output files from the remote machine |
| `clear` | Remove cached output files on the remote machine |

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
| `--debug`, `-d` | — | Save intermediate files on remote *(server only)* |
| `--config` | — | Remote pipeline config YAML *(server only)* |
| `--output`, `-o` | — | Output directory to clear *(clear only)* |

`REMOTE_USER`, `REMOTE_HOST`, `REMOTE_DIR`, `PORT`, and `ASSET_PORT` can also be set as environment variables.

---

## CLI Reference

`frame.sh` is the unified entry point for all server operations. Global options apply to all subcommands.

```
./frame.sh [--env ENV] [--seed VALUE] [-v] <command> [options]

Commands:
  run       Run the pipeline on an image (or directory) and produce a .frame archive
  server    Start the local generation server
  local     Serve an existing .frame archive without re-running the pipeline
  download  Download all models required by the pipeline
  remote    Start the server on a remote machine via SSH (subcommands: server, pull, clear)
  setup     Install dependencies and configure conda environments
  clear     Remove cached files from the output directory
```

**Global options:**

| Flag | Description |
|------|-------------|
| `--env ENV` | Conda environment name (default: `frame`) |
| `--seed VALUE` | Random seed — bare integer or `STAGE:VALUE` pair; repeatable |
| `-v`, `--verbose` | Print logs to the terminal instead of a log file |

Pass `-h` or `--help` anywhere to see options:

```bash
./frame.sh                 # global help (no subcommand)
./frame.sh -h              # global help (explicit)
./frame.sh run -h          # run options
./frame.sh server -h       # server options
./frame.sh local -h        # local options
./frame.sh remote -h       # remote options
```

---

## Clearing Output

To remove all cached files from the output directory:

```bash
./frame.sh clear
```

By default this clears `./output`. Pass `--output` to target a different directory:

```bash
./frame.sh clear --output ./my-output
```

---

## Project Structure

```
into-frame/
├── frame.sh             # Unified CLI — setup, run, local, server, remote
├── server/
│   ├── main.py          # Python entry point (server, run, download)
│   ├── pipeline/        # AI generation pipeline stages
│   ├── scene/           # 3D scene representation
│   ├── server/          # WebSocket/HTTP server logic
│   └── samples/         # Sample input images
├── scripts/
│   ├── setup.sh         # Full environment install script
│   └── remote-server.sh # Standalone remote SSH launcher
├── requirements/        # Per-environment pip requirements
├── Into Frame/          # Unity client (C#)
├── IntoFrame visionOS/  # macOS client (Swift + Metal, Xcode)
└── docs/                # Project website
```
