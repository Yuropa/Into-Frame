# Into Frame

**Into Frame** transforms a single image into a fully explorable, interactive 3D environment using Generative AI.

Upload a photo, painting, or any scene — and step inside it.

---

## How It Works

Into Frame is built around a two-part architecture: a Python server that runs the AI generation pipeline, and one of two clients that render the resulting scene.

| Component | Description |
|-----------|-------------|
| **`Server/`** | A Python server that runs the AI generation pipeline. Handles model inference, scene construction, and serves assets to connected clients. |
| **`Into Frame/`** | A Unity project (C#) that connects to the Python server and renders the generated 3D scene in real time. |
| **`IntoFrame visionOS/`** | A native macOS app (Swift + Metal) that connects to the Python server and renders the scene with full immersive support. |

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
chmod +x setup.sh
./setup.sh
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f` | Force clean installation — removes existing libraries and conda environments |
| `-s` | Save existing installation logs (default wipes the logs directory) |

> **Note:** The setup script downloads all required models automatically. If that step fails (e.g. missing Hugging Face access for gated models), you can re-run the download manually:
> ```bash
> conda activate frame
> python Server/main.py download
> ```

### 2. Start the Generation Server

```bash
conda activate frame
python Server/main.py server
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

#### macOS App

Open `IntoFrame visionOS/IntoFrame.xcodeproj` in Xcode, build, and run. The app connects to `ws://localhost:8080` by default — this can be overridden via the `ServerWSURL` key in the app's `Info.plist`.

---

## Running the Pipeline Directly

You can also run the generation pipeline on a single image without starting the server:

```bash
conda activate frame
python Server/main.py run path/to/image.jpg
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./output` | Output directory |
| `--debug`, `-d` | `True` | Save intermediate pipeline files |

---

## CLI Reference

```
python Server/main.py <command>

Commands:
  server      Start the generation server
  run         Run the pipeline on a single image
  download    Download all models required by the pipeline
```

---

## Project Structure

```
into-frame/
├── Server/
│   ├── main.py          # Entry point — CLI for server, run, and download
│   ├── pipeline/        # AI generation pipeline
│   └── server/          # WebSocket/HTTP server logic
├── Into Frame/          # Unity client (C#)
├── IntoFrame visionOS/  # macOS client (Swift + Metal, Xcode)
├── docs/                # Project website
├── setup.sh             # Environment setup script
└── README.md
```
