# Into Frame

**Into Frame** transforms a single image into a fully explorable, interactive 3D environment using Generative AI.

Upload a photo, painting, or any scene — and step inside it.

---

## How It Works

Into Frame is built around a two-part architecture:

| Component | Description |
|-----------|-------------|
| **`Server/`** | A Python server that runs the AI generation pipeline. Handles model inference, scene construction, and serves assets to the Unity client. |
| **`Into Frame/`** | A Unity project (C#) that connects to the Python server and renders the generated 3D scene in real time. |

---

## Getting Started

### Prerequisites

- Python 3.12
- Unity 2022.3 LTS or later
- CUDA 13.0 + CUDA Toolkit *(recommended — see note below)*

> **Note on CUDA:** The pipeline can run without a CUDA-compatible GPU, but output quality and performance may vary. macOS MPS is supported automatically where available.

### 1. Setup

Run the setup script to configure the Python environment. **This may take a while** as it installs all required dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Download Models

Before running for the first time, download all the required AI models:

```bash
python Server/main.py download
```

### 3. Start the Generation Server

```bash
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

### 4. Open in Unity

Open the `Into Frame/` folder as a Unity project. With the Python server running, press **Play** to connect and explore your generated scene.

---

## Running the Pipeline Directly

You can also run the generation pipeline on a single image without starting the server:

```bash
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
├── Into Frame/          # Unity project (C#)
├── setup.sh             # Environment setup script
└── README.md
```