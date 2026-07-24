import asyncio
import json
import math
import uuid
import threading
import queue
import os
from aiohttp import web
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, TYPE_CHECKING
# PipelineContext/ContextKey live in pipeline_context.py, not pipeline.py -- importing
# from pipeline.pipeline here would drag in every stage class (and therefore torch/
# transformers/diffusers/...) just to serve a pre-built .frame archive's files, which
# is all `local` mode does (pipeline=None; see _progress_scene). Pipeline itself is
# only ever used as a type hint in this file (the caller constructs and passes one
# in), so it's TYPE_CHECKING-only -- no runtime import, no runtime dependency on the
# heavy stage graph.
from pipeline.pipeline_context import PipelineContext, ContextKey
from server.messages import ServerMessages, ClientMessages
from scene.scene import Scene
from scene.object import Object3D
from scene.splat_material import SplatMaterial
from util.path_utils import resource_directory
from pipeline.pipeline_input import PipelineInput

import websockets

if TYPE_CHECKING:
    from pipeline.pipeline import Pipeline

class SimulationServerConfiguration():
    def __init__(self) -> None:
        self.address = "localhost"
        self.port = 8080
        self.asset_port = 3000
        self.log = None

class SimulationServer():
    _context: Optional[PipelineContext]

    def __init__(
        self,
        config: SimulationServerConfiguration,
        pipeline: Optional["Pipeline"] = None,
        context: Optional[PipelineContext] = None,
        asset_dir: Optional[Path] = None,
        input_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.log = config.log
        self.clients: set = set()
        self._input_path = input_path or resource_directory() / "Mount Rainier.jpg"

        if asset_dir is not None:
            self.asset_dir = asset_dir
        elif pipeline is not None:
            self.asset_dir = pipeline.config.temp / "assets"
        else:
            raise ValueError("Either pipeline or asset_dir must be provided")

        self.scene = Scene()
        self._splat_material: Optional[SplatMaterial] = None
        self._scene_id: Optional[str] = None

        self._pipeline_task: asyncio.Task | None = None
        self._client_connected = asyncio.Event()

        self._asset_server = web.Application()
        self._context = context

    def port(self):
        return self.config.port
    
    def address(self):
        return self.config.address

    def host(self):
        addr = self.address()
        if addr == "localhost":
            return "0.0.0.0"
        else:
            return addr
        
    def asset_port(self):
        return self.config.asset_port
        
    def _find_asset(self, filename) -> Optional[Path]:
        matches = [p for p in self.asset_dir.glob(f"{filename}.*") if p.suffix != ".meta"]
        if not matches:
            return None
        return matches[0]

    async def _serve_assets(self):
        async def serve_asset(request):
            filename = request.match_info["filename"]

            match = self._find_asset(filename)
            if not match:
                if self._context is not None:
                    match = self._context.save_object(filename, self.asset_dir)

            if not match:
                # Coulnd't write the file out either
                return web.Response(status=404)

            return web.FileResponse(str(match))

        self._asset_server.router.add_get("/assets/{filename}", serve_asset)

        runner = web.AppRunner(self._asset_server)
        await runner.setup()
        site = web.TCPSite(runner, self.host(), self.asset_port())
        await site.start()
        self.log.info(f"*  Asset server running on http://{self.address()}:{self.asset_port()}/assets/")
        
    async def _start(self):
        self.log.info("Waiting for a client to connect…")
        await self._client_connected.wait()
        self.log.info("Client connected")

    async def run(self):
        asyncio.ensure_future(self._start())
        asyncio.ensure_future(self._serve_assets())
        async with websockets.serve(self._handler, self.host(), self.port()):
            self.log.info(f"*  Scene server running on ws://{self.address()}:{self.port()}")
            await asyncio.Future()

    async def broadcast(self, message: ClientMessages, payload: dict, exclude=None):
        if not self.clients:
            return
        data = json.dumps({
            "type": str(message),
            "payload": payload
        })
        targets = [c for c in self.clients if c != exclude]
        if targets:
            results = await asyncio.gather(
                *[c.send(data) for c in targets],
                return_exceptions=True
            )
            for c, result in zip(targets, results):
                if isinstance(result, Exception):
                    self.log.warning(f"[!] Failed to send to client: {result}")
                    self.clients.discard(c)

    async def report_progress(self, step: str, percent: float, detail: str = ""):
        self.log.info(f"[{int(percent * 100):3d}%] {step}" + (f" — {detail}" if detail else ""))
        await self.broadcast(ClientMessages.PROGRESS, {
            "step":    step,
            "percent": round(percent, 4),
            "detail":  detail,
        })

    def get_snapshot(self) -> dict:
        snapshot = {"scene": self.scene.encode(), "scene_id": self._scene_id}
        if self._splat_material is not None:
            snapshot["terrain_material"] = self._splat_material.encode()
        return snapshot

    async def _handler(self, ws):
        client_id = str(uuid.uuid4())[:8]
        self.clients.add(ws)
        self.log.info(f"[+] {client_id} connected ({ws.remote_address})")
        self._client_connected.set()

        try:
            async for raw in ws:
                self.log.info(f"Raw message: {repr(raw)}")  # add this temporarily
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    self.log.warning(f"Bad JSON from {client_id}")
                    continue

                msg_type = msg.get("type")
                payload  = msg.get("payload", {})

                if msg_type == ServerMessages.CLIENT_READY:
                    self.log.info(f"{client_id} is ready")
                    if self._scene_id is not None:
                        # Scene already generated — resend the cached snapshot instead of
                        # regenerating. The client compares scene_id and skips reload if unchanged.
                        await self.broadcast(ClientMessages.SCENE_INIT, self.get_snapshot())
                    else:
                        await self._request_pipeline()

                elif msg_type == ServerMessages.OBJECT_EVENT:
                    self.log.info(f"{client_id}: {payload}")
                    await self._handle_object_event(payload)

                else:
                    self.log.warning(f"Unknown type '{msg_type}' from {client_id}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.log.info(f"{client_id} disconnected")

    async def _request_pipeline(self):
        # Cancel the running pipeline if there is one
        if self._pipeline_task and not self._pipeline_task.done():
            self.log.info("Cancelling running pipeline")
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass  # expected
            await self.broadcast(ClientMessages.PIPELINE_CANCELLED, {
                "message": "Pipeline cancelled — starting new run"
            })

        self._pipeline_task = asyncio.ensure_future(self._progress_scene())

    async def _progress_scene(self):
        if self.pipeline is None:
            # Local mode: serve from the pre-loaded context without running the pipeline.
            if self._context is not None:
                self.scene = self._context.scene(ContextKey.SCENE)
                self._splat_material = self._context.splat_material(ContextKey.TERRAIN_MATERIAL)
                self._scene_id = str(uuid.uuid4())
                self.log.info("Serving pre-loaded scene")
                await self.broadcast(ClientMessages.SCENE_INIT, self.get_snapshot())
            else:
                await self.broadcast(ClientMessages.PIPELINE_ERROR, {"message": "No context loaded"})
            return

        self.log.info("Starting pipeline")
        progress_queue = queue.SimpleQueue()

        async def drain():
            while True:
                try:
                    update = await asyncio.get_running_loop().run_in_executor(None, progress_queue.get)
                    if update is None:
                        break
                    await self.broadcast(ClientMessages.PROGRESS, update)
                except asyncio.CancelledError:
                    progress_queue.put(None)  # unblock the queue.get in executor
                    raise
                except Exception as e:
                    self.log.error(f"Progress drain error: {e}")
                    break

        drain_task = asyncio.ensure_future(drain())

        # Deferred: pulls in pipeline.pipeline (every stage class, transitively
        # torch/transformers/diffusers/...) -- only worth paying for when a pipeline
        # actually needs to run, never in `local` mode (self.pipeline is None, handled
        # above, before this branch is reached at all).
        from pipeline.pipeline_runner import PipelineRunner

        input = PipelineInput(self._input_path)
        runner = PipelineRunner(self.pipeline)

        def run_pipeline():
            return runner.run(input, progress_queue)

        try:
            context_result = await asyncio.get_running_loop().run_in_executor(None, run_pipeline)
            self.scene = context_result.scene(ContextKey.SCENE)
            self._splat_material = context_result.splat_material(ContextKey.TERRAIN_MATERIAL)
            self._context = context_result
            self._scene_id = str(uuid.uuid4())
        except asyncio.CancelledError:
            progress_queue.put(None)   # unblock drain
            await drain_task
            self.log.info("Pipeline cancelled")
            raise   # must re-raise so the Task is properly marked cancelled
        except Exception as e:
            self.log.error(f"Pipeline error: {e}", exc_info=True)
            progress_queue.put(None)
            await drain_task
            await self.broadcast(ClientMessages.PIPELINE_ERROR, {"message": str(e)})
            return

        progress_queue.put(None)
        await drain_task

        self.log.info("Pipeline complete — sending scene")
        await self.broadcast(ClientMessages.SCENE_INIT, self.get_snapshot())
 
    async def _handle_object_event(self, payload: dict):
        # Do we need to handle anything directly from the unity scene here?
        pass