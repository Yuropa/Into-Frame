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
from typing import Any, Optional
from pipeline.pipeline import Pipeline, PipelineContext, ContextKey
from server.messages import ServerMessages, ClientMessages
from scene.scene import Scene
from scene.object import Object3D

import websockets

class SimulationServerConfiguration():
    def __init__(self) -> None:
        self.address = "localhost"
        self.port = 8080
        self.asset_port = 3000
        self.log = None

class SimulationServer():
    _context: Optional[PipelineContext]

    def __init__(self, config: SimulationServerConfiguration, pipeline: Pipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.log = config.log
        self.clients: set = set()

        self.asset_dir = pipeline.config.temp / "assets"

        self.scene = Scene()

        self._pipeline_task: asyncio.Task | None = None
        self._client_connected = asyncio.Event()

        self._asset_server = web.Application()
        self._context = None

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
        matches = list(self.asset_dir.glob(f"{filename}.*"))
        if not matches:
            return None
        if len(matches) > 0:
            return matches[0]
        
        return None

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
        return {
            "scene":  self.scene.encode(),
        }

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

        self.pipeline.set_input("../input.jpeg")

        try:
            context_result = await asyncio.get_running_loop().run_in_executor(None, self.pipeline.run, progress_queue)
            self.scene = context_result.scene(ContextKey.SCENE)
            self._context = context_result
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