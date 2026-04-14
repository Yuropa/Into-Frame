import asyncio
import json
import math
import uuid
import threading
import queue
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any
from pipeline.pipeline import Pipeline
from server.messages import ServerMessages, ClientMessages
from scene.scene import Scene
from scene.object import Object3D

import websockets

class SimulationServerConfiguration():
    def __init__(self) -> None:
        self.address = "localhost"
        self.port = 8080
        self.log = None

class SimulationServer():
    def __init__(self, config: SimulationServerConfiguration, pipeline: Pipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.log = config.log
        self.clients: set = set()

        self.scene = Scene()

        self._pipeline_task: asyncio.Task | None = None
        self._client_connected = asyncio.Event()

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
        
    async def _start(self):
        self.log.info("Waiting for a client to connect…")
        await self._client_connected.wait()
        self.log.info("Client connected")

    async def run(self):
        asyncio.ensure_future(self._start())
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
            await asyncio.get_running_loop().run_in_executor(None, self.pipeline.run, progress_queue)
        except asyncio.CancelledError:
            progress_queue.put(None)   # unblock drain
            await drain_task
            self.log.info("Pipeline cancelled")
            raise   # must re-raise so the Task is properly marked cancelled
        except Exception as e:
            self.log.error(f"Pipeline error: {e}")
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