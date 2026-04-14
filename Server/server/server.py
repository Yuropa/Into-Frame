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
from scene.object import Object

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
        """
        Waits for the first client, runs the pipeline, then sends the full scene.
        Progress is pushed to the client throughout via report_progress().
        """
        self.log.info("Waiting for a client to connect…")
        await self._client_connected.wait()
        self.log.info("Client connected — starting pipeline")
 
        progress_queue = queue.SimpleQueue()

        # Drain the queue and broadcast to Unity as fast as updates arrive
        async def drain():
            loop = asyncio.get_event_loop()
            while True:
                try:
                    update = await loop.run_in_executor(None, progress_queue.get)
                    if update is None: 
                        break
                    await self.broadcast(ClientMessages.PROGRESS, update)
                except Exception as e:
                    self.log.error(f"Progress drain error: {e}")
                    break

        drain_task = asyncio.ensure_future(drain())

        # Run the blocking pipeline in a thread so the event loop stays free
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.pipeline.run, progress_queue)
        except Exception as e:
            self.log.error(f"[!] Pipeline error: {e}")
            progress_queue.put(None)          # unblock the drain task
            await drain_task
            await self.broadcast(ClientMessages.PIPELINE_ERROR, {"message": str(e)})
            return

        progress_queue.put(None)              # signal drain to stop
        await drain_task                      # wait for all progress to flush

        self.log.info("Pipeline complete — sending scene")
        await self.broadcast(ClientMessages.SCENE_INIT, self.get_snapshot())

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
        targets = [c for c in self.clients if c != exclude and c.open]
        if targets:
            await asyncio.gather(*[c.send(data) for c in targets], return_exceptions=True)

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
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    self.log.warning(f"Bad JSON from {client_id}")
                    continue
 
                msg_type = msg.get("type")
                payload  = msg.get("payload", {})
 
                if msg_type == ServerMessages.CLIENT_READY:
                    # Client is ready to receive — pipeline will push PROGRESS
                    # messages and SCENE_INIT when done. Nothing to send yet.
                    self.log.info(f"{client_id} is ready")
 
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
 
    async def _handle_object_event(self, payload: dict):
        # Do we need to handle anything directly from the unity scene here?
        pass