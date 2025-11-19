# api/src/cv/websocket_stream.py
"""
WebSocket streaming for real-time CV pipeline processing.

Provides real-time video analysis with immediate feedback on:
- Player detections and tracking
- Shot events (attempts, makes, misses)
- Court positions and ball tracking

Usage:
    # Server
    python -m api.src.cv.websocket_stream

    # Client (JavaScript)
    const ws = new WebSocket('ws://localhost:8765');
    ws.send(frameData);  // Send frame as binary
    ws.onmessage = (event) => {
        const result = JSON.parse(event.data);
        // Handle detections, shots, etc.
    };
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StreamState:
    """State for a streaming session."""
    session_id: str
    frame_count: int = 0
    last_frame_time: float = 0.0
    fps_estimate: float = 30.0

    # Accumulated events
    shot_events: List[Dict] = field(default_factory=list)
    active_tracks: Dict[int, Dict] = field(default_factory=dict)

    # Processing metrics
    total_processing_time: float = 0.0
    frames_processed: int = 0


@dataclass
class WebSocketStreamServer:
    """
    WebSocket server for real-time CV processing.

    Accepts video frames, processes them through the CV pipeline,
    and streams back results immediately.
    """
    host: str = "0.0.0.0"
    port: int = 8765
    frame_skip: int = 1
    buffer_size: int = 30
    jpeg_quality: int = 80

    # Pipeline components
    _cfg = None
    _player_model = None
    _court_model = None
    _tracker = None
    _sessions: Dict[str, StreamState] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize state."""
        self._sessions = {}

    async def initialize_pipeline(self) -> bool:
        """
        Initialize CV pipeline components.

        Returns:
            True if initialization successful
        """
        try:
            from api.src.cv.config import CVConfig, load_models
            from api.src.cv.tracker import create_tracker

            self._cfg = CVConfig()
            self._player_model, self._court_model = load_models(self._cfg)
            self._tracker = create_tracker(self._cfg)

            print(f"[INFO][websocket] Pipeline initialized")
            return True

        except Exception as e:
            print(f"[ERROR][websocket] Pipeline init failed: {e}")
            return False

    async def process_frame(
        self,
        frame: np.ndarray,
        session: StreamState,
    ) -> Dict[str, Any]:
        """
        Process a single frame through the pipeline.

        Args:
            frame: BGR image
            session: Session state

        Returns:
            Dict with detections and events
        """
        import cv2
        from api.src.cv.shot_pipeline import detect_players

        start_time = time.time()
        result = {
            "frame_idx": session.frame_count,
            "timestamp": time.time(),
            "players": [],
            "ball": None,
            "shot_event": None,
        }

        try:
            # Player detection
            if self._player_model is not None:
                dets = detect_players(frame, self._player_model, self._cfg)

                if dets is not None and len(dets) > 0:
                    # Track players
                    if self._tracker is not None:
                        tracked = self._tracker.update(dets, session.frame_count)

                        # Format player data
                        for i in range(len(tracked)):
                            player = {
                                "track_id": int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1,
                                "bbox": tracked.xyxy[i].tolist(),
                                "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                            }
                            result["players"].append(player)

                            # Update active tracks
                            if tracked.tracker_id is not None:
                                tid = int(tracked.tracker_id[i])
                                session.active_tracks[tid] = {
                                    "last_frame": session.frame_count,
                                    "bbox": tracked.xyxy[i].tolist(),
                                }
                    else:
                        # No tracking, just detections
                        for i in range(len(dets)):
                            result["players"].append({
                                "track_id": -1,
                                "bbox": dets.xyxy[i].tolist(),
                                "confidence": float(dets.confidence[i]) if dets.confidence is not None else 1.0,
                            })

        except Exception as e:
            result["error"] = str(e)

        # Update metrics
        proc_time = time.time() - start_time
        session.total_processing_time += proc_time
        session.frames_processed += 1
        result["processing_ms"] = proc_time * 1000

        # Estimate FPS
        if session.last_frame_time > 0:
            dt = time.time() - session.last_frame_time
            if dt > 0:
                session.fps_estimate = 0.9 * session.fps_estimate + 0.1 * (1.0 / dt)

        session.last_frame_time = time.time()
        result["fps_estimate"] = session.fps_estimate

        return result

    async def handle_connection(self, websocket, path):
        """
        Handle a WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Request path
        """
        import uuid
        import cv2

        session_id = str(uuid.uuid4())[:8]
        session = StreamState(session_id=session_id)
        self._sessions[session_id] = session

        print(f"[INFO][websocket] Client connected: {session_id}")

        try:
            async for message in websocket:
                # Decode frame
                if isinstance(message, bytes):
                    # Binary frame data (JPEG or raw)
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send(json.dumps({
                            "error": "Invalid frame data"
                        }))
                        continue

                elif isinstance(message, str):
                    # JSON command
                    try:
                        cmd = json.loads(message)

                        if cmd.get("type") == "ping":
                            await websocket.send(json.dumps({
                                "type": "pong",
                                "session_id": session_id,
                            }))
                            continue

                        elif cmd.get("type") == "reset":
                            if self._tracker:
                                self._tracker.reset()
                            session.shot_events.clear()
                            session.active_tracks.clear()
                            await websocket.send(json.dumps({
                                "type": "reset_ack",
                            }))
                            continue

                        elif cmd.get("type") == "stats":
                            await websocket.send(json.dumps({
                                "type": "stats",
                                "frames_processed": session.frames_processed,
                                "total_time": session.total_processing_time,
                                "avg_fps": session.frames_processed / session.total_processing_time if session.total_processing_time > 0 else 0,
                                "active_tracks": len(session.active_tracks),
                            }))
                            continue

                    except json.JSONDecodeError:
                        pass

                    continue

                else:
                    continue

                # Skip frames if needed
                if self.frame_skip > 1 and session.frame_count % self.frame_skip != 0:
                    session.frame_count += 1
                    continue

                # Process frame
                result = await self.process_frame(frame, session)
                session.frame_count += 1

                # Send result
                await websocket.send(json.dumps(result))

        except Exception as e:
            print(f"[ERROR][websocket] Connection error: {e}")

        finally:
            del self._sessions[session_id]
            print(f"[INFO][websocket] Client disconnected: {session_id}")

    async def start(self):
        """Start the WebSocket server."""
        import websockets

        # Initialize pipeline
        await self.initialize_pipeline()

        print(f"[INFO][websocket] Starting server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max message
        ):
            await asyncio.Future()  # Run forever


def create_websocket_server(cfg=None) -> WebSocketStreamServer:
    """
    Create WebSocket server from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured WebSocketStreamServer
    """
    if cfg is None:
        return WebSocketStreamServer()

    return WebSocketStreamServer(
        host=getattr(cfg, "websocket_host", "0.0.0.0"),
        port=getattr(cfg, "websocket_port", 8765),
        frame_skip=getattr(cfg, "streaming_frame_skip", 1),
        buffer_size=getattr(cfg, "streaming_buffer_size", 30),
        jpeg_quality=getattr(cfg, "streaming_quality", 80),
    )


async def main():
    """Main entry point."""
    from api.src.cv.config import CVConfig

    cfg = CVConfig()
    server = create_websocket_server(cfg)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
