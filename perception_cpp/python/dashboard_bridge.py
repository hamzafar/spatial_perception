"""
dashboard_bridge.py
====================

Pushes live perception data from your ROS2 callbacks into
perception-dashboard-v2.html over a local WebSocket.

Usage
-----
    from dashboard_bridge import DashboardBridge

    bridge = DashboardBridge()
    bridge.start()          # call once, e.g. in your node's __init__

    # ... inside any ROS callback, whenever you have new values ...
    bridge.push({
        "ego": {
            "heading_deg": imu_pipeline.heading,
            "speed_mps": gnss_pipeline.speed,
            "world_x": gnss_pipeline.position_enu[0],
            "world_y": gnss_pipeline.position_enu[1],
        }
    })

    # ... on shutdown ...
    bridge.stop()

Open perception-dashboard-v2.html in a browser (it auto-connects to
ws://localhost:8765). No changes needed on the HTML side to add new
data — just push whatever keys you have; see the schema comment block at
the top of the HTML file's <script> section for the full field list.

WHICH FIELD NEEDS WHAT VALUE
-----------------------------
fps            float                  inference FPS
latency_ms     float                  end-to-end pipeline latency (ms)
objects_count  {vehicle,person,cyclist: int}
gpu_pct        float 0-100
cpu_pct        float 0-100
sensors        {cam,lidar,radar,gnss,imu: bool}   True = publishing/healthy

cameras.<front|left|rear|right>.boxes = [
    {
        "cls":  "vehicle" | "person" | "truck" | "cyclist",
        "id":   int,            # ByteTrack ID
        "conf": float 0-1,
        "box":  [cx, cy, w, h], # ALL fractions 0-1 of image size, (cx,cy) = box CENTER
    }, ...
]
  Send only the camera(s) whose frame you just processed, e.g.
  push({"cameras": {"front": {"boxes": [...]}}})

ego = {
    "heading_deg":   float 0-360,   # from IMU
    "speed_mps":     float,         # from GNSS
    "accelerating":  bool,
    "braking":       bool,
    "turning_left":  bool,
    "turning_right": bool,
    "world_x":       float,  # meters, ego +X FORWARD, arbitrary session origin
    "world_y":       float,  # meters, ego +Y LEFT
}
  Only send world_x/world_y as scalars — the browser keeps its own trail
  history for the trajectory inset; don't send an array here.

bev_objects = [
    {"id": int, "cls": "vehicle"|"person"|"truck"|"cyclist",
     "x": float, "y": float}   # METERS, EGO-RELATIVE, +X=forward, +Y=left
                                # (same convention as your Phase 9A world frame)
]
  Send the FULL current list each time (this replaces, doesn't merge) —
  push your unified/de-duplicated (Phase 9B) object list here.

nearest_objects = [
    {"id": int, "cls": str, "label": "Car"|"Pedestrian"|"Truck"|"Cyclist",
     "dist_m": float, "speed_mps": float,
     "motion": "approaching"|"receding"|"stationary"}   # from radar
]
  Any order — the browser sorts ascending by dist_m itself.

frame_idx / session / pipeline : int / str / str   (footer/status bar text)

All top-level keys and all nested sub-keys are OPTIONAL on every push();
send only what changed on that callback. Nothing else gets overwritten.
"""

import asyncio
import json
import threading

import websockets


class DashboardBridge:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._loop = None
        self._server = None
        self._clients = set()
        self._thread = None

    # ---- lifecycle ----

    def start(self):
        """Starts the WebSocket server on a background thread. Call once."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_forever())
        except asyncio.CancelledError:
            pass

    async def _serve_forever(self):
        async with websockets.serve(self._handle_client, self.host, self.port) as server:
            self._server = server
            print(f"[DashboardBridge] listening on ws://{self.host}:{self.port}")
            await server.serve_forever()


    async def _handle_client(self, websocket):
        self._clients.add(websocket)

        print(
            f"[DashboardBridge] Client connected: "
            f"{websocket.remote_address}"
        )

        try:
            async for _ in websocket:
                pass

        finally:
            self._clients.discard(websocket)

            print(
                f"[DashboardBridge] Client disconnected"
            )

    # async def _handle_client(self, websocket):
    #     self._clients.add(websocket)
    #     try:
    #         async for _ in websocket:  # dashboard doesn't send anything back; drain
    #             pass
    #     finally:
    #         self._clients.discard(websocket)

    # ---- the function you call on each ROS callback ----

    def push(self, data: dict):
        """
        Send a (partial) data update to the dashboard. Thread-safe — call
        this directly from any ROS2 callback, no matter what thread/executor
        it runs on. See the module docstring above for the field schema.
        """
        if self._loop is None:
            return  # start() not called yet, or already stopped
        try:
            payload = json.dumps(data)
        except (TypeError, ValueError) as e:
            print(f"[DashboardBridge] push() got non-serializable data: {e}")
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str):
        if not self._clients:
            return
        dead = []
        for ws in self._clients:
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


# ---------------------------------------------------------------------------
# Standalone smoke test: run `python3 dashboard_bridge.py`, open the HTML,
# and you should see the FPS/heading/nearest-objects panel update every 0.5s.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import time

    bridge = DashboardBridge()
    bridge.start()
    time.sleep(0.5)
    print("Pushing test data every 0.5s. Open the dashboard HTML now. Ctrl+C to stop.")

    x = y = 0.0
    try:
        while True:
            x += random.uniform(-0.3, 0.6)
            y += random.uniform(-0.3, 0.3)
            bridge.push({
                "fps": round(26 + random.uniform(-1, 1), 1),
                "latency_ms": round(38 + random.uniform(-3, 3), 0),
                "sensors": {"cam": True, "lidar": True, "radar": True, "gnss": True, "imu": True},
                "ego": {
                    "heading_deg": round((x * 3) % 360, 1),
                    "speed_mps": round(12 + random.uniform(-1, 1), 1),
                    "accelerating": True, "braking": False,
                    "turning_left": False, "turning_right": True,
                    "world_x": x, "world_y": y,
                },
                "bev_objects": [
                    {"id": 12, "cls": "vehicle", "x": 15, "y": -4},
                    {"id": 14, "cls": "person", "x": 6, "y": 3},
                ],
                "nearest_objects": [
                    {"id": 14, "cls": "person", "label": "Pedestrian",
                     "dist_m": 6.7, "speed_mps": 1.1, "motion": "stationary"},
                    {"id": 12, "cls": "vehicle", "label": "Car",
                     "dist_m": 15.5, "speed_mps": 8.2, "motion": "approaching"},
                ],
            })
            time.sleep(0.5)
    except KeyboardInterrupt:
        bridge.stop()
