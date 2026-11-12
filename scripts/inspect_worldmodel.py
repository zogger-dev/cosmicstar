#!/usr/bin/env python3
"""Launches the interactive Replay & Neural World Model Inspector server.

Usage:
    .venv/bin/python scripts/inspect_worldmodel.py --port 8765
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cosmicstar.worldmodel.inspector import WorldModelInspector

INSPECTOR: WorldModelInspector | None = None
UI_HTML_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "cosmicstar",
        "worldmodel",
        "inspector_ui",
        "index.html",
    )
)
ASSET_DIRS = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "worldmodel", "assets")
    ),
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "cosmicstar",
            "worldmodel",
            "inspector_ui",
            "assets",
        )
    ),
]


class InspectorHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args):
        # Keep console logs clean
        pass

    def _send_json(self, data: dict, status: int = 200):
        raw = json.dumps(data).encode("utf-8")
        accept_enc = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept_enc and len(raw) > 1024:
            body = gzip.compress(raw, compresslevel=5)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def _send_html(self, html_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def do_GET(self):
        assert INSPECTOR is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            with open(UI_HTML_PATH, "rb") as f:
                self._send_html(f.read())
            return

        if path.startswith("/assets/"):
            rel_path = path[len("/assets/") :].lstrip("/")
            if ".." in rel_path:
                self._send_json({"error": "Forbidden"}, status=403)
                return
            for base_dir in ASSET_DIRS:
                full_path = os.path.join(base_dir, rel_path)
                if os.path.isfile(full_path):
                    content_type = "application/octet-stream"
                    if full_path.endswith(".png"):
                        content_type = "image/png"
                    elif full_path.endswith(".json"):
                        content_type = "application/json; charset=utf-8"
                    elif full_path.endswith(".svg"):
                        content_type = "image/svg+xml"
                    elif full_path.endswith(".css"):
                        content_type = "text/css; charset=utf-8"
                    elif full_path.endswith(".js"):
                        content_type = "application/javascript; charset=utf-8"

                    with open(full_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self._send_json({"error": f"Asset not found: {path}"}, status=404)
            return

        if path == "/api/init":
            self._send_json(
                {
                    "games": INSPECTOR.catalog,
                    "checkpoints": INSPECTOR.list_checkpoints(),
                    "active_l1_ckpt": INSPECTOR.active_l1_ckpt,
                    "active_l2_ckpt": INSPECTOR.active_l2_ckpt,
                    "active_observer_ckpt": INSPECTOR.active_observer_ckpt,
                }
            )
            return

        # /api/game/{gid}/observer/{loop_t}
        if path.startswith("/api/game/") and "/observer/" in path:
            parts = path.strip("/").split("/")
            gid = parts[2]
            loop_t = int(parts[4])
            n_branches = int(qs.get("branches", ["32"])[0])
            horizon = int(qs.get("horizon", ["8"])[0])
            self._send_json(
                INSPECTOR.inspect_observer_rollout(
                    gid, loop_t=loop_t, n_branches=n_branches, horizon_steps=horizon
                )
            )
            return

        # /api/game/{gid}/timeline
        if path.startswith("/api/game/") and path.endswith("/timeline"):
            parts = path.strip("/").split("/")
            gid = parts[2]
            mode = qs.get("mode", ["cont"])[0]
            self._send_json(INSPECTOR.get_timeline(gid, mode=mode))
            return

        # /api/game/{gid}/frame/{loop_t}
        if path.startswith("/api/game/") and "/frame/" in path:
            parts = path.strip("/").split("/")
            gid = parts[2]
            loop_t = int(parts[4])
            win_len = int(qs.get("window", ["8"])[0])
            self._send_json(
                INSPECTOR.inspect_frame(gid, loop_t=loop_t, window_len=win_len)
            )
            return

        # /api/token/{tok_idx}
        if path.startswith("/api/token/"):
            parts = path.strip("/").split("/")
            tok_idx = int(parts[2])
            layer = int(qs.get("layer", ["5"])[0])
            head = int(qs.get("head", ["0"])[0])
            self._send_json(
                INSPECTOR.query_token_details(tok_idx=tok_idx, layer=layer, head=head)
            )
            return

        # /api/gram256?owner=...&mat_type=...&i=...&j=...
        if path == "/api/gram256":
            owner = qs.get("owner", ["p1"])[0]
            mat_type = qs.get("mat_type", ["G_cc_256"])[0]
            cell_i = int(qs.get("i", ["0"])[0])
            cell_j = int(qs.get("j", ["0"])[0])
            self._send_json(
                INSPECTOR.query_gram_256(
                    owner=owner, mat_type=mat_type, cell_i=cell_i, cell_j=cell_j
                )
            )
            return

        self._send_json({"error": f"Not found: {path}"}, status=404)

    def do_POST(self):
        assert INSPECTOR is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/api/checkpoint":
            if "l1_ckpt" in body and os.path.exists(body["l1_ckpt"]):
                INSPECTOR.load_l1_checkpoint(body["l1_ckpt"])
            if "l2_ckpt" in body and os.path.exists(body["l2_ckpt"]):
                INSPECTOR.load_l2_checkpoint(body["l2_ckpt"])
            if "observer_ckpt" in body and os.path.exists(body["observer_ckpt"]):
                INSPECTOR.load_observer_checkpoint(body["observer_ckpt"])
            self._send_json(
                {
                    "active_l1_ckpt": INSPECTOR.active_l1_ckpt,
                    "active_l2_ckpt": INSPECTOR.active_l2_ckpt,
                    "active_observer_ckpt": INSPECTOR.active_observer_ckpt,
                }
            )
            return

        # POST /api/game/{gid}/frame/{loop_t} (with counterfactual JSON body)
        if path.startswith("/api/game/") and "/frame/" in path:
            parts = path.strip("/").split("/")
            gid = parts[2]
            loop_t = int(parts[4])
            win_len = int(qs.get("window", ["8"])[0])
            self._send_json(
                INSPECTOR.inspect_frame(
                    gid, loop_t=loop_t, window_len=win_len, counterfactual=body
                )
            )
            return

        self._send_json({"error": f"Not found: {path}"}, status=404)


def main():
    global INSPECTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--l1_ckpt",
        default="data/worldmodel/runs/energy_r16_zerogram/model_step44285.pt",
    )
    ap.add_argument(
        "--l2_ckpt",
        default="data/worldmodel/runs/energy_l2_535g_cont_5k/model_l2_step5000.pt",
    )
    ap.add_argument(
        "--observer_ckpt",
        default="data/worldmodel/runs/observer_v1/model_best.pt",
    )
    args = ap.parse_args()

    print("Initializing CosmicStar WorldModelInspector on GPU...", flush=True)
    manifest_path = os.path.join(ASSET_DIRS[0], "icons", "manifest.json")
    if not os.path.exists(manifest_path):
        try:
            print("Extracting StarCraft II game icons from local CASC storage...", flush=True)
            from extract_icons import extract_icons
            extract_icons(out_dir=os.path.join(ASSET_DIRS[0], "icons"))
        except Exception as e:
            print(f"Warning: could not auto-extract SC2 game icons ({e}); falling back to SVGs.", flush=True)

    INSPECTOR = WorldModelInspector(
        l1_ckpt=args.l1_ckpt, l2_ckpt=args.l2_ckpt, observer_ckpt=args.observer_ckpt
    )
    # Warm up CUDA autograd kernels on first unseen game so first browser load is <140ms
    if INSPECTOR.catalog:
        gid0 = INSPECTOR.catalog[0]["gid"]
        INSPECTOR.inspect_frame(gid0, loop_t=2000, window_len=8)
    server = ThreadingHTTPServer((args.host, args.port), InspectorHTTPHandler)
    print(
        f"WorldModel Inspector running at http://{args.host}:{args.port} "
        f"({len(INSPECTOR.catalog)} games loaded)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
