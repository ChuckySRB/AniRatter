"""Convenience launcher: serve the web/ directory on http://localhost:8000."""
from __future__ import annotations

import http.server
import socketserver

import config

PORT = 8000


def main() -> None:
    if not (config.WEB_DIR / "index.html").exists():
        raise SystemExit(f"missing {config.WEB_DIR / 'index.html'}")

    handler = http.server.SimpleHTTPRequestHandler

    class Handler(handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(config.WEB_DIR), **kwargs)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"serving {config.WEB_DIR} at http://localhost:{PORT}/  (Ctrl-C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
