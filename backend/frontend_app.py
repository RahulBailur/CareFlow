from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_frontend(api: FastAPI, build_dir: Path) -> bool:
    """Serve the built React app from the same server. Returns False if there is no build.

    Must be called after every API router is registered: the catch-all route is last on purpose.
    """
    build_dir = build_dir.resolve()
    index = build_dir / "index.html"
    if not index.is_file():
        return False

    assets = build_dir / "assets"
    if assets.is_dir():
        api.mount("/assets", StaticFiles(directory=assets), name="assets")

    @api.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path == "api" or path.startswith(("api/", "socket.io")):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
        candidate = (build_dir / path).resolve()
        # A real file inside the build (favicon etc.) is served as is; never anything outside it
        if candidate.is_file() and candidate.is_relative_to(build_dir):
            return FileResponse(candidate)
        # Any other path is a client-side route, so the React app handles it
        return FileResponse(index)

    return True
