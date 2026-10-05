from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from frontend_app import mount_frontend


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text("<html>the app</html>")
    (build / "assets" / "app.js").write_text("console.log('app')")
    (build / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside the build")
    return build


def _app(build: Path) -> FastAPI:
    api = FastAPI()

    @api.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    assert mount_frontend(api, build) is True
    return api


async def _get(app: FastAPI, path: str) -> tuple[int, str]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.get(path)
        return response.status_code, response.text


async def test_serves_the_app_shell_assets_and_root_files(build_dir: Path) -> None:
    app = _app(build_dir)

    assert await _get(app, "/") == (200, "<html>the app</html>")
    assert await _get(app, "/assets/app.js") == (200, "console.log('app')")
    assert await _get(app, "/favicon.svg") == (200, "<svg/>")


async def test_client_side_routes_fall_back_to_the_app_shell(build_dir: Path) -> None:
    app = _app(build_dir)

    assert await _get(app, "/book") == (200, "<html>the app</html>")
    assert await _get(app, "/history") == (200, "<html>the app</html>")


async def test_api_routes_are_not_swallowed_by_the_fallback(build_dir: Path) -> None:
    app = _app(build_dir)

    assert (await _get(app, "/api/health"))[0] == 200
    assert (await _get(app, "/api/does-not-exist"))[0] == 404


@pytest.mark.guardrail
@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"])
async def test_files_outside_the_build_are_never_served(build_dir: Path, path: str) -> None:
    status_code, body = await _get(_app(build_dir), path)

    assert "outside the build" not in body
    assert status_code in (200, 404)


def test_nothing_is_mounted_without_a_build(tmp_path: Path) -> None:
    api = FastAPI()

    assert mount_frontend(api, tmp_path / "missing") is False
    assert [route for route in api.routes if getattr(route, "path", "") == "/{path:path}"] == []
