from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from auth_utils import JWT_ALGORITHM
from config import Settings, get_settings
from models.user import Role, User
from scripts.seed_data import seed
from tests.conftest import PATIENT


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_register_creates_patient_and_returns_working_token(client: AsyncClient) -> None:
    response = await client.post("/api/auth/register", json=PATIENT)

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["role"] == "patient"
    assert body["user"]["email"] == PATIENT["email"]
    assert "password" not in body["user"] and "password_hash" not in body["user"]

    me = await client.get("/api/auth/me", headers=_bearer(body["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == PATIENT["email"]


async def test_password_is_stored_hashed(client: AsyncClient) -> None:
    await client.post("/api/auth/register", json=PATIENT)

    user = await User.find_one(User.email == PATIENT["email"])
    assert user is not None
    assert user.password_hash != PATIENT["password"]
    assert user.password_hash.startswith("$2")


@pytest.mark.guardrail
@pytest.mark.parametrize("role", ["doctor", "admin"])
async def test_cannot_self_register_as_doctor_or_admin(client: AsyncClient, role: str) -> None:
    response = await client.post("/api/auth/register", json={**PATIENT, "role": role})

    assert response.status_code == 201
    assert response.json()["user"]["role"] == "patient"
    user = await User.find_one(User.email == PATIENT["email"])
    assert user is not None and user.role == Role.PATIENT


@pytest.mark.guardrail
@pytest.mark.parametrize("role", ["doctor", "admin"])
async def test_no_role_specific_register_endpoint(client: AsyncClient, role: str) -> None:
    response = await client.post(f"/api/auth/register/{role}", json=PATIENT)

    # 404 normally; 405 when the built frontend's catch-all page route is mounted
    assert response.status_code in (404, 405)
    assert await User.count() == 0


@pytest.mark.parametrize("phone", ["12345", "1234567890", "98765432100", "98765abcde"])
async def test_register_rejects_invalid_indian_phone(client: AsyncClient, phone: str) -> None:
    response = await client.post("/api/auth/register", json={**PATIENT, "phone": phone})

    assert response.status_code == 422


async def test_register_rejects_short_password(client: AsyncClient) -> None:
    response = await client.post("/api/auth/register", json={**PATIENT, "password": "short"})

    assert response.status_code == 422


async def test_register_rejects_duplicate_email_case_insensitively(client: AsyncClient) -> None:
    await client.post("/api/auth/register", json=PATIENT)

    response = await client.post(
        "/api/auth/register", json={**PATIENT, "email": PATIENT["email"].upper()}
    )

    assert response.status_code == 409


async def test_login_succeeds_with_correct_password(client: AsyncClient) -> None:
    await client.post("/api/auth/register", json=PATIENT)

    response = await client.post(
        "/api/auth/login", json={"email": PATIENT["email"], "password": PATIENT["password"]}
    )

    assert response.status_code == 200
    assert response.json()["access_token"]


@pytest.mark.parametrize(
    ("email", "password"),
    [(PATIENT["email"], "wrong-password"), ("nobody@example.com", PATIENT["password"])],
)
async def test_login_rejects_bad_credentials(
    client: AsyncClient, email: str, password: str
) -> None:
    await client.post("/api/auth/register", json=PATIENT)

    response = await client.post("/api/auth/login", json={"email": email, "password": password})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


async def test_seeded_staff_can_login_with_their_role(client: AsyncClient) -> None:
    await seed()
    password = get_settings().seed_default_password

    doctor = await client.post(
        "/api/auth/login", json={"email": "asha.rao@careflow.example", "password": password}
    )
    admin = await client.post(
        "/api/auth/login", json={"email": "admin@careflow.example", "password": password}
    )

    assert doctor.json()["user"]["role"] == "doctor"
    assert doctor.json()["user"]["department"] == "General Medicine"
    assert admin.json()["user"]["role"] == "admin"


async def test_me_requires_a_token(client: AsyncClient) -> None:
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_me_rejects_garbage_and_wrongly_signed_tokens(client: AsyncClient) -> None:
    registered = await client.post("/api/auth/register", json=PATIENT)
    user_id = registered.json()["user"]["id"]
    forged = jwt.encode(
        {"sub": user_id, "exp": datetime.now(UTC) + timedelta(hours=1)},
        "a-different-secret-that-is-also-32-bytes",
        algorithm=JWT_ALGORITHM,
    )

    assert (await client.get("/api/auth/me", headers=_bearer("not-a-jwt"))).status_code == 401
    assert (await client.get("/api/auth/me", headers=_bearer(forged))).status_code == 401


async def test_me_rejects_expired_token(client: AsyncClient) -> None:
    registered = await client.post("/api/auth/register", json=PATIENT)
    expired = jwt.encode(
        {"sub": registered.json()["user"]["id"], "exp": datetime.now(UTC) - timedelta(minutes=1)},
        get_settings().jwt_secret,
        algorithm=JWT_ALGORITHM,
    )

    assert (await client.get("/api/auth/me", headers=_bearer(expired))).status_code == 401


async def test_token_expires_after_configured_hours(client: AsyncClient) -> None:
    registered = await client.post("/api/auth/register", json=PATIENT)
    settings = get_settings()

    claims = jwt.decode(
        registered.json()["access_token"], settings.jwt_secret, algorithms=[JWT_ALGORITHM]
    )

    assert claims["exp"] - claims["iat"] == settings.jwt_expiry_hours * 3600


async def test_auth_endpoints_are_rate_limited(client: AsyncClient) -> None:
    credentials = {"email": "nobody@example.com", "password": "wrong-password"}

    statuses = [
        (await client.post("/api/auth/login", json=credentials)).status_code for _ in range(6)
    ]

    assert statuses == [401, 401, 401, 401, 401, 429]


@pytest.mark.guardrail
@pytest.mark.parametrize("environment", ["development", "demo"])
def test_mock_db_is_refused_outside_tests(environment: str) -> None:
    with pytest.raises(ValidationError, match="USE_MOCK_DB"):
        Settings(environment=environment, use_mock_db=True, jwt_secret="x" * 40)
