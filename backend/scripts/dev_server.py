"""Run CareFlow with an in-memory database, already seeded. No MongoDB or Docker needed.

For looking at the app while developing. Everything is lost when the server stops.

    python scripts/dev_server.py
"""

import asyncio
import os

# The in-memory database is only allowed in the test environment, so this is a test-mode server
os.environ["ENVIRONMENT"] = "test"
os.environ["USE_MOCK_DB"] = "true"
os.environ.setdefault("JWT_SECRET", "dev-server-secret-not-for-any-real-deployment")
os.environ.setdefault("SEED_DEFAULT_PASSWORD", "careflow-demo-1")
os.environ.setdefault("BCRYPT_ROUNDS", "4")
os.environ["REDIS_URL"] = ""  # no Redis here: the response cache is off

PORT = 8000


async def main() -> None:
    import uvicorn

    from config import get_settings
    from database import init_db
    from main import app
    from scripts.seed_data import ADMIN, DOCTORS, PATIENTS, seed

    await init_db()
    await seed()

    password = get_settings().seed_default_password
    print(f"\nCareFlow dev server — http://localhost:{PORT}  (in-memory data, test mode)")
    print(f"Every account uses the password: {password}")
    print(f"  patient  {PATIENTS[0][1]}")
    print(f"  doctor   {DOCTORS[0][1]}")
    print(f"  admin    {ADMIN[1]}\n")

    # lifespan is off because the database above is already initialised and seeded
    server = uvicorn.Server(uvicorn.Config(app, port=PORT, lifespan="off"))
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
