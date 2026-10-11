"""Load test: many patients using CareBot by text at once.

Each simulated patient registers, then chats in one conversation and now and then checks
availability and their own appointments, with a pause of 1-3 s between actions.

The app's rate limits are per IP, and every simulated patient comes from this one machine,
so run against the app with the limits raised (and, for the mocked run, no LLM):

    docker compose -f docker-compose.yml -f backend/loadtest/docker-compose.loadtest.yml up -d
    locust -f backend/loadtest/locustfile.py --host http://localhost:8000 \\
        --headless --users 50 --spawn-rate 10 --run-time 2m --csv backend/loadtest/results/run

Add --tags chat (or availability, me) to load one kind of request on its own.

At the end it prints which tier answered the chat turns (cache, gemini, rules...).
"""

import random
import secrets
from collections import Counter
from typing import Any

from locust import HttpUser, between, events, tag, task

QUESTIONS = [
    "What are the OPD timings?",
    "Where is the cardiology department?",
    "What is the emergency contact number?",
    "I have a fever and cough",
    "My knee hurts when I climb stairs",
    "Show me my last prescription",
    "Do I have any upcoming appointments?",
    "Is Dr. Asha Rao free this Thursday?",
    "hello",
    "ओपीडी का समय क्या है?",
    "ನನಗೆ ಜ್ವರ ಇದೆ",
]

answered_by: Counter[str] = Counter()


class Patient(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        number = secrets.randbelow(10**9)
        response = self.client.post(
            "/api/auth/register",
            json={
                "name": "Load Test Patient",
                "email": f"load-{secrets.token_hex(6)}@example.com",
                "phone": f"9{number:09d}",
                "password": "load-test-password-1",
            },
            name="/api/auth/register",
        )
        token = response.json().get("access_token", "")
        self.client.headers["Authorization"] = f"Bearer {token}"
        self.session_id: str | None = None

    @tag("chat")
    @task(6)
    def chat(self) -> None:
        body: dict[str, Any] = {"message": random.choice(QUESTIONS)}  # noqa: S311 — not security
        if self.session_id:
            body["session_id"] = self.session_id
        with self.client.post("/api/chat", json=body, name="/api/chat", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            data = r.json()
            self.session_id = data["session_id"]
            answered_by[data.get("answered_by", "?")] += 1

    @tag("availability")
    @task(2)
    def availability(self) -> None:
        self.client.get("/api/appointments/availability", name="/api/appointments/availability")

    @tag("me")
    @task(1)
    def my_appointments(self) -> None:
        self.client.get("/api/appointments/me", name="/api/appointments/me")


@events.test_stop.add_listener
def report_tiers(**_: Any) -> None:
    total = sum(answered_by.values())
    if total:
        shares = ", ".join(
            f"{tier} {count / total:.0%}" for tier, count in answered_by.most_common()
        )
        print(f"\nChat turns answered by: {shares} ({total} turns)")
