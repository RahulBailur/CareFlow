"""The admin analytics API: department load today, and how CareBot has been doing."""

from typing import Any

import pytest
from httpx import AsyncClient

from agents.intent_classifier import IntentClassifier
from agents.orchestrator import run_text_turn
from models.appointment import AppointmentStatus
from models.doctor_status import DoctorStatus
from models.turn_analytics import TurnAnalytics
from models.user import Role
from services.analytics import percentile, turn_summary
from services.failover import CircuitBreaker, FailoverLLM, error_kind
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot
from tests.test_chat import seed_hospital
from tests.test_failover import Tier
from time_utils import today_ist

# --- who may see it ---------------------------------------------------------------------


@pytest.mark.guardrail
@pytest.mark.parametrize("path", ["/api/admin/stats", "/api/admin/analytics"])
@pytest.mark.parametrize("role", [Role.PATIENT, Role.DOCTOR])
async def test_only_admins_see_analytics(client: AsyncClient, path: str, role: Role) -> None:
    response = await client.get(path, headers=auth(await make_user(role)))

    assert response.status_code == 403


@pytest.mark.parametrize("path", ["/api/admin/stats", "/api/admin/analytics"])
async def test_analytics_need_a_login(client: AsyncClient, path: str) -> None:
    assert (await client.get(path)).status_code == 401


# --- department load --------------------------------------------------------------------


async def test_department_load_counts_todays_queue(client: AsyncClient) -> None:
    await seed_hospital()
    admin, doctor, other_doctor = (
        await make_user(Role.ADMIN),
        await make_doctor("Dr. Busy"),  # Cardiology, on duty every day
        await make_doctor("Dr. Quiet"),
    )
    patients = [await make_user() for _ in range(5)]
    await make_appointment(patients[0], doctor, slot(0, 0), AppointmentStatus.DONE)
    await make_appointment(patients[1], doctor, slot(0, 1), AppointmentStatus.IN_CONSULTATION)
    await make_appointment(patients[2], doctor, slot(0, 2))
    await make_appointment(patients[3], doctor, slot(0, 3), AppointmentStatus.CANCELLED)
    await make_appointment(patients[4], doctor, slot(1, 0))  # tomorrow: not today's load
    await DoctorStatus(doctor_id=doctor.id, day=today_ist().isoformat(), delay_minutes=25).insert()
    await DoctorStatus(doctor_id=other_doctor.id, day="2000-01-01", delay_minutes=90).insert()

    body = (await client.get("/api/admin/stats", headers=auth(admin))).json()

    cardiology = next(row for row in body["departments"] if row["department"] == "Cardiology")
    assert cardiology == {
        "department": "Cardiology",
        "doctors": 2,
        "doctors_on_duty": 2,
        "appointments_today": 3,
        "waiting": 1,
        "in_consultation": 1,
        "done": 1,
        "no_show": 0,
        "doctors_running_late": 1,  # an old delay does not count
        "longest_delay_minutes": 25,
    }
    ent = next(row for row in body["departments"] if row["department"] == "ENT")
    assert ent["doctors"] == 0 and ent["appointments_today"] == 0
    assert body["totals"]["appointments_today"] == 3 and body["totals"]["waiting"] == 1


# --- CareBot analytics ------------------------------------------------------------------


def turn(**fields: Any) -> TurnAnalytics:
    defaults: dict[str, Any] = {
        "pipeline": "text",
        "intent": "support",
        "language": "en",
        "routed_by": "keywords",
        "used_llm": True,
        "answered_by": "gemini",
        "blocked": False,
        "tools_called": [],
        "stt_provider": "",
        "tts_provider": "",
        "spoken": False,
        "barged_in": False,
        "stages": {"total_ms": 1000.0},
    }
    return TurnAnalytics(**(defaults | fields))


async def test_latency_is_reported_per_pipeline_and_stage(client: AsyncClient) -> None:
    for ttfa in (2000.0, 2500.0, 9000.0):
        await turn(pipeline="live", answered_by="gemini-live", stages={"ttfa_ms": ttfa}).insert()
    await turn(pipeline="cascade", stages={"ttfa_ms": 4000.0, "stt_ms": 1000.0}).insert()

    summary = await turn_summary(days=7)

    live = summary["pipelines"]["live"]
    assert live["turns"] == 3
    assert live["stages"]["ttfa_ms"] == {"n": 3, "p50": 2500.0, "p95": 9000.0}
    assert summary["pipelines"]["cascade"]["stages"]["stt_ms"]["p50"] == 1000.0


async def test_the_fallback_rate_counts_turns_no_first_choice_model_answered(
    client: AsyncClient,
) -> None:
    for tier in ("gemini", "gemini", "gemini-live", "cache", "rules", "rules", "ollama", "rules"):
        await turn(answered_by=tier).insert()

    summary = await turn_summary(days=7)

    assert summary["turns"] == 8
    assert summary["fallback_rate"] == pytest.approx(4 / 8)
    assert summary["cache_rate"] == pytest.approx(1 / 8)
    assert summary["primary_rate"] == pytest.approx(3 / 8)
    assert summary["answered_by"]["rules"] == 3


async def test_intents_errors_and_safety_events_are_counted(client: AsyncClient) -> None:
    await turn(intent="triage", blocked=True, llm_errors=["gemini:429"]).insert()
    await turn(intent="triage", llm_errors=["gemini:429", "gemini:circuit_open"]).insert()
    await turn(intent="booking", pipeline="cascade", spoken=False, barged_in=True).insert()

    summary = await turn_summary(days=7)

    assert summary["intents"] == {"triage": 2, "booking": 1}
    assert summary["llm_errors"] == {"gemini:429": 2, "gemini:circuit_open": 1}
    assert summary["turns_with_llm_errors"] == 2
    assert summary["blocked"] == 1 and summary["barged_in"] == 1 and summary["unspoken"] == 1


async def test_only_turns_inside_the_window_count(client: AsyncClient) -> None:
    from datetime import timedelta

    from time_utils import now_utc

    await turn().insert()
    await turn(created_at=now_utc() - timedelta(days=10)).insert()

    assert (await turn_summary(days=7))["turns"] == 1
    assert (await turn_summary(days=30))["turns"] == 2


async def test_an_empty_window_is_all_zeros(client: AsyncClient) -> None:
    summary = await turn_summary(days=7)

    assert summary["turns"] == 0 and summary["fallback_rate"] == 0.0 and summary["pipelines"] == {}


async def test_the_api_returns_the_summary_and_checks_the_window(client: AsyncClient) -> None:
    admin = await make_user(Role.ADMIN)
    await turn().insert()

    body = (await client.get("/api/admin/analytics?days=1", headers=auth(admin))).json()
    bad = await client.get("/api/admin/analytics?days=1000", headers=auth(admin))

    assert body["turns"] == 1 and body["days"] == 1
    assert bad.status_code == 422


@pytest.mark.guardrail
async def test_analytics_hold_no_words_and_no_user(client: AsyncClient) -> None:
    await seed_hospital()
    patient = await make_user()
    response = await client.post(
        "/api/chat", json={"message": "What are the OPD timings?"}, headers=auth(patient)
    )
    assert response.status_code == 200

    (record,) = await TurnAnalytics.find_all().to_list()
    stored = str(record.model_dump())
    assert record.pipeline == "text" and "total_ms" in record.stages
    assert "OPD" not in stored and "9 am" not in stored and str(patient.id) not in stored


# --- recording LLM errors ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Gemini returned HTTP 429", "429"),
        ("Gemini request failed: ReadTimeout", "timeout"),
        ("Gemini returned HTTP 503", "5xx"),
        ("circuit open", "circuit_open"),
        ("Gemini returned no usable candidate", "error"),
    ],
)
def test_error_kinds(message: str, kind: str) -> None:
    assert error_kind(message) == kind


async def test_a_turn_records_why_it_needed_a_fallback(client: AsyncClient) -> None:
    await seed_hospital()
    llm = FailoverLLM([(Tier("gemini", "Gemini returned HTTP 429"), CircuitBreaker(1, 60))])
    patient = await make_user()

    first = await run_text_turn(
        patient, "What are the OPD timings?", "s-1", llm, IntentClassifier()
    )
    second = await run_text_turn(
        patient, "What are the OPD timings?", "s-1", llm, IntentClassifier()
    )

    assert first.llm_errors == ["gemini:429"] and first.answered_by == "rules"
    assert second.llm_errors == ["gemini:circuit_open"]


def test_percentile_is_shared_with_the_benchmark() -> None:
    from eval.latency_bench import percentile as bench_percentile

    assert bench_percentile is percentile
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


async def test_rates_leave_out_turns_recorded_before_the_tier_was(client: AsyncClient) -> None:
    for tier in ("gemini", "rules", "", "", ""):
        await turn(answered_by=tier).insert()

    summary = await turn_summary(days=7)

    assert summary["turns"] == 5 and summary["turns_without_tier"] == 3
    assert summary["primary_rate"] == 0.5 and summary["fallback_rate"] == 0.5
