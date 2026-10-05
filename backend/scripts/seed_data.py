"""Seed synthetic data: hospital config, doctors, an admin, patients, schedules, appointments.

Everything here is invented. No real people, registration numbers or prescriptions.
Safe to run more than once: existing records are left untouched.

    python scripts/seed_data.py
"""

import asyncio
from datetime import UTC, datetime, time, timedelta

from auth_utils import hash_password
from config import get_settings
from database import close_db, init_db
from models.appointment import Appointment, AppointmentStatus
from models.hospital_config import Department, HospitalConfig
from models.schedule import Schedule
from models.user import Role, User
from time_utils import IST

SLOT_MINUTES = 15
OPD_START = "09:00"
OPD_END = "13:00"
OPD_WEEKDAYS = range(0, 6)  # Monday–Saturday

DEPARTMENTS = [
    ("General Medicine", "Ground floor, Block A"),
    ("Cardiology", "First floor, Block A"),
    ("Orthopedics", "Ground floor, Block B"),
    ("Pediatrics", "First floor, Block B"),
    ("Dermatology", "Second floor, Block A"),
    ("ENT", "Second floor, Block B"),
]

# (name, email, phone, department, synthetic registration number)
DOCTORS = [
    ("Dr. Asha Rao", "asha.rao@careflow.example", "9000000001", "General Medicine", "SYN-0001"),
    ("Dr. Vikram Shetty", "vikram.shetty@careflow.example", "9000000002", "Cardiology", "SYN-0002"),
    ("Dr. Meera Iyer", "meera.iyer@careflow.example", "9000000003", "Orthopedics", "SYN-0003"),
    ("Dr. Arjun Hegde", "arjun.hegde@careflow.example", "9000000004", "Pediatrics", "SYN-0004"),
    ("Dr. Kavya Nair", "kavya.nair@careflow.example", "9000000005", "Dermatology", "SYN-0005"),
    ("Dr. Rohan Kulkarni", "rohan.kulkarni@careflow.example", "9000000006", "ENT", "SYN-0006"),
]

ADMIN = ("CareFlow Admin", "admin@careflow.example", "9000000100")

PATIENTS = [
    ("Ananya Sharma", "ananya@careflow.example", "9000000201"),
    ("Kiran Gowda", "kiran@careflow.example", "9000000202"),
    ("Priya Menon", "priya@careflow.example", "9000000203"),
    ("Suresh Patil", "suresh@careflow.example", "9000000204"),
    ("Divya Reddy", "divya@careflow.example", "9000000205"),
]


async def _get_or_create_user(
    name: str,
    email: str,
    phone: str,
    role: Role,
    password_hash: str,
    department: str | None = None,
    registration: str | None = None,
) -> tuple[User, bool]:
    existing = await User.find_one(User.email == email)
    if existing:
        return existing, False
    user = User(
        name=name,
        email=email,
        phone=phone,
        password_hash=password_hash,
        role=role,
        department=department,
        medical_registration_number=registration,
    )
    await user.insert()
    return user, True


def _slot(day: datetime, index: int) -> tuple[datetime, datetime]:
    """UTC start/end of the Nth OPD slot on the given IST day."""
    opd_open = datetime.combine(day.date(), time.fromisoformat(OPD_START), tzinfo=IST)
    start = opd_open + timedelta(minutes=SLOT_MINUTES * index)
    return start.astimezone(UTC), (start + timedelta(minutes=SLOT_MINUTES)).astimezone(UTC)


async def _create_appointment(
    patient: User,
    doctor: User,
    day: datetime,
    index: int,
    status: AppointmentStatus,
    prescription: str | None = None,
) -> bool:
    start, end = _slot(day, index)
    exists = await Appointment.find_one(
        Appointment.doctor_id == doctor.id, Appointment.slot_start == start
    )
    if exists or patient.id is None or doctor.id is None or doctor.department is None:
        return False
    await Appointment(
        patient_id=patient.id,
        doctor_id=doctor.id,
        department=doctor.department,
        slot_start=start,
        slot_end=end,
        status=status,
        reason="Synthetic demo visit",
        prescription=prescription,
    ).insert()
    return True


async def seed() -> dict[str, int]:
    """Insert the synthetic dataset. Returns how many records of each kind were created."""
    settings = get_settings()
    if not settings.seed_default_password:
        raise RuntimeError("Set SEED_DEFAULT_PASSWORD in .env before seeding")
    password_hash = hash_password(settings.seed_default_password)
    created = {"users": 0, "schedules": 0, "appointments": 0, "hospital_config": 0}

    if await HospitalConfig.find_one() is None:
        timings = f"Mon–Sat {OPD_START}–{OPD_END}"
        await HospitalConfig(
            name="CareFlow Demo Hospital",
            address="12 Sample Road, Bengaluru (fictional)",
            opd_timings=timings,
            emergency_contact="080-0000-0000 (demo number)",
            departments=[
                Department(name=name, location=location, opd_timings=timings)
                for name, location in DEPARTMENTS
            ],
        ).insert()
        created["hospital_config"] = 1

    doctors: list[User] = []
    for name, email, phone, department, registration in DOCTORS:
        doctor, is_new = await _get_or_create_user(
            name, email, phone, Role.DOCTOR, password_hash, department, registration
        )
        doctors.append(doctor)
        created["users"] += is_new

    _, is_new = await _get_or_create_user(*ADMIN, Role.ADMIN, password_hash)
    created["users"] += is_new

    patients: list[User] = []
    for name, email, phone in PATIENTS:
        patient, is_new = await _get_or_create_user(name, email, phone, Role.PATIENT, password_hash)
        patients.append(patient)
        created["users"] += is_new

    for doctor in doctors:
        if doctor.id is None or doctor.department is None:
            continue
        for weekday in OPD_WEEKDAYS:
            exists = await Schedule.find_one(
                Schedule.doctor_id == doctor.id, Schedule.weekday == weekday
            )
            if exists:
                continue
            await Schedule(
                doctor_id=doctor.id,
                department=doctor.department,
                weekday=weekday,
                start_time=OPD_START,
                end_time=OPD_END,
                slot_minutes=SLOT_MINUTES,
            ).insert()
            created["schedules"] += 1

    today = datetime.now(IST)
    for index, patient in enumerate(patients):
        # One past visit each, so visit history has something to show
        past_doctor = doctors[index % len(doctors)]
        created["appointments"] += await _create_appointment(
            patient,
            past_doctor,
            today - timedelta(days=30 + index),
            index,
            AppointmentStatus.DONE,
            prescription=f"Synthetic prescription note #{index + 1} — demo data only",
        )
        # Everyone queues for the first doctor today, so the live queue is populated
        created["appointments"] += await _create_appointment(
            patient, doctors[0], today, index, AppointmentStatus.BOOKED
        )

    return created


async def main() -> None:
    await init_db()
    try:
        created = await seed()
    finally:
        close_db()
    for kind, count in created.items():
        print(f"{kind}: {count} created")


if __name__ == "__main__":
    asyncio.run(main())
