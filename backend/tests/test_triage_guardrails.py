import pytest

from agents.guardrails import (
    DISCLAIMERS,
    SAFE_FALLBACKS,
    Violation,
    find_violations,
    guard_triage_reply,
    is_safe,
)

pytestmark = pytest.mark.guardrail

DIAGNOSES = [
    "You have a urinary tract infection.",
    "It sounds like you have the flu.",
    "You might have gastritis.",
    "This could be a migraine.",
    "You are suffering from high blood pressure.",
    "These are symptoms of dengue.",
    "It is most likely a muscle strain.",
    "Your chest pain may be caused by angina.",
    "You probably have a mild sinus infection.",
    "You may be having a heart attack.",
    "Sounds like an infection to me.",
    "aapko infection hai",
    "nimage skin allergy ide",
    "आपको फेफड़ों का संक्रमण हो सकता है",
    "ನಿಮಗೆ ಚರ್ಮದ ಸೋಂಕು ಇರಬಹುದು",
]

PRESCRIPTIONS = [
    "Take a paracetamol and rest.",
    "I recommend taking an antacid.",
    "You should take something for the fever.",
    "Dolo will bring the temperature down.",
    "You can take a painkiller tonight.",
    "Have one tablet in the morning.",
    "Apply the cream on the rash.",
    "ek goli le lo",
    "dawai kha lijiye",
    "एक गोली खाइए और आराम कीजिए",
    "ಈ ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಿ",
]

DOSAGES = [
    "500 mg should be enough.",
    "Give 5 ml of the syrup.",
    "Have it twice a day after meals.",
    "Repeat every 6 hours.",
    "2 tablets at night.",
    "Have one tablet in the morning.",
    "do goli subah",
    "दिन में दो बार लें",
    "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ",
]

# What a well-behaved triage or booking reply looks like. None of these may be blocked.
SAFE = [
    "For chest pain, the Cardiology department is the right place to start.",
    "A skin rash is usually seen by Dermatology. Shall I look for a slot?",
    "You have an appointment with Dr. Asha Rao at 10:15 am tomorrow.",
    "You have 3 patients ahead of you.",
    "I can't diagnose conditions, but I can suggest a department.",
    "Fever and cough are usually seen in General Medicine.",
    "Your son can be seen in Pediatrics on the first floor of Block B.",
    "This is the earliest slot available. It is on Friday.",
    "The doctor will decide on any medicine after examining you.",
    "The OPD is open from 9 am to 1 pm, Monday to Saturday.",
    "Dr. Vikram Shetty is running about 20 minutes late due to an emergency.",
    "If this is an emergency, please call the emergency number right away.",
    "Dr. Meera Iyer sees arthritis patients in Orthopedics.",
    "You have a new slot due to a cancellation.",
    "It looks like Dr. Rao is fully booked on Friday.",
    "aapko Cardiology department mein dikhana chahiye",
    "nimage Friday 10 gantege appointment ide",
    "सीने में दर्द के लिए कार्डियोलॉजी विभाग में दिखाइए।",
    "ಜ್ವರ ಮತ್ತು ಕೆಮ್ಮಿಗೆ ಜನರಲ್ ಮೆಡಿಸಿನ್ ವಿಭಾಗಕ್ಕೆ ಹೋಗಿ.",
]


@pytest.mark.parametrize("text", DIAGNOSES)
def test_a_diagnosis_is_caught(text: str) -> None:
    assert Violation.DIAGNOSIS in find_violations(text)


@pytest.mark.parametrize("text", PRESCRIPTIONS)
def test_a_prescription_is_caught(text: str) -> None:
    assert Violation.PRESCRIPTION in find_violations(text)


@pytest.mark.parametrize("text", DOSAGES)
def test_a_dosage_is_caught(text: str) -> None:
    assert Violation.DOSAGE in find_violations(text)


@pytest.mark.parametrize("text", SAFE)
def test_department_routing_and_booking_replies_are_not_blocked(text: str) -> None:
    assert find_violations(text) == ()


@pytest.mark.parametrize("language", ["en", "hi", "kn"])
def test_a_violating_reply_is_replaced_whole_by_the_safe_fallback(language: str) -> None:
    unsafe = "You have bronchitis. Take 500 mg of amoxicillin twice a day."

    reply = guard_triage_reply(unsafe, language)  # type: ignore[arg-type]

    assert reply.blocked
    assert reply.text == SAFE_FALLBACKS[language]  # type: ignore[index]
    assert "bronchitis" not in reply.text and "amoxicillin" not in reply.text
    assert set(reply.violations) == {
        Violation.DIAGNOSIS,
        Violation.PRESCRIPTION,
        Violation.DOSAGE,
    }


def test_a_safe_reply_passes_through_unchanged() -> None:
    text = "For knee pain, Orthopedics is the right department."

    reply = guard_triage_reply(text)

    assert not reply.blocked
    assert reply.text == text
    assert reply.violations == ()


@pytest.mark.parametrize("language", ["en", "hi", "kn"])
@pytest.mark.parametrize("text", ["Cardiology is the right department.", "You have pneumonia."])
def test_every_triage_reply_carries_the_disclaimer(language: str, text: str) -> None:
    reply = guard_triage_reply(text, language)  # type: ignore[arg-type]

    assert reply.disclaimer == DISCLAIMERS[language]  # type: ignore[index]
    assert reply.disclaimer.strip()


@pytest.mark.parametrize("text", [*SAFE_FALLBACKS.values(), *DISCLAIMERS.values()])
def test_the_fallback_and_disclaimer_text_pass_the_filter_themselves(text: str) -> None:
    assert is_safe(text)


def test_the_filter_ignores_case_and_punctuation() -> None:
    assert not is_safe("YOU HAVE... PNEUMONIA!!")
    assert not is_safe("take 500mg")
