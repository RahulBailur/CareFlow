"""Output guardrails for anything CareBot says about symptoms.

The triage agent may name a department. It must never diagnose, prescribe or give a dose.
The system prompt asks for that; this module enforces it on the text actually produced,
whichever pipeline or model produced it: a violating reply is replaced, not edited.

Pattern-based, so it is a backstop and not a proof. English coverage is the strongest;
Hindi and Kannada cover medicine names, dose units and the commonest phrasings.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from agents.intent_classifier import Language, normalise


class Violation(StrEnum):
    DIAGNOSIS = "diagnosis"
    PRESCRIPTION = "prescription"
    DOSAGE = "dosage"


DISCLAIMERS: dict[Language, str] = {
    "en": "This is not a diagnosis. Please consult the doctor.",
    "hi": "यह कोई निदान नहीं है। कृपया डॉक्टर से सलाह लें।",
    "kn": "ಇದು ರೋಗನಿರ್ಣಯ ಅಲ್ಲ. ದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
}

SAFE_FALLBACKS: dict[Language, str] = {
    "en": (
        "I can't give medical advice. I can suggest which department to visit and help you "
        "book an appointment, and the doctor will advise you from there."
    ),
    "hi": (
        "मैं चिकित्सा सलाह नहीं दे सकता। मैं बता सकता हूँ कि किस विभाग में जाना है और "
        "अपॉइंटमेंट बुक करने में मदद कर सकता हूँ।"
    ),
    "kn": (
        "ನಾನು ವೈದ್ಯಕೀಯ ಸಲಹೆ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ. ಯಾವ ವಿಭಾಗಕ್ಕೆ ಹೋಗಬೇಕು ಎಂದು ಸೂಚಿಸಬಹುದು ಮತ್ತು "
        "ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕ್ ಮಾಡಲು ಸಹಾಯ ಮಾಡಬಹುದು."
    ),
}

# Named conditions. Symptoms (pain, fever, cough) are deliberately absent: repeating a
# symptom back to the patient is fine, naming what causes it is not.
_CONDITIONS = (
    r"\w+itis|\w+osis|\w+[ae]mia|\w+oma|infection|flu|influenza|covid|pneumonia|asthma|"
    r"diabet\w+|hypertension|migraine|angina|heart attack|stroke|cancer|tumou?r|fracture|"
    r"sprain|ulcer|allergy|eczema|psoriasis|malaria|dengue|typhoid|jaundice|tuberculosis|tb|"
    r"thyroid|kidney stones?|gallstones?|hernia|sinus\w*|food poisoning|acid reflux|gerd|"
    r"depression|anxiety disorder|uti|urinary tract infection|vertigo|sciatica|gout"
)

_MEDICINES = (
    r"paracetamol|acetaminophen|ibuprofen|aspirin|diclofenac|amoxicillin|azithromycin|"
    r"ciprofloxacin|cetirizine|metformin|omeprazole|pantoprazole|antibiotics?|painkillers?|"
    r"antacids?|steroids?|insulin|dolo|crocin|combiflam|calpol|"
    r"पैरासिटामोल|एंटीबायोटिक|ಪ್ಯಾರಸಿಟಮಾಲ್|ಆಂಟಿಬಯೋಟಿಕ್"
)

_PATTERNS: list[tuple[Violation, re.Pattern[str]]] = [
    # --- diagnosis ---
    (
        Violation.DIAGNOSIS,
        re.compile(
            r"\b(you are suffering from|you re suffering from|you have been diagnosed|"
            r"you are diagnosed|i diagnose|my diagnosis|your diagnosis is|"
            r"sounds like you have|looks like you have|it is (most )?(likely|probably) (a |an )?\w)"
        ),
    ),
    (
        Violation.DIAGNOSIS,
        re.compile(
            r"\b(you (may |might |could |probably |likely |most likely |seem to |appear to )?"
            r"(have|ve got|be having|are having)|it (could|may|might) be|"
            r"this (is|could be|may be|might be)|(sounds|looks|seems) like|"
            r"(sign|signs|symptom|symptoms|case) of|caused by|due to|"
            r"aapko|aap ko|tumhe|tumko|nimage|nimge)"
            rf"\b(\s+\w+){{0,3}}?\s+(?!diagnos|prognos)({_CONDITIONS})\b"
        ),
    ),
    (
        Violation.DIAGNOSIS,
        re.compile(r"आपको .{0,30}(बीमारी|रोग|संक्रमण|इन्फेक्शन) (है|हो सकता है|लगता है)"),
    ),
    (Violation.DIAGNOSIS, re.compile(r"ನಿಮಗೆ .{0,30}(ಕಾಯಿಲೆ|ರೋಗ|ಸೋಂಕು) (ಇದೆ|ಇರಬಹುದು)")),
    # --- prescription ---
    (Violation.PRESCRIPTION, re.compile(rf"(?<!\w)({_MEDICINES})(?!\w)")),
    (
        Violation.PRESCRIPTION,
        re.compile(
            r"\b(i (would )?(prescribe|recommend taking|suggest taking)|you should take|"
            r"you can take|try taking|start taking|"
            r"(take|have|pop|use|apply|try) (a|an|some|one|two|half|this|these|the) "
            r"(tablet|pill|capsule|medicine|medication|syrup|dose|drops|ointment|cream))"
        ),
    ),
    # Romanised Hindi and Kannada: "goli le lo", "dawai khao", "maatre tagolli"
    (
        Violation.PRESCRIPTION,
        re.compile(
            r"\b(goli|dawai?|davai|tablet|syrup) (\w+ )?(le|lo|lena|lijiye|kha|khao|khana|khaiye)\b|"
            r"\b(maa?tre|tablet|aushadhi|syrup) (\w+ )?(tago+l+i|togo+l+i|tegedukol+i|sevisi)\b"
        ),
    ),
    (Violation.PRESCRIPTION, re.compile(r"(दवा|गोली|टैबलेट|सिरप) (ले|लें|लीजिए|खा|खाएं|खाइए)")),
    (Violation.PRESCRIPTION, re.compile(r"(ಮಾತ್ರೆ|ಔಷಧಿ|ಔಷಧ|ಸಿರಪ್)\S* (ತೆಗೆದುಕೊಳ್ಳಿ|ತಗೊಳ್ಳಿ|ಸೇವಿಸಿ)")),
    # --- dosage ---
    (
        Violation.DOSAGE,
        re.compile(
            r"\b\d+(\s*\d+)?\s*(mg|mcg|ml|g|iu|milligrams?|tablets?|pills?|capsules?|drops|"
            r"teaspoons?|tsp|मिलीग्राम|एमजी|गोली|गोलियां|ಮಿಲಿಗ್ರಾಂ|ಮಾತ್ರೆ)(?!\w)"
        ),
    ),
    (
        Violation.DOSAGE,
        re.compile(
            r"\b(once|twice|thrice|\d+ times|two times|three times) (a|per|every) day\b|"
            r"\bevery \d+ hours\b|\b(before|after) (food|meals)\b|"
            r"\b(one|two|three|half|ek|do|teen|aadha|ondu|eradu) "
            r"(tablets?|pills?|capsules?|spoons?|teaspoons?|goli|goliyan|maa?tre)\b|"
            r"दिन में (एक|दो|तीन|\d+) बार|ದಿನಕ್ಕೆ (ಒಂದು|ಎರಡು|ಮೂರು|\d+) ಬಾರಿ"
        ),
    ),
]


@dataclass(frozen=True)
class GuardedReply:
    text: str
    disclaimer: str
    blocked: bool
    violations: tuple[Violation, ...]


def find_violations(text: str) -> tuple[Violation, ...]:
    clean = normalise(text)
    found = {violation for violation, pattern in _PATTERNS if pattern.search(clean)}
    return tuple(sorted(found))


def is_safe(text: str) -> bool:
    return not find_violations(text)


def guard_triage_reply(text: str, language: Language = "en") -> GuardedReply:
    """What may actually be shown or spoken for a triage turn.

    A reply with any violation is thrown away whole and the safe fallback is used instead.
    The disclaimer is returned on every reply, blocked or not, for the UI and TTS to append.
    """
    violations = find_violations(text)
    return GuardedReply(
        text=SAFE_FALLBACKS[language] if violations else text,
        disclaimer=DISCLAIMERS[language],
        blocked=bool(violations),
        violations=violations,
    )
