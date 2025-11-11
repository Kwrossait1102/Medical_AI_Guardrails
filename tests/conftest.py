import os
import sys
import time
import pytest

# Make sure src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(scope="session")
def guard_pipeline():
    """
    Lightweight keyword-based guardrail pipeline for quick testing.
    Replace with real detectors later if desired.
    """
    emergency_kw = {"chest pain", "bleeding", "unconscious"}
    suicide_kw = {"suicide", "kill myself", "end my life"}
    dosage_kw = {"mg", "dosage", "take per day", "inject"}
    inj_kw = {"ignore previous", "disregard", "you are now", "act as", "bypass"}

    def _pipeline(text: str):
        t0 = time.time()
        txt = (text or "").lower()
        triggers, decision, reason = [], "PASS", "No guardrails triggered."

        if any(k in txt for k in inj_kw):
            triggers.append("injection_detector")
            decision, reason = "BLOCK", "Prompt injection detected."
        elif any(k in txt for k in emergency_kw):
            triggers.append("emergency_detector")
            decision, reason = "ASK", "Possible emergency symptoms."
        elif any(k in txt for k in suicide_kw):
            triggers.append("suicide_detector")
            decision, reason = "BLOCK", "Suicide risk detected."
        elif any(k in txt for k in dosage_kw):
            triggers.append("medical_scope_detector")
            decision, reason = "BLOCK", "Specific dosage request detected."

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "decision": decision,
            "reason": reason,
            "triggers": triggers,
            "latency_ms": latency_ms,
        }

    return _pipeline

@pytest.fixture(scope="session")
def results_recorder():
    """Collects all test outputs for later metrics."""
    store = {"rows": []}
    return store

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast rule-based tests")
    config.addinivalue_line("markers", "eval: full evaluation tests (may be slow)")
