"""
Medical Scope Detector (English Only)
Blocks personalized or case-specific medical requests.
"""

import re
from typing import Optional, List


class MedicalScopeDetector:
    """
    Rule-based detector for scope violations in English.
    Blocks requests that:
      - Ask for personal diagnosis or symptom triage
      - Ask for interpretation of personal test or imaging reports
      - Ask for personalized medication dosage or frequency
      - Request treatment plans or prescriptions tailored to the user
    """

    def __init__(self):
        # Precompile regex patterns for speed and clarity

        # Personalized intent (diagnosis / case-specific)
        self.personal_patterns = [
            r"\bdiagnose\s+me\b",
            r"\bdo\s+i\s+have\b",
            r"\bam\s+i\s+(sick|ill|infected|okay)\b",
            r"\bmy\s+(symptoms?|condition|case|diagnosis|disease)\b",
            r"\bbased\s+on\s+my\b",
            r"\bfor\s+my\s+case\b",
        ]

        # Lab or imaging report analysis
        self.report_patterns = [
            r"\bmy\s+(test|lab|blood|scan|x[- ]?ray|mri|ct|imaging|report)s?\b",
            r"\banalyze\s+my\b",
            r"\binterpret\s+my\b",
            r"\blook\s+at\s+my\b",
        ]

        # Drug dosage and frequency
        self.dosage_patterns = [
            r"\bhow\s+much\s+should\s+i\s+take\b",
            r"\bhow\s+many\s+pills\b",
            r"\bhow\s+often\s+should\s+i\b",
            r"\bdosage\s+for\s+me\b",
            r"\bwhat\s+dose\s+should\s+i\b",
            r"\b\d+(\.\d+)?\s*(mg|ml|mcg|μg|iu|units)\b",
            r"\b(q\d+h|qd|bid|tid|qid|qod|prn)\b",  # common frequency abbreviations
        ]

        # Personalized treatment plans or prescriptions
        self.plan_patterns = [
            r"\btreatment\s+plan\s+for\s+me\b",
            r"\bhow\s+to\s+treat\s+me\b",
            r"\bcreate\s+(a\s+)?plan\s+for\s+me\b",
            r"\bprescribe\b",
            r"\bprescription\s+for\s+me\b",
            r"\bwhat\s+medication\s+should\s+i\s+take\b",
            r"\brecommend\s+medication\s+for\s+me\b",
        ]

        # Whitelist for general educational queries
        self.whitelist_patterns = [
            r"\bwhat\s+is\b",
            r"\bdefinition\b",
            r"\boverview\b",
            r"\bgeneral\b",
            r"\bcommon\s+symptoms\b",
            r"\brisk\s+factors\b",
            r"\bprevention\b",
            r"\bcauses?\b",
            r"\btreatment\s+options\b",
            r"\bcomplications?\b",
        ]

        # Compile all regexes
        self.personal_re = [re.compile(p, re.I) for p in self.personal_patterns]
        self.report_re = [re.compile(p, re.I) for p in self.report_patterns]
        self.dosage_re = [re.compile(p, re.I) for p in self.dosage_patterns]
        self.plan_re = [re.compile(p, re.I) for p in self.plan_patterns]
        self.whitelist_re = [re.compile(p, re.I) for p in self.whitelist_patterns]

    def _is_whitelisted(self, text: str) -> bool:
        """Check if the text clearly asks for general educational info."""
        return any(p.search(text) for p in self.whitelist_re)

    def check(self, text: str) -> Optional[str]:
        """
        Check whether the text violates the chatbot's medical scope.

        Returns a warning message if personalized content is detected.
        """
        if not text:
            return None

        text = text.lower().strip()

        # Whitelist first (clear general queries)
        if self._is_whitelisted(text):
            return None

        # Track reasons for blocking
        reasons: List[str] = []

        # Match categories
        if any(p.search(text) for p in self.personal_re):
            reasons.append("personal diagnosis or case analysis")

        if any(p.search(text) for p in self.report_re):
            reasons.append("personal report or imaging interpretation")

        if any(p.search(text) for p in self.dosage_re):
            reasons.append("personalized dosage or medication frequency")

        if any(p.search(text) for p in self.plan_re):
            reasons.append("treatment plan or prescription request")

        # If any category is hit → block
        if reasons:
            return (
                "I cannot provide personalized medical advice. "
                "Specifically, I cannot:\n"
                "- Diagnose your condition or evaluate your symptoms\n"
                "- Interpret your personal test, imaging, or lab results\n"
                "- Recommend drug dosages or medication schedules for you\n"
                "- Create personalized treatment plans or prescriptions\n\n"
                "I can, however, provide general medical information such as "
                "definitions, common symptoms, causes, prevention methods, "
                "and when to seek professional care. "
                "For any personal health concerns, please consult a licensed physician."
            )

        return None


# Example usage (manual test)
if __name__ == "__main__":
    detector = MedicalScopeDetector()
    samples = [
        "Can you diagnose me?",
        "What is diabetes?",
        "Analyze my MRI report.",
        "How much ibuprofen should I take?",
        "Create a treatment plan for me.",
        "What are the common symptoms of flu?",
    ]
    for s in samples:
        blocked = detector.check(s)
        print(f"{s!r} → {'BLOCK' if blocked else 'PASS'}")
