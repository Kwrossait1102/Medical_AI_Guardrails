import json

class EmergencyDetector:
    """
    A simple rule-based emergency detector that checks whether
    user input contains any predefined emergency-related keywords.
    """

    def __init__(self, keywords_path: str):
        """
        Initialize the EmergencyDetector with a given JSON file
        containing keyword lists.

        Args:
            keywords_path (str): Path to the JSON file containing emergency keywords.
        """
        self.keywords = self._load_keywords(keywords_path)
    
    def _load_keywords(self, path: str):
        """
        Load and flatten emergency-related keywords from a JSON file.

        The JSON file should follow this structure:
        {
            "source1": {
                "category1": ["word1", "word2"],
                "category2": ["word3", "word4"]
            },
            "source2": { ... },
            "meta": { ... }   # Optional metadata, will be ignored
        }

        Args:
            path (str): Path to the JSON file.

        Returns:
            set: A set of lowercase keywords.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        flat = set()
        for source, groups in data.items():
            if source == "meta":
                continue
            for _, words in groups.items():
                for w in words:
                    flat.add(w.strip().lower())
        return flat
    
    def is_emergency(self, text: str) -> bool:
        """
        Check whether the given text contains any emergency keyword.

        Args:
            text (str): The user input text.

        Returns:
            bool: True if any keyword is found, False otherwise.
        """
        if not text:
            return False
        t = text.lower()
        return any(kw in t for kw in self.keywords)
    
    def check(self, user_input: str):
        """
        Return a warning message if the input indicates a possible emergency.

        Args:
            user_input (str): The input text to analyze.

        Returns:
            str | None: Emergency warning message if matched, otherwise None.
        """
        if self.is_emergency(user_input):
            return (
                "This may be a medical emergency!\n"
                "Please take the following actions immediately:\n"
                "Call emergency services (Europe: 112, USA: 911)\n"
                "Go to the nearest hospital or emergency room.\n"
                "I cannot handle medical emergencies. Seek help right now!"
            )
        return None


