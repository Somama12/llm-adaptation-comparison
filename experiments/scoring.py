"""Conservative task-specific scoring. Raw answers are retained for human audit.

This is a factual-content heuristic, not a general semantic judge. Ambiguous,
contradictory and unrecognized paraphrases should be manually reviewed.
"""
import re


def clean(text):
    return re.sub(r"\s+", " ", text.lower().replace("*", "")).strip()


def score_answer(qa, response, version):
    text = clean(response)
    expected = clean(qa[f"answer_{version}"])
    if qa["id"] == "changed_2":
        # Scope matters: 'mandatory for admins' does not mean mandatory for staff.
        optional = bool(re.match(r"^no[.! ]*$", text)) or bool(re.search(r"\b(optional|not (?:mandatory|required)|not obligated)\b", text))
        required = bool(re.search(r"\b(mandatory|required|must|yes)\b", text))
        standard = "standard" in text
        admin_only = "admin" in text and not re.search(r"all (?:employee|account|staff)|including standard", text)
        if version == "v1":
            return optional and not bool(re.search(r"mandatory for all|required for all", text))
        return required and not optional and not (admin_only and not standard)
    # Match full numeric tokens and units; 2 must not match 24, 12.99 or 2FA.
    values = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", expected)
    for value in values:
        if not re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", text):
            return False
    unit = re.search(r"\b(days|hours|gb|tb)\b", expected)
    if unit and not re.search(rf"\b{unit[1]}\b", text):
        return False
    if qa["changed"]:
        other = clean(qa["answer_v2" if version == "v1" else "answer_v1"])
        old_values = set(re.findall(r"\d+(?:\.\d+)?", other)) - set(values)
        if any(re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])\s*{unit[1] if unit else ''}\b", text) for v in old_values):
            return False
    return bool(values)
