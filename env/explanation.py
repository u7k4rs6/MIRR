"""Lightweight explanation quality signal (keywords + structure), separate from fix reward."""


def score_diagnosis_evidence(
    text: str,
    *,
    target: str,
    failure_mode: str,
    root_cause: str,
    true_mode: str,
) -> float:
    """
    Returns a small auxiliary score (roughly 0..4) for logging / leaderboards.
    Not added to env reward by default — surfaced in info when episode ends.
    """
    if not text or not isinstance(text, str):
        return 0.0
    t = text.strip()
    if len(t) < 8:
        return 0.0
    low = t.lower()
    correct = target == root_cause and failure_mode == true_mode
    pts = 0.0
    if root_cause in low or root_cause.replace("-", " ") in low:
        pts += 1.25
    if true_mode in low or true_mode.replace("_", " ") in low:
        pts += 1.25
    for w in ("log", "metric", "because", "evidence", "latency", "cpu", "alert", "trace"):
        if w in low:
            pts += 0.15
    pts = min(pts, 4.0)
    return pts if correct else min(pts, 1.25) * 0.35
